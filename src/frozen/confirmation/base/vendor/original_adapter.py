from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from basicsr.data.transforms import paired_random_crop, random_augmentation
from basicsr.utils import img2tensor


class LOLv2FrozenSplitPairedDataset(Dataset):
    """LOLv2-real paired dataset constrained to a frozen group-aware manifest."""

    _EXPECTED_COUNTS = {"train": 551, "validation": 138}
    _FORBIDDEN_PATH_TOKENS = ("\\test\\", "lol-v1", "lsrw", "acdc")

    def __init__(self, opt: dict[str, Any]) -> None:
        self.opt = opt
        self.phase = opt["phase"]
        self.frozen_split = opt["frozen_split"]
        if self.phase not in {"train", "val"}:
            raise ValueError(f"Unsupported BasicSR phase: {self.phase}")
        if self.frozen_split not in self._EXPECTED_COUNTS:
            raise ValueError(f"Unsupported frozen split: {self.frozen_split}")
        expected_phase = "train" if self.frozen_split == "train" else "val"
        if self.phase != expected_phase:
            raise ValueError(
                f"phase={self.phase} is incompatible with frozen_split={self.frozen_split}"
            )

        self.low_root = Path(opt["low_root"]).resolve()
        self.gt_root = Path(opt["gt_root"]).resolve()
        self.manifest_path = Path(opt["split_manifest"]).resolve()
        self._validate_roots()
        self._validate_manifest_hash()
        self.records = self._read_frozen_records()
        expected_count = int(opt.get("expected_count", self._EXPECTED_COUNTS[self.frozen_split]))
        if expected_count != self._EXPECTED_COUNTS[self.frozen_split]:
            raise ValueError(
                f"expected_count must be {self._EXPECTED_COUNTS[self.frozen_split]} "
                f"for frozen_split={self.frozen_split}, got {expected_count}"
            )
        if len(self.records) != expected_count:
            raise RuntimeError(
                f"Frozen split count mismatch for {self.frozen_split}: "
                f"expected {expected_count}, found {len(self.records)}"
            )

        self.geometric_augs = bool(opt.get("geometric_augs", False))
        self.gt_size = int(opt["gt_size"]) if self.phase == "train" else None
        self.scale = int(opt.get("scale", 1))
        if self.gt_size is not None and self.gt_size <= 0:
            raise ValueError("gt_size must be positive")

    def _validate_roots(self) -> None:
        for root, role, expected_tail in (
            (self.low_root, "low", ("train", "low")),
            (self.gt_root, "gt", ("train", "normal")),
        ):
            if not root.is_dir():
                raise FileNotFoundError(f"{role}_root is not a directory: {root}")
            normalized = str(root).replace("/", "\\").casefold()
            if any(token in normalized for token in self._FORBIDDEN_PATH_TOKENS):
                raise ValueError(f"Forbidden dataset path for {role}: {root}")
            tail = tuple(part.casefold() for part in root.parts[-2:])
            if tail != expected_tail:
                raise ValueError(f"Unexpected {role}_root; expected ...\\{expected_tail[0]}\\{expected_tail[1]}: {root}")

    def _validate_manifest_hash(self) -> None:
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Frozen split manifest not found: {self.manifest_path}")
        expected_hash = self.opt.get("split_manifest_sha256")
        if expected_hash is None:
            raise ValueError("split_manifest_sha256 is required")
        actual_hash = hashlib.sha256(self.manifest_path.read_bytes()).hexdigest().upper()
        if actual_hash != str(expected_hash).upper():
            raise RuntimeError(
                f"Frozen split manifest hash mismatch: expected {expected_hash}, got {actual_hash}"
            )

    def _read_frozen_records(self) -> list[dict[str, str]]:
        with self.manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        required_columns = {"sample_id", "group_id", "split"}
        if not rows or not required_columns.issubset(rows[0]):
            raise ValueError(f"Malformed frozen split manifest: {self.manifest_path}")

        sample_ids = [row["sample_id"] for row in rows]
        if len(sample_ids) != 689 or len(set(sample_ids)) != len(sample_ids):
            raise RuntimeError("Frozen manifest must contain 689 unique sample IDs")
        split_groups: dict[str, set[str]] = defaultdict(set)
        group_splits: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            split = row["split"]
            if split not in self._EXPECTED_COUNTS:
                raise ValueError(f"Unexpected split label in manifest: {split}")
            sample_id = row["sample_id"]
            if len(sample_id) != 5 or not sample_id.isdigit():
                raise ValueError(f"Unexpected sample_id in frozen manifest: {sample_id}")
            group_id = row["group_id"].upper()
            split_groups[split].add(group_id)
            group_splits[group_id].add(split)
        if split_groups["train"] & split_groups["validation"]:
            raise RuntimeError("Frozen group-aware manifest has train/validation group overlap")
        if any(len(assigned_splits) != 1 for assigned_splits in group_splits.values()):
            raise RuntimeError("Identical GT-content group crosses frozen split")

        records: list[dict[str, str]] = []
        seen_ids: set[str] = set()
        for row in rows:
            if row["split"] != self.frozen_split:
                continue
            sample_id = row["sample_id"]
            if sample_id in seen_ids:
                raise RuntimeError(f"Duplicate sample_id within frozen split: {sample_id}")
            seen_ids.add(sample_id)
            low_path = (self.low_root / f"low{sample_id}.png").resolve()
            gt_path = (self.gt_root / f"normal{sample_id}.png").resolve()
            if low_path.parent != self.low_root or gt_path.parent != self.gt_root:
                raise RuntimeError("Pair path escaped the permitted LOLv2-real Train roots")
            if not low_path.is_file() or not gt_path.is_file():
                raise FileNotFoundError(
                    f"Missing frozen pair for sample_id={sample_id}: low={low_path.exists()} gt={gt_path.exists()}"
                )
            records.append(
                {"sample_id": sample_id, "lq_path": str(low_path), "gt_path": str(gt_path)}
            )
        return records

    @staticmethod
    def _decode_bgr(path: str) -> Any:
        encoded = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise RuntimeError(f"Failed to decode RGB image: {path}")
        return image.astype("float32") / 255.0

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        img_lq = self._decode_bgr(record["lq_path"])
        img_gt = self._decode_bgr(record["gt_path"])
        if img_lq.shape != img_gt.shape:
            raise RuntimeError(
                f"Paired image dimensions differ for sample_id={record['sample_id']}: "
                f"{img_lq.shape} versus {img_gt.shape}"
            )
        if self.phase == "train":
            img_gt, img_lq = paired_random_crop(
                img_gt, img_lq, self.gt_size, self.scale, record["gt_path"]
            )
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        return {
            "lq": img_lq,
            "gt": img_gt,
            "lq_path": record["lq_path"],
            "gt_path": record["gt_path"],
            "sample_id": record["sample_id"],
        }

    def __len__(self) -> int:
        return len(self.records)
