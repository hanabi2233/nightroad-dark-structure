"""Fixed-last Retinexformer inference only; no training, metric, or test access.

Each real-data run requires an explicit manifest and creates a fresh output
directory. The separate --synthetic-only mode checks engineering on an
in-memory zero image and never opens a dataset image.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


EXPECTED_CHECKPOINTS = {
    "A": "3210578DBCD248B78D7A68E3A60D38C5CAD4F39F7596FBF237C9B20B837228AE",
    "SUM4": "48214956CF3C5C19F5AF3DE65DF53AEF135D21E0F8F5D45A7F55E623F9F7ED70",
    "B": "2C85C7EABE873E01AFEBBB2CC1798AD3BDDC04EA58EC8989FDF44C70F2B06A89",
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_model(architecture, checkpoint, variant):
    if sha256(checkpoint) != EXPECTED_CHECKPOINTS[variant]:
        raise ValueError("CHECKPOINT_SHA_MISMATCH")
    spec = importlib.util.spec_from_file_location("road_fixed_arch", architecture)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.RetinexFormer(
        in_channels=3, out_channels=3, n_feat=40, stage=1, num_blocks=[1, 2, 2]
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or len(state) != 122:
        raise ValueError("CHECKPOINT_STATE_FORMAT")
    model.load_state_dict(state, strict=True)
    return model.to("cuda").eval().requires_grad_(False)


def predict(model, image):
    x = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    h, w = x.shape[:2]
    tensor = torch.from_numpy(x).permute(2, 0, 1)[None].to("cuda")
    with torch.inference_mode():
        result = model(F.pad(tensor, (0, (-w) % 4, 0, (-h) % 4), mode="reflect"))
        result = result[..., :h, :w].clamp(0, 1)
    if result.shape != (1, 3, h, w) or not bool(torch.isfinite(result).all()):
        raise ValueError("INVALID_MODEL_OUTPUT")
    array = np.rint(result[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def checked_relative(value):
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError("UNSAFE_RELATIVE_PATH")
    return path


def read_members(manifest):
    with open(manifest, newline="", encoding="utf-8-sig") as stream:
        members = list(csv.DictReader(stream))
    required = {"sample_id", "sequence", "input_relative", "input_sha256", "width", "height"}
    if len(members) != 106 or any(not required.issubset(row) for row in members):
        raise ValueError("MANIFEST_SCHEMA_OR_COUNT")
    if len({row["sample_id"] for row in members}) != 106:
        raise ValueError("DUPLICATE_SAMPLE_ID")
    for row in members:
        checked_relative(row["input_relative"])
        if row["sequence"] not in {"GOPR0351", "GOPR0356"}:
            raise ValueError("UNEXPECTED_SEQUENCE")
        if not row["sample_id"].startswith(row["sequence"] + "_"):
            raise ValueError("SAMPLE_SEQUENCE_MISMATCH")
        if len(row["input_sha256"]) != 64:
            raise ValueError("INVALID_INPUT_HASH")
        if int(row["width"]) <= 0 or int(row["height"]) <= 0:
            raise ValueError("INVALID_DIMENSIONS")
    if {s: sum(r["sequence"] == s for r in members) for s in ("GOPR0351", "GOPR0356")} != {
        "GOPR0351": 56, "GOPR0356": 50
    }:
        raise ValueError("SEQUENCE_MEMBERSHIP")
    return members


def run(args):
    if args.variant not in EXPECTED_CHECKPOINTS:
        raise ValueError("UNKNOWN_VARIANT")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED")
    start = time.monotonic()
    model = load_model(args.architecture, args.checkpoint, args.variant)
    if args.synthetic_only:
        image = Image.fromarray(np.zeros((65, 67, 3), dtype=np.uint8), mode="RGB")
        output = predict(model, image)
        if output.size != (67, 65):
            raise ValueError("SYNTHETIC_SHAPE")
        print(json.dumps({"status": "SYNTHETIC_PASS", "variant": args.variant,
                          "shape": [65, 67, 3], "gpu": torch.cuda.get_device_name(0),
                          "elapsed_seconds": time.monotonic() - start}), flush=True)
        return

    if not args.manifest or not args.input_root or not args.output or not args.cohort:
        raise ValueError("REAL_RUN_ARGUMENTS_REQUIRED")
    if args.cohort not in {"synthetic", "real_night"}:
        raise ValueError("UNKNOWN_COHORT")
    members = read_members(args.manifest)
    input_root = Path(args.input_root).resolve(strict=True)
    output_root = Path(args.output)
    if output_root.exists():
        raise FileExistsError("OUTPUT_MUST_BE_FRESH")
    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "predictions").mkdir()
    rows = []
    try:
        for row in members:
            source = (input_root / checked_relative(row["input_relative"])).resolve(strict=True)
            if input_root not in source.parents or not source.is_file():
                raise ValueError("INPUT_OUTSIDE_ROOT")
            if sha256(source) != row["input_sha256"].upper():
                raise ValueError("INPUT_SHA_MISMATCH: " + row["sample_id"])
            with Image.open(source) as image:
                if image.size != (int(row["width"]), int(row["height"])):
                    raise ValueError("INPUT_DIMENSIONS: " + row["sample_id"])
                result = predict(model, image)
            target = output_root / "predictions" / (row["sample_id"] + ".png")
            result.save(target)
            rows.append({"sample_id": row["sample_id"], "sequence": row["sequence"],
                         "input_sha256": row["input_sha256"].upper(),
                         "output_sha256": sha256(target), "output_bytes": target.stat().st_size})
            if len(rows) % 10 == 0:
                print(f"{args.cohort} {args.variant}: {len(rows)}/106", flush=True)
        status = "COMPLETE"
    except Exception as error:
        status = "FAILED_INCOMPLETE"
        (output_root / "FAILURE.json").write_text(
            json.dumps({"error": str(error), "completed": len(rows)}, indent=2), encoding="utf-8"
        )
        raise
    finally:
        with open(output_root / "OUTPUT_MANIFEST.csv", "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["sample_id", "sequence", "input_sha256",
                                                          "output_sha256", "output_bytes"])
            writer.writeheader()
            writer.writerows(rows)
        (output_root / "SUMMARY.json").write_text(json.dumps({
            "status": status, "cohort": args.cohort, "variant": args.variant,
            "checkpoint_sha256": EXPECTED_CHECKPOINTS[args.variant],
            "architecture_sha256": sha256(args.architecture), "manifest_sha256": sha256(args.manifest),
            "completed": len(rows), "expected": 106, "gpu": torch.cuda.get_device_name(0),
            "elapsed_seconds": time.monotonic() - start
        }, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--variant", choices=EXPECTED_CHECKPOINTS, required=True)
    parser.add_argument("--synthetic-only", action="store_true")
    parser.add_argument("--cohort")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.synthetic_only and not all([args.cohort, args.manifest, args.input_root, args.output]):
        parser.error("real runs require cohort, manifest, input-root and output")
    run(args)


if __name__ == "__main__":
    main()
