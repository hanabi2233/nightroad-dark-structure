"""Isolated scratch training runner. Smoke never constructs validation datasets."""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
SOURCE = PROJECT / "third_party/Retinexformer_1e9a0efce4b306b6701b824768370ff26066c32a"
DESIGN = PROJECT / "experiments/PAPER_B_SMALL_METHOD_DESIGN_V1/METHOD_DESIGN_AND_MINIMAL_PILOT.md"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def append_event(out, value):
    with (out / "attempts.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False) + "\n")
    print(json.dumps(value, ensure_ascii=False), flush=True)


def batch_schedule(count, attempts, seed):
    import torch
    stream, epoch = [], 0
    while len(stream) < attempts * 8:
        stream.extend(torch.randperm(count, generator=torch.Generator().manual_seed(seed + epoch)).tolist())
        epoch += 1
    return [stream[i*8:(i+1)*8] for i in range(attempts)]


def data_seed(attempt):
    return 20260910 + 100003 * (attempt + 1)


@contextlib.contextmanager
def data_rng(attempt):
    import numpy as np
    import torch
    py, nu, tc = random.getstate(), np.random.get_state(), torch.get_rng_state()
    try:
        random.seed(data_seed(attempt))
        np.random.seed(data_seed(attempt) % (2**32))
        # CPU default generator only; optimizer and CUDA RNG are not used for augmentation.
        torch.random.default_generator.manual_seed(data_seed(attempt))
        yield
    finally:
        random.setstate(py)
        np.random.set_state(nu)
        torch.set_rng_state(tc)


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def settings(variant):
    return json.loads((ROOT / f"config_{variant}.json").read_text(encoding="utf-8"))


def setup():
    import numpy as np
    import torch
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=False)
    random.seed(20260910)
    np.random.seed(20260910)
    torch.manual_seed(20260910)
    if not torch.cuda.is_available():
        raise RuntimeError("NO_CUDA: no CPU/server fallback permitted")
    torch.cuda.manual_seed_all(20260910)
    sys.path.insert(0, str(SOURCE))
    sys.path.insert(0, str(PROJECT / "data/adapters"))


def make_dataset(config):
    from lolv2_frozen_split_paired_dataset import LOLv2FrozenSplitPairedDataset
    # Train-only adapter constructs and decodes only records with frozen_split=train.
    dataset = LOLv2FrozenSplitPairedDataset(config["dataset"])
    if len(dataset) != 551 or dataset.frozen_split != "train":
        raise RuntimeError("Frozen train identity mismatch")
    return dataset


def batch(dataset, indices, attempt):
    import torch
    from basicsr.models.image_restoration_model import Mixing_Augment
    with data_rng(attempt):
        samples = [dataset[i] for i in indices]
        x = torch.stack([s["lq"] for s in samples])
        y = torch.stack([s["gt"] for s in samples])
        mixer = Mixing_Augment(1.2, True, torch.device("cpu"))
        y, x = mixer(y, x)
    if x.shape != (8, 3, 256, 256) or y.shape != x.shape:
        raise RuntimeError(f"Shape contract violated: {x.shape}, {y.shape}")
    ids = [s["sample_id"] for s in samples]
    return x, y, {"sample_ids": ids, "input_tensor_sha256": tensor_sha(x), "gt_tensor_sha256": tensor_sha(y),
                  "augmentation_seed": data_seed(attempt)}


def protected_identity():
    files = {
        "design": DESIGN,
        "original_entry": PROJECT / "scripts/r3h_run2_stage1_entry.py",
        "architecture": SOURCE / "basicsr/models/archs/RetinexFormer_arch.py",
        "adapter": PROJECT / "data/adapters/lolv2_frozen_split_paired_dataset.py",
        "original_config": PROJECT / "configs/R3H_RUN2_STAGE1.yaml",
    }
    value = {name: {"path": str(path), "sha256": digest(path)} for name, path in files.items()}
    if value["architecture"]["sha256"] != "1567C89D55285A3C1A4D4CA33F288B1FD98DC6B6BBDF971EDA02D745F9087631":
        raise RuntimeError("Frozen architecture SHA conflict")
    return value


def test_gate(test_result):
    result = json.loads(test_result.read_text(encoding="utf-8"))
    if result["status"] != "PASS":
        raise RuntimeError("Numerical tests have not passed")
    for name, expected in result["code_sha256"].items():
        if digest(ROOT / name).lower() != expected.lower():
            raise RuntimeError(f"Code changed after numerical tests: {name}")


def save_trace(out, name, tensors, adam):
    import torch
    payload = {k: v.detach().cpu() for k, v in tensors.items()}
    payload["adam_first_moment"] = adam.m.detach().cpu()
    payload["adam_second_moment"] = adam.v.detach().cpu()
    payload["adam_step_count"] = adam.step_count
    payload["role"] = "ENGINEERING_TRACE_NOT_A_RESUMABLE_MODEL"
    with (out / name).open("xb") as f:
        torch.save(payload, f)
    return name


def worker(args, out):
    import torch
    import core
    started = time.time()
    result = {"status": "RUNNING", "mode": args.mode, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "validation_images_read": False, "test_images_read": False, "historical_weights_read": False,
              "formal_pilot_started": args.mode == "pilot", "variants": {}, "attempts_total": 0, "committed_total": 0}
    try:
        test_gate(Path(args.tests).resolve())
        identity = protected_identity()
        setup()
        from basicsr.models.archs.RetinexFormer_arch import RetinexFormer
        environment = {"executable": sys.executable, "python": sys.version, "torch": torch.__version__,
                       "cuda_build": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                       "vram_bytes": torch.cuda.get_device_properties(0).total_memory,
                       "cudnn": torch.backends.cudnn.version(), "deterministic": True, "amp": False,
                       "tf32": False, "batch": 8, "crop": 256, "wall_time_budget_s": args.max_seconds}
        try:
            environment["nvidia_smi"] = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                text=True, timeout=10).strip()
        except Exception as e:
            environment["nvidia_smi_error"] = str(e)
        write_json(out / "environment.json", environment)
        write_json(out / "protected_identity_before.json", identity)
        variants = list("ABC") if args.mode == "smoke" else [args.variant]
        first_init, first_batch_hashes = None, {}
        for variant in variants:
            setup()  # same scratch seed, no state or checkpoint transfer
            config = settings(variant)
            if config["variant"] != variant or config["total_attempts"] != 2000 or config["l1_warmup"] != 500:
                raise RuntimeError("Configuration contract mismatch")
            write_json(out / f"config_{variant}.json", config)
            dataset = make_dataset(config)
            attempts = 6 if args.mode == "smoke" else 2000
            schedule = batch_schedule(551, attempts, config["seed"])
            write_json(out / f"selected_train_{variant}.json",
                       [[dataset.records[i]["sample_id"] for i in row] for row in schedule])
            model = RetinexFormer(in_channels=3, out_channels=3, n_feat=40, stage=1, num_blocks=[1, 2, 2]).train()
            initial = core.flatten([p.detach() for p in model.parameters()])
            init_hash = tensor_sha(initial)
            if first_init is None:
                first_init = init_hash
            elif init_hash != first_init:
                raise RuntimeError("Shared scratch initialization mismatch")
            write_json(out / f"parameter_map_{variant}.json",
                       {"initialization_tensor_sha256": init_hash, "parameter_count": initial.numel(),
                        "parameters": [{"name": n, "shape": list(p.shape)} for n, p in model.named_parameters()]})
            model = model.cuda()
            adam = core.AdamCandidate(core.flatten([p.detach() for p in model.parameters()]))
            rng = random.Random(config["seed"])
            vr = {"attempted": 0, "committed": 0, "rows": [], "initialization_sha256": init_hash}
            result["variants"][variant] = vr
            for index, indices in enumerate(schedule):
                if time.time() - started >= args.max_seconds:
                    raise TimeoutError("Resource cap reached before next attempt")
                # Isolated engineering branch coverage: 2 L1 + 4 four-target, not 500 warmup.
                four = variant != "A" and index >= (2 if args.mode == "smoke" else 500)
                branch = "four" if four else "L1"
                x, y, sample = batch(dataset, indices, index)
                key = index
                pair_hash = (sample["input_tensor_sha256"], sample["gt_tensor_sha256"])
                if key in first_batch_hashes and pair_hash != first_batch_hashes[key]:
                    raise RuntimeError("ABC input/augmentation tensor identity mismatch")
                first_batch_hashes[key] = pair_hash
                x, y = x.cuda(), y.cuda()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                vr["attempted"] += 1
                result["attempts_total"] += 1
                append_event(out, {"event": "ATTEMPT_START", "variant": variant, "attempt": index+1,
                                   "branch": branch, **sample})
                start = time.perf_counter()
                try:
                    detail, tensors = core.training_update(model, x, y, adam, variant, four, rng, core.lr_at(index))
                except core.NumericalContractError as error:
                    if hasattr(error, "tensors"):
                        save_trace(out, f"failed_{variant}_{index+1:04d}.pt", error.tensors, adam)
                    append_event(out, {"event": "NUMERICAL_FAILURE", "variant": variant, "attempt": index+1,
                                       "error": str(error), "detail": getattr(error, "update_detail", {})})
                    raise
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                vr["committed"] += 1
                result["committed_total"] += 1
                row = {"event": "COMMIT", "variant": variant, "attempt": index+1, "branch": branch,
                       "compute_seconds": elapsed, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                       "peak_reserved_bytes": torch.cuda.max_memory_reserved(), **detail}
                # Every smoke update retains full gradient/step tensors; pilot checkpoints are separate.
                if args.mode == "smoke":
                    t = time.perf_counter()
                    row["trace"] = save_trace(out, f"trace_{variant}_{index+1:04d}.pt", tensors, adam)
                    row["trace_io_seconds"] = time.perf_counter() - t
                append_event(out, row)
                vr["rows"].append(row)
                del x, y, tensors
            if args.mode == "pilot":
                with (out / f"final_{variant}_iter2000.pth").open("xb") as f:
                    torch.save({"params": model.state_dict(), "role": "NEW_SCRATCH_PILOT_FIXED_ENDPOINT",
                                "variant": variant, "attempt": 2000, "initialization_sha256": init_hash}, f)
            del model, adam
            torch.cuda.empty_cache()
        if protected_identity() != identity:
            raise RuntimeError("Protected source identity changed during execution")
        write_json(out / "protected_identity_after.json", identity)
        result["status"] = "SMOKE_PASS" if args.mode == "smoke" else "TRAINING_COMPLETE_VALIDATION_NOT_RUN"
    except Exception as e:
        result["status"] = "BLOCKED"
        result["error_type"] = type(e).__name__
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        print(result["traceback"], flush=True)
    result["elapsed_wall_seconds"] = time.time() - started
    result["ended"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    for variant, value in result["variants"].items():
        value["timing"] = {}
        for branch in ("L1", "four"):
            rows = [r for r in value["rows"] if r["branch"] == branch]
            measured = rows[1:]  # exclude first observation in each branch as warmup
            value["timing"][branch] = {"warmup_excluded": min(1, len(rows)), "n": len(measured),
                "mean_s": statistics.mean(r["compute_seconds"] for r in measured) if measured else None,
                "median_s": statistics.median(r["compute_seconds"] for r in measured) if measured else None,
                "max_s": max((r["compute_seconds"] for r in measured), default=None)}
    write_json(out / "result.json", result)
    print(json.dumps({"status": result["status"], "attempted": result["attempts_total"],
                      "committed": result["committed_total"], "error": result.get("error")}, ensure_ascii=False), flush=True)
    return 0 if result["status"] != "BLOCKED" else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["smoke", "pilot"])
    parser.add_argument("--variant", choices=list("ABC"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--tests", required=True)
    parser.add_argument("--max-seconds", type=int, required=True)
    parser.add_argument("--confirm-formal-training", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    if ROOT not in out.parents:
        parser.error("Output must be a new child directory of this implementation")
    if args.max_seconds <= 0 or (args.mode == "smoke" and args.max_seconds > 1800):
        parser.error("Smoke cumulative subprocess wall cap must be <=1800 seconds")
    if args.mode == "pilot" and (not args.confirm_formal_training or args.variant is None):
        parser.error("Pilot requires explicit confirmation, one variant, and new resource cap")
    if args.worker:
        return worker(args, out)
    out.mkdir(parents=True, exist_ok=False)
    # Parent watchdog bounds even blocking CUDA operations; includes CPU/IO, hence conservative.
    cmd = [sys.executable, "-B", str(Path(__file__).resolve()), *sys.argv[1:], "--worker"]
    write_json(out / "launch.json", {"command": cmd, "max_seconds": args.max_seconds,
                                   "launch_time": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    with (out / "console.log").open("x", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(PROJECT))
        try:
            return proc.wait(timeout=args.max_seconds)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            write_json(out / "watchdog_stop.json", {"status": "RESOURCE_CAP_STOP", "seconds": args.max_seconds,
                                                  "no_retry": True, "pid": proc.pid})
            return 124


if __name__ == "__main__":
    sys.exit(main())
