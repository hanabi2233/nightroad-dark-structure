"""Authorized, no-resume, fixed-last runner for the six V11 component deletions."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import sys
import time
import traceback
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
EXPERIMENT = "V11_THREE_TASK_COMPONENT_SUPPLEMENT_20261016"
VARIANTS = ("MERGED_DROP_S", "MERGED_DROP_M")
SEEDS = (20261004, 20261005, 20261006)
CALENDAR_DEADLINE = 1792296000.0  # 2026-10-18 12:00:00 Asia/Shanghai
MAX_BUDGET = 18 * 3600


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_sources(package, merged_dir):
    package = Path(package).resolve()
    merged_dir = Path(merged_dir).resolve()
    sys.path.insert(0, str(package))
    import common
    import core
    import train as sealed_train
    if Path(common.__file__).resolve() != package / "common.py":
        raise ValueError("UNEXPECTED_COMMON_IMPORT")
    if Path(core.__file__).resolve() != package / "base/core.py":
        raise ValueError("UNEXPECTED_CORE_IMPORT")
    if Path(sealed_train.__file__).resolve() != package / "train.py":
        raise ValueError("UNEXPECTED_TRAIN_IMPORT")
    spec = importlib.util.spec_from_file_location("weekend_sealed_merged_core", merged_dir / "merged_core.py")
    merged_core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(merged_core)
    import component_core
    if Path(component_core.__file__).resolve() != HERE / "component_core.py":
        raise ValueError("UNEXPECTED_COMPONENT_IMPORT")
    return common, core, sealed_train, merged_core, component_core


def authorize(args, common, phase, allowed_runs):
    auth = common.read(common.regular(args.authorization))
    if auth.get("approved") is not True:
        raise ValueError("EXPLICIT_COMPONENT_APPROVAL_REQUIRED")
    output = Path(args.output).resolve()
    package = Path(args.package).resolve()
    merged = Path(args.merged_dir).resolve()
    expected = {
        "experiment": EXPERIMENT, "approved": True, "phase": phase,
        "protocol_sha256": digest(common.regular(args.protocol)),
        "sealed_manifest_sha256": digest(common.regular(package / "MANIFEST.csv")),
        "merged_core_sha256": digest(common.regular(merged / "merged_core.py")),
        "component_core_sha256": digest(HERE / "component_core.py"),
        "component_train_sha256": digest(HERE / "component_train.py"),
        "component_preflight_sha256": digest(HERE / "component_preflight.py"),
        "package_root": str(package), "merged_root": str(merged),
        "binding_sha256": digest(common.regular(args.binding)),
        "output": str(output), "allowed_runs": allowed_runs,
        "deadline_epoch": args.deadline, "calendar_deadline_epoch": CALENDAR_DEADLINE,
    }
    mismatches = [key for key, value in expected.items() if auth.get(key) != value]
    if mismatches:
        raise ValueError("EXACT_COMPONENT_AUTHORIZATION_REQUIRED: " + ",".join(mismatches))
    budget = auth.get("budget_seconds")
    start = auth.get("window_start_epoch")
    if isinstance(budget, bool) or not isinstance(budget, int) or not 0 < budget <= MAX_BUDGET:
        raise ValueError("INVALID_SHARED_RESOURCE_BUDGET")
    if isinstance(start, bool) or not isinstance(start, (float, int)) or start <= 0:
        raise ValueError("SHARED_WINDOW_START_MISSING")
    now = time.time()
    if not start <= now < args.deadline <= min(start + budget, CALENDAR_DEADLINE):
        raise ValueError("OUTSIDE_SHARED_OR_CALENDAR_DEADLINE")
    for protected in (package, merged, HERE):
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError("OUTPUT_OVERLAPS_SOURCE_TREE")
    return auth


def add_arguments(parser):
    parser.add_argument("--package", required=True)
    parser.add_argument("--merged-dir", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--binding", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--deadline", type=float, required=True)


def run(args):
    common, core, sealed_train, merged_core, component_core = load_sources(args.package, args.merged_dir)
    auth = authorize(args, common, "train", [{"seed": args.seed, "variant": args.variant}])
    output = common.fresh_output(args.output)
    started = time.time()
    attempt = commits = nonzero = 0
    try:
        import torch
        from sealed_initialization import verify_state_mapping
        count = common.verify_package()
        if count != 78:
            raise ValueError("SEALED_PACKAGE_MEMBER_COUNT")
        common.check_deadline(args.deadline)
        common.setup(args.seed)
        base_config = common.configuration(args.seed, "B")
        expected_names = {"MERGED_DROP_S": ["R", "merged_luminance"], "MERGED_DROP_M": ["R", "S"]}
        common.dump(output / "CONFIGURATION.json", {
            **base_config, "variant": args.variant,
            "candidate": {"type": "V11_THREE_TASK_COMPONENT_DELETION",
                          "source_variant": "MERGED_LUM_PCGRAD",
                          "objectives_after_warmup": expected_names[args.variant],
                          "pcgrad": "standard_two_tasks_sum", "dummy_tasks": 0,
                          "remaining_loss_rescaling": False},
        })
        common.dump(output / "RUN_AUTHORIZATION_COPY.json", auth)
        common.dump(output / "SOURCE_IDENTITY.json", {
            **{key: value for key, value in auth.items()
               if key.endswith("sha256") or key.endswith("root")},
            "authorization_sha256": common.sha(args.authorization),
            "sealed_member_count": count,
        })
        common.dump(output / "ENVIRONMENT.json", sealed_train.environment())
        binding, members = common.binding_check(args.binding, "train")
        common.dump(output / "INPUT_BINDING.json", {
            "split": "train551", "binding_sha256": common.sha(args.binding), "files": 1102,
            "data_manifest_sha256": common.sha(Path(args.package) / "base/DATA_MANIFEST.csv"),
            "schedule_sha256": common.sha(Path(args.package) / "schedules" / f"{args.seed}.json"),
        })
        data = common.TrainingData(binding, members, args.seed)
        model, identity = common.sealed_model(args.seed)
        model = model.cuda(0).train()
        torch.cuda.manual_seed_all(args.seed)
        verify_state_mapping(model.state_dict(), common.read(
            Path(args.package) / "initialization" / str(args.seed) / "IDENTITY.json"), "cuda_loaded")
        common.dump(output / "INITIALIZATION.json", identity)
        adam = core.AdamCandidate(core.flatten([p.detach() for p in model.parameters()]))
        rng = random.Random(args.seed)
        torch.cuda.reset_peak_memory_stats()
        for index in range(common.STEPS):
            common.check_deadline(args.deadline)
            attempt = index + 1
            began = time.perf_counter()
            x, y, sample = data.batch(index)
            common.event(output / "attempts.jsonl", {
                "event": "ATTEMPT_START", "attempt": attempt, "variant": args.variant,
                "seed": args.seed, **sample,
            })
            x, y = x.cuda(0), y.cuda(0)
            torch.cuda.synchronize()
            prepared = time.perf_counter()
            detail, trace = component_core.training_update(
                core, merged_core, model, x, y, adam, args.variant, attempt, rng, common.lr_at(index))
            torch.cuda.synchronize()
            computed = time.perf_counter()
            common.check_deadline(args.deadline)
            if adam.step_count != attempt:
                raise ValueError("COMPONENT_OPTIMIZER_COUNTER")
            expected = ["R"] if attempt <= 500 else expected_names[args.variant]
            if detail["objectives"] != expected or trace["raw_objective_gradients"].shape[0] != len(expected):
                raise ValueError("COMPONENT_ACTIVE_OBJECTIVE_CONTRACT")
            trace_name = None
            trace_seconds = 0.0
            if attempt in (500, 501, 8000):
                trace_start = time.perf_counter()
                trace_name = f"trace_{attempt:04d}.pt"
                common.save_trace(output / trace_name, trace, adam)
                trace_seconds = time.perf_counter() - trace_start
            commits += 1
            nonzero += int(detail["nonzero_update"])
            common.event(output / "attempts.jsonl", {
                "event": "COMMIT", "attempt": attempt, "seed": args.seed,
                "component_deletion_active": attempt > 500,
                "data_seconds": prepared - began, "core_seconds": computed - prepared,
                "trace_seconds": trace_seconds, "trace": trace_name,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(), **detail,
            })
            del x, y, trace
            if attempt % 100 == 0:
                print(f"{args.seed} {args.variant} {attempt}/{common.STEPS} {time.time()-started:.1f}s", flush=True)
        common.check_deadline(args.deadline)
        if (attempt, commits, adam.step_count) != (8000, 8000, 8000):
            raise ValueError("INCOMPLETE_FIXED_LAST_TRAINING")
        checkpoint = output / "final_iter8000.pth"
        with checkpoint.open("xb") as stream:
            torch.save(model.cpu().state_dict(), stream)
        common.dump(output / "FINAL_CHECKPOINT.json", {
            "role": "NEW_SCRATCH_FIXED_LAST_V11_COMPONENT_DELETION",
            "seed": args.seed, "variant": args.variant, "attempt": common.STEPS,
            "sha256": common.sha(checkpoint), "initializer_sha256": identity["file_sha256"],
            "selection": "fixed last; no intermediate validation",
            "authorization_sha256": common.sha(args.authorization),
        })
        status = "TRAIN_COMPLETE"
    except Exception as error:
        status = "FAILED_INCOMPLETE"
        common.dump(output / "FAILURE.json", {
            "error": str(error), "traceback": traceback.format_exc(), "attempt": attempt,
            "seed": args.seed, "variant": args.variant,
        })
        print(traceback.format_exc(), flush=True)
    common.dump(output / "TRAIN_SUMMARY.json", {
        "status": status, "seed": args.seed, "variant": args.variant,
        "attempted": attempt, "committed": commits, "nonzero": nonzero,
        "elapsed_seconds": time.time() - started,
        "window_start_epoch": auth["window_start_epoch"], "deadline_epoch": args.deadline,
        "authorization_sha256": common.sha(args.authorization),
    })
    return 0 if status == "TRAIN_COMPLETE" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS)
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    raise SystemExit(run(parser.parse_args()))
