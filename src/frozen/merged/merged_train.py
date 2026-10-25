"""Isolated fixed-last runner for the proposed merged-luminance comparison.

This file does not alter the V2 seal. Importing or invoking --help never trains.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEALED = HERE.parent / "PAPER_B_PCGRAD_METHOD_FOLLOWUP_PACKAGE_V2" / "package"
sys.path.insert(0, str(SEALED))

import common  # noqa: E402
import core  # noqa: E402
import train as sealed_train  # noqa: E402
import merged_core  # noqa: E402

VARIANT = "MERGED_LUM_PCGRAD"


def authorize(path: str, seed: int, deadline: float, binding: str, output: str):
    auth = common.read(common.regular(path))
    expected = {
        "experiment": "PCGRAD_MERGED_LUM_V1",
        "protocol_sha256": common.sha(HERE / "PROTOCOL_AND_SCOPE_ZH.md"),
        "merged_core_sha256": common.sha(HERE / "merged_core.py"),
        "merged_train_sha256": common.sha(HERE / "merged_train.py"),
        "sealed_manifest_sha256": common.sha(SEALED / "MANIFEST.csv"),
        "phase": "train",
        "deadline_epoch": deadline,
        "binding_sha256": common.sha(common.regular(binding)),
        "output": str(Path(output).resolve()),
    }
    if auth.get("approved") is not True or any(auth.get(k) != v for k, v in expected.items()):
        raise ValueError("EXACT_MERGED_AUTHORIZATION_REQUIRED")
    if {"seed": seed, "variant": VARIANT} not in auth.get("allowed_runs", []):
        raise ValueError("MERGED_RUN_NOT_AUTHORIZED")
    budget = auth.get("budget_seconds")
    if not isinstance(budget, int) or budget <= 0 or not isinstance(deadline, float):
        raise ValueError("RESOURCE_BUDGET_OR_DEADLINE_MISSING")
    if time.time() >= deadline or deadline - time.time() > budget:
        raise ValueError("MERGED_DEADLINE_OUTSIDE_AUTHORIZED_BUDGET")
    return auth


def run(args) -> int:
    authorize(args.authorization, args.seed, args.deadline, args.binding, args.output)
    common.verify_package()
    common.check_deadline(args.deadline)
    output = common.fresh_output(args.output)
    started = time.time()
    attempt = commits = nonzero = 0
    try:
        import torch
        from sealed_initialization import verify_state_mapping

        common.setup(args.seed)
        base_config = common.configuration(args.seed, "B")
        common.dump(output / "CONFIGURATION.json", {
            **base_config,
            "variant": VARIANT,
            "candidate": {
                "type": "MERGED_LUMINANCE_PCGRAD",
                "objectives_after_warmup": ["R", "S", "mean_abs_block_bias"],
                "pcgrad": "standard_three_tasks",
            },
        })
        common.dump(output / "ENVIRONMENT.json", sealed_train.environment())
        binding, members = common.binding_check(args.binding, "train")
        common.dump(output / "INPUT_BINDING.json", {
            "split": "train551", "binding_sha256": common.sha(args.binding), "files": 1102,
        })
        data = common.TrainingData(binding, members, args.seed)
        model, identity = common.sealed_model(args.seed)
        model = model.cuda(0).train()
        torch.cuda.manual_seed_all(args.seed)
        verify_state_mapping(model.state_dict(),
                             common.read(SEALED / "initialization" / str(args.seed) / "IDENTITY.json"),
                             "cuda_loaded")
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
                "event": "ATTEMPT_START", "attempt": attempt, "variant": VARIANT,
                "seed": args.seed, **sample,
            })
            x, y = x.cuda(0), y.cuda(0)
            torch.cuda.synchronize()
            prepared = time.perf_counter()
            detail, trace = merged_core.training_update(
                core, model, x, y, adam, attempt, rng, common.lr_at(index)
            )
            torch.cuda.synchronize()
            computed = time.perf_counter()
            if adam.step_count != attempt:
                raise ValueError("MERGED_OPTIMIZER_COUNTER")
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
                "merged_active": attempt > 500,
                "data_seconds": prepared - began, "core_seconds": computed - prepared,
                "trace_seconds": trace_seconds, "trace": trace_name,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(), **detail,
            })
            del x, y, trace
            if attempt % 100 == 0:
                print(f"{args.seed} {VARIANT} {attempt}/{common.STEPS} {time.time()-started:.1f}s",
                      flush=True)
        common.check_deadline(args.deadline)
        checkpoint = output / "final_iter8000.pth"
        with checkpoint.open("xb") as file:
            torch.save(model.cpu().state_dict(), file)
        common.dump(output / "FINAL_CHECKPOINT.json", {
            "role": "NEW_SCRATCH_FIXED_LAST_MERGED_LUM_V1", "seed": args.seed,
            "variant": VARIANT, "attempt": common.STEPS,
            "sha256": common.sha(checkpoint),
            "initializer_sha256": identity["file_sha256"],
            "selection": "fixed last; no intermediate validation",
        })
        status = "TRAIN_COMPLETE"
    except Exception as error:
        status = "FAILED_INCOMPLETE"
        common.dump(output / "FAILURE.json", {
            "error": str(error), "traceback": traceback.format_exc(), "attempt": attempt,
        })
        print(traceback.format_exc(), flush=True)
    common.dump(output / "TRAIN_SUMMARY.json", {
        "status": status, "seed": args.seed, "variant": VARIANT,
        "attempted": attempt, "committed": commits, "nonzero": nonzero,
        "elapsed_seconds": time.time() - started,
    })
    return 0 if status == "TRAIN_COMPLETE" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, choices=common.SEEDS)
    parser.add_argument("--binding", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--deadline", type=float, required=True)
    raise SystemExit(run(parser.parse_args()))
