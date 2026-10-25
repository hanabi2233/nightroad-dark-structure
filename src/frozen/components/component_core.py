"""Two-task component deletion of the sealed three-task MERGED computation.

No data loading, model construction, authorization, or training on import.
The complete mode exists only for synthetic equivalence tests, not the run CLI.
"""
from __future__ import annotations

import random
import torch
from torch.nn import functional as F

VARIANTS = ("MERGED_DROP_S", "MERGED_DROP_M")
FULL_TEST_ONLY = "MERGED_LUM_PCGRAD_TEST_ONLY"
INDICES = {"MERGED_DROP_S": (0, 2), "MERGED_DROP_M": (0, 1),
           FULL_TEST_ONLY: (0, 1, 2)}
NAMES = ("R", "S", "merged_luminance")


def selected_objectives(core, merged_core, x, y, z, variant):
    if variant not in INDICES:
        raise ValueError("UNAPPROVED_COMPONENT_VARIANT")
    original, detail = merged_core.merged_objectives(core, x, y, z)
    if detail["objectives"] != list(NAMES) or len(original) != 3:
        raise ValueError("SEALED_MERGED_OBJECTIVE_IDENTITY")
    indices = INDICES[variant]
    return [original[i] for i in indices], {
        **detail, "objectives": [NAMES[i] for i in indices],
        "source_three_tasks": list(NAMES), "selected_task_indices": list(indices),
        "remaining_loss_rescaling": False, "dummy_tasks": 0,
    }


def training_update(core, merged_core, model, x, y, adam, variant: str,
                    attempt: int, rng: random.Random, lr: float):
    if variant not in INDICES or not 1 <= attempt <= 8000:
        raise ValueError("COMPONENT_VARIANT_OR_ATTEMPT_OUTSIDE_CONTRACT")
    if variant == FULL_TEST_ONLY and x.device.type != "cpu":
        raise ValueError("FULL_EQUIVALENCE_MODE_IS_CPU_TEST_ONLY")
    params = list(model.parameters())
    before = core.flatten([p.detach() for p in params])
    z = model(x)
    if attempt <= 500:
        losses = [F.l1_loss(z, y)]
        detail = {"objectives": ["R"], "phase": "L1_WARMUP",
                  "dummy_tasks": 0, "remaining_loss_rescaling": False}
    else:
        losses, detail = selected_objectives(core, merged_core, x, y, z, variant)
        detail["phase"] = "THREE_TASK_COMPONENT_DELETION_PCGRAD"
    raw = core.objective_gradients(losses, params)
    expected_rows = 1 if attempt <= 500 else len(INDICES[variant])
    if raw.ndim != 2 or raw.shape[0] != expected_rows:
        raise ValueError("EXACT_ACTIVE_TASK_COUNT")
    if attempt > 500:
        projected, combined, orders = core.pcgrad(raw, rng)
    else:
        projected, combined, orders = raw.clone(), raw[0].clone(), []
    clipped, norm = core.clip_for_params(combined, params)
    proposal = adam.propose(clipped, lr)
    after = before - proposal
    actual = before.double() - after.double()
    if not bool(torch.isfinite(after).all()):
        raise ValueError("NONFINITE_COMPONENT_PARAMETERS")
    if adam.step_count != attempt:
        raise ValueError("COMPONENT_OPTIMIZER_COUNTER")
    trace = {
        "parameters_before": before, "raw_objective_gradients": raw,
        "pcgrad_objective_gradients": projected, "pcgrad_sum_before_clip": combined,
        "gradient_after_clip": clipped, "adam_candidate_step": proposal,
        "projected_ideal_step": proposal, "final_actual_parameter_step": actual,
    }
    detail.update({
        "variant": variant, "losses": [float(loss.detach()) for loss in losses],
        "pcgrad_applied": bool(orders), "pcgrad_orders": orders,
        "gradient_norm_before_clip": norm, "lr": lr, "adam_step": adam.step_count,
        "optimizer_updates": 1, "nonzero_update": bool(torch.any(actual != 0)),
        "projection": None, "rounding": {"actual_norm": float(torch.linalg.vector_norm(actual))},
        "relative_projection_change": 0.0, "active_task_count": expected_rows,
        "gradient_aggregation": "sum_without_rescaling",
    })
    core.assign_flat(params, after)
    return detail, trace
