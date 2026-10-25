"""Isolated proposed merged-luminance PCGrad computation; no training on import."""

from __future__ import annotations

import random

import torch
from torch.nn import functional as F


def merged_objectives(core, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
    """Keep frozen R and S; merge the two signed block-bias tasks before PCGrad."""
    original, detail = core.objectives(x, y, z, four=True)
    if detail["objectives"] != ["R", "S", "over", "under"] or len(original) != 4:
        raise ValueError("FROZEN_B_OBJECTIVE_IDENTITY_CHANGED")
    merged = original[2] + original[3]
    return [original[0], original[1], merged], {
        **detail,
        "objectives": ["R", "S", "merged_luminance"],
        "original_signed_losses": [float(original[2].detach()), float(original[3].detach())],
    }


def training_update(core, model, x, y, adam, attempt: int, rng: random.Random, lr: float):
    """Match frozen B update order, except PCGrad receives 3 tasks after warmup."""
    if not 1 <= attempt <= 8000:
        raise ValueError("ATTEMPT_OUTSIDE_CONTRACT")
    params = list(model.parameters())
    before = core.flatten([p.detach() for p in params])
    z = model(x)
    if attempt <= 500:
        losses = [F.l1_loss(z, y)]
        detail = {"objectives": ["R"], "phase": "L1_WARMUP"}
    else:
        losses, detail = merged_objectives(core, x, y, z)
        detail["phase"] = "MERGED_LUMINANCE_PCGRAD"
    raw = core.objective_gradients(losses, params)
    if attempt > 500:
        projected, combined, orders = core.pcgrad(raw, rng)
    else:
        projected, combined, orders = raw.clone(), raw[0].clone(), []
    clipped, norm = core.clip_for_params(combined, params)
    proposal = adam.propose(clipped, lr)
    after = before - proposal
    actual = before.double() - after.double()
    if not bool(torch.isfinite(after).all()):
        raise ValueError("NONFINITE_MERGED_PARAMETERS")
    if adam.step_count != attempt:
        raise ValueError("MERGED_OPTIMIZER_COUNTER")
    trace = {
        "parameters_before": before,
        "raw_objective_gradients": raw,
        "pcgrad_objective_gradients": projected,
        "pcgrad_sum_before_clip": combined,
        "gradient_after_clip": clipped,
        "adam_candidate_step": proposal,
        "projected_ideal_step": proposal,
        "final_actual_parameter_step": actual,
    }
    detail.update({
        "variant": "MERGED_LUM_PCGRAD",
        "losses": [float(loss.detach()) for loss in losses],
        "pcgrad_applied": bool(orders),
        "pcgrad_orders": orders,
        "gradient_norm_before_clip": norm,
        "lr": lr,
        "adam_step": adam.step_count,
        "optimizer_updates": 1,
        "nonzero_update": bool(torch.any(actual != 0)),
        "projection": None,
        "rounding": {"actual_norm": float(torch.linalg.vector_norm(actual))},
        "relative_projection_change": 0.0,
    })
    core.assign_flat(params, after)
    return detail, trace
