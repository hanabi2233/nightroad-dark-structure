"""Local-only candidate computation core; importing this file never trains.

GT-Mean is independently expressed from the mathematical operations described
by Liao et al. (ICCV 2025) and their public RetinexFormer configuration.
Their GPL-3.0 implementation is not copied into this file.
"""

from __future__ import annotations

import math
import random

import torch
from torch.nn import functional as F


SOURCE_COMMIT = "1a09e2e077613c195ae7890005ffdfc8ce01dd4d"
NAMES = ("R", "S", "over", "under")
DROP_VARIANTS = {
    "B_DROP_R": "R",
    "B_DROP_S": "S",
    "B_DROP_OVER": "over",
    "B_DROP_UNDER": "under",
}
VARIANTS = ("GTMEAN", *DROP_VARIANTS)
WARMUP = 500


class CandidateNumericalError(RuntimeError):
    """Reject an undefined or non-finite candidate update without changing parameters."""


def gt_mean_loss(pred: torch.Tensor, target: torch.Tensor, sigma: float = 0.1):
    """Same regular-case operations as the authors' dou_KL_2, reduction=none loss.

    The authors' ``num_iter`` argument is unused by the selected weighting
    function. We do not add epsilon to the ratio or KL calculation: this would
    change the loss. Undefined near-zero batch elements are rejected instead.
    """
    if pred.shape != target.shape or pred.ndim != 4 or pred.shape[1] != 3:
        raise ValueError("GT-Mean requires matching N,3,H,W tensors")
    if not pred.is_floating_point() or not target.is_floating_point():
        raise ValueError("GT-Mean requires floating-point tensors")
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be positive and finite")
    if not bool(torch.isfinite(pred).all() and torch.isfinite(target).all()):
        raise CandidateNumericalError("non-finite GT-Mean input")

    # TorchVision's RGB tensor grayscale convention used in the cited code.
    def gray_mean(tensor):
        r, g, b = tensor.unbind(dim=1)
        return (0.2989 * r + 0.587 * g + 0.114 * b).mean(dim=(1, 2))

    mu_target = gray_mean(target)
    mu_pred = gray_mean(pred)
    abs_target, abs_pred = mu_target.abs(), mu_pred.abs()
    if bool((abs_target <= 1e-12).any() or (abs_pred <= 1e-12).any()):
        raise CandidateNumericalError("GT-Mean is undefined for near-zero image mean")

    sig_target, sig_pred = sigma * abs_target, sigma * abs_pred
    mu_mid = (abs_target + abs_pred) / 2
    sig_mid = ((sig_target.square() + sig_pred.square()) / 2).sqrt()

    def kl(mu, sig):
        return (sig_mid / sig).log() + (
            sig.square() + (mu - mu_mid).square()
        ) / (2 * sig_mid.square()) - 0.5

    blend = ((kl(abs_target, sig_target) + kl(abs_pred, sig_pred)) / 2).clamp(0, 1).detach()
    scale = mu_target / mu_pred
    corrected = (pred * scale[:, None, None, None]).clamp(0, 1)
    raw_l1 = (pred - target).abs()
    corrected_l1 = (corrected - target).abs()
    loss = (blend[:, None, None, None] * raw_l1 +
            (1 - blend[:, None, None, None]) * corrected_l1).mean()
    if not bool(torch.isfinite(loss)):
        raise CandidateNumericalError("non-finite GT-Mean loss")
    return loss, {"blend_weight": blend.detach(), "scale": scale.detach()}


def select_objectives(core, x, y, z, variant: str):
    """Select 3 of the unchanged four frozen B objectives, without reweighting."""
    if variant not in DROP_VARIANTS:
        raise ValueError("not a registered leave-one-out variant")
    complete, detail = core.objectives(x, y, z, four=True)
    if tuple(detail["objectives"]) != NAMES or len(complete) != 4:
        raise CandidateNumericalError("frozen B objective identity changed")
    removed = DROP_VARIANTS[variant]
    indices = [i for i, name in enumerate(NAMES) if name != removed]
    selected = [complete[i] for i in indices]
    return selected, {**detail, "objectives": [NAMES[i] for i in indices],
                      "removed_objective": removed}


def training_update(core, model, x, y, adam, variant: str,
                    attempt: int, rng: random.Random, lr: float):
    """One candidate update with B's raw gradients, PCGrad, clip and Adam order.

    This is an isolated CPU-testable core, not a sealed training runner.
    The common first 500 updates use RGB L1 even for B_DROP_R; that branch
    therefore removes R only during the multi-objective training phase.
    """
    if variant not in VARIANTS or not 1 <= attempt <= 8000:
        raise ValueError("candidate identity or attempt outside contract")
    params = list(model.parameters())
    before = core.flatten([p.detach() for p in params])
    z = model(x)
    if attempt <= WARMUP:
        losses = [F.l1_loss(z, y)]
        detail = {"objectives": ["R"], "phase": "L1_WARMUP"}
    elif variant == "GTMEAN":
        loss, diagnostic = gt_mean_loss(z, y)
        losses = [loss]
        detail = {"objectives": ["GTMEAN"], "phase": "CANDIDATE",
                  "blend_weight": diagnostic["blend_weight"].cpu().tolist(),
                  "brightness_scale": diagnostic["scale"].cpu().tolist()}
    else:
        losses, detail = select_objectives(core, x, y, z, variant)
        detail["phase"] = "CANDIDATE"

    raw = core.objective_gradients(losses, params)
    if attempt > WARMUP and variant in DROP_VARIANTS:
        projected, combined, orders = core.pcgrad(raw, rng)
    else:
        projected, combined, orders = raw.clone(), raw[0].clone(), []
    clipped, norm = core.clip_for_params(combined, params)
    proposal = adam.propose(clipped, lr)
    after = before - proposal
    actual = before.double() - after.double()
    if not bool(torch.isfinite(after).all()):
        raise CandidateNumericalError("non-finite candidate parameters")
    if adam.step_count != attempt:
        raise CandidateNumericalError("optimizer counter mismatch")
    trace = {"parameters_before": before, "raw_objective_gradients": raw,
             "pcgrad_objective_gradients": projected,
             "pcgrad_sum_before_clip": combined, "gradient_after_clip": clipped,
             "adam_candidate_step": proposal, "final_actual_parameter_step": actual}
    detail.update({"variant": variant, "losses": [float(loss.detach()) for loss in losses],
                   "pcgrad_applied": bool(orders), "pcgrad_orders": orders,
                   "gradient_norm_before_clip": norm, "lr": lr,
                   "adam_step": adam.step_count, "optimizer_updates": 1,
                   "nonzero_update": bool(torch.any(actual != 0))})
    core.assign_flat(params, after)
    return detail, trace
