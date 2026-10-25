"""Training-only implementation. No data loading, checkpoint loading or networking."""
from __future__ import annotations
import itertools
import math
import random
import numpy as np
import torch
from torch.nn import functional as F

NAMES = ("R", "S", "over", "under")
INACTIVE = 1e-12
QP_REL_TOL = 1e-8
ROUND_REL_TOL = 1e-3


class NumericalContractError(RuntimeError):
    pass


def luma(x):
    return (x * x.new_tensor([.2126, .7152, .0722])[None, :, None, None]).sum(1, keepdim=True)


def sobel(x):
    # One-pixel symmetric half-sample extension equals replicate, not torch reflect.
    k = x.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8
    # Explicit concatenation also avoids nondeterministic CUDA replicate-pad backward.
    padded = torch.cat((x[..., :1], x, x[..., -1:]), dim=3)
    padded = torch.cat((padded[:, :, :1], padded, padded[:, :, -1:]), dim=2)
    return F.conv2d(padded, torch.stack((k, k.T))[:, None])


def dark_mask(x):
    with torch.no_grad():
        y = luma(x)
        g = torch.linalg.vector_norm(sobel(x.mean(1, keepdim=True)), dim=1, keepdim=True)
        qy = torch.quantile(y.flatten(1), .25, dim=1, interpolation="linear")[:, None, None, None]
        qg = torch.quantile(g.flatten(1), .75, dim=1, interpolation="linear")[:, None, None, None]
        return ((y <= qy) & (g >= qg)).detach()


def objectives(x, y, z, four=True):
    losses = [F.l1_loss(z, y)]
    if not four:
        return losses, {"objectives": ["R"]}
    if tuple(x.shape[-2:]) != (256, 256):
        raise ValueError("Contract requires 256x256 crops")
    mask = dark_mask(x)
    counts = mask.flatten(1).sum(1)
    distance = ((sobel(luma(z)) - sobel(luma(y))).square().sum(1, keepdim=True) + 1e-12).sqrt()
    structural = (distance * mask).flatten(1).sum(1) / counts.clamp_min(1)
    structural = torch.where(counts >= 32, structural, structural * 0)
    # Same nonoverlapping 32x32 mean, avoiding avg_pool's long serial FP32 summation.
    bias = (luma(z) - luma(y)).reshape(len(z), 1, 8, 32, 8, 32).mean(dim=(3, 5))
    losses += [structural.mean(), bias.relu().mean(), (-bias).relu().mean()]
    return losses, {"objectives": list(NAMES), "mask_counts": counts.detach().cpu().tolist(),
                    "empty_structure_images": int((counts < 32).sum().item())}


def flatten(values):
    return torch.cat([v.reshape(-1) for v in values])


def objective_gradients(losses, params):
    rows = []
    for i, loss in enumerate(losses):
        if not bool(torch.isfinite(loss)):
            raise NumericalContractError("Nonfinite training objective")
        grads = torch.autograd.grad(loss, params, retain_graph=i < len(losses) - 1, allow_unused=True)
        rows.append(flatten([torch.zeros_like(p) if g is None else g for p, g in zip(params, grads)]))
    result = torch.stack(rows).detach()
    if not bool(torch.isfinite(result).all()):
        raise NumericalContractError("Nonfinite raw objective gradient")
    return result


def pcgrad(raw, rng: random.Random):
    projected = raw.clone()
    norms = raw.double().square().sum(1)
    orders = []
    for i in range(len(raw)):
        order = [j for j in range(len(raw)) if j != i]
        rng.shuffle(order)
        orders.append(order)
        for j in order:
            if float(norms[j]) <= INACTIVE**2:
                continue
            dot = torch.dot(projected[i].double(), raw[j].double())
            if float(dot) < 0:
                projected[i].add_(raw[j], alpha=-float(dot / norms[j]))
    return projected, projected.sum(0), orders


def clip_for_params(vector, params):
    offset = 0
    for p in params:
        p.grad = vector[offset:offset + p.numel()].view_as(p).clone()
        offset += p.numel()
    norm = torch.nn.utils.clip_grad_norm_(params, .01, error_if_nonfinite=True, foreach=False)
    clipped = flatten([p.grad for p in params]).detach()
    for p in params:
        p.grad = None
    return clipped, float(norm)


class AdamCandidate:
    """Standard Adam moments; proposes but never applies a parameter step."""
    def __init__(self, like):
        self.m = torch.zeros_like(like)
        self.v = torch.zeros_like(like)
        self.step_count = 0

    def propose(self, grad, lr):
        self.step_count += 1
        self.m.lerp_(grad, 1 - .9)
        self.v.mul_(.999).addcmul_(grad, grad, value=1 - .999)
        denom = self.v.sqrt().div_(math.sqrt(1 - .999**self.step_count)).add_(1e-8)
        step = self.m / denom * (lr / (1 - .9**self.step_count))
        if not bool(torch.isfinite(step).all()):
            raise NumericalContractError("Nonfinite Adam candidate")
        return step


def lr_at(attempt_zero_based):
    return 1e-6 + (2e-4 - 1e-6) * (1 + math.cos(math.pi * attempt_zero_based / 2000)) / 2


def project_step(d0, raw):
    """Exact active-set enumeration of the <=4-constraint Euclidean cone QP."""
    raw64, base = raw.double(), d0.double()
    norm = torch.linalg.vector_norm(raw64, dim=1)
    active_rows = norm > INACTIVE
    H = raw64[active_rows] / norm[active_rows, None]
    step_norm = float(torch.linalg.vector_norm(base))
    tol = QP_REL_TOL * max(step_norm, 1e-12)
    if not len(H):
        return base, H, {"tol": tol, "active_rows": [], "multipliers": [], "min_slack": None,
                         "stationarity": 0., "complementarity": 0., "constraint_count": 0}
    gram = (H @ H.T).cpu().numpy()
    q = (H @ base).cpu().numpy()
    candidates = []
    for n in range(len(H) + 1):
        for subset in itertools.combinations(range(len(H)), n):
            lam = np.zeros(len(H), dtype=np.float64)
            if n:
                block = gram[np.ix_(subset, subset)]
                if np.linalg.matrix_rank(block) != n:
                    continue
                lam[list(subset)] = np.linalg.solve(block, -q[list(subset)])
            slack = q + gram @ lam
            comp = float(np.max(np.abs(lam * slack)))
            if lam.min() >= -tol and slack.min() >= -tol and comp <= tol * max(step_norm, 1e-12):
                candidates.append((float(lam @ gram @ lam), tuple(subset), lam))
    if not candidates:
        raise NumericalContractError(f"No KKT-valid active set; Gram={gram.tolist()}, q={q.tolist()}, tol={tol}")
    _, subset, lam = min(candidates, key=lambda row: row[:2])
    multiplier = torch.as_tensor(lam, device=base.device)
    correction = H.T @ multiplier
    result = base + correction
    slack = H @ result
    stationarity = float(torch.linalg.vector_norm(result - base - correction))
    comp = float((multiplier * slack).abs().max())
    if float(slack.min()) < -tol or stationarity > tol or comp > tol * max(step_norm, 1e-12):
        raise NumericalContractError("FP64 reconstructed projection fails frozen KKT tolerance")
    return result, H, {"tol": tol, "active_rows": active_rows.nonzero().flatten().cpu().tolist(),
                       "multipliers": lam.tolist(), "binding_subset": list(subset),
                       "min_slack": float(slack.min()), "stationarity": stationarity,
                       "complementarity": comp, "constraint_count": len(H)}


def rounded_candidate(before, ideal, H, tol):
    after = (before.double() - ideal.double()).to(before.dtype)
    actual = before.double() - after.double()
    rounding = actual - ideal.double()
    ideal_norm = float(torch.linalg.vector_norm(ideal.double()))
    round_norm = float(torch.linalg.vector_norm(rounding))
    ratio = round_norm / max(ideal_norm, 1e-12)
    slack = H @ actual if len(H) else None
    detail = {"ideal_norm": ideal_norm, "actual_norm": float(torch.linalg.vector_norm(actual)),
              "rounding_norm": round_norm, "rounding_ratio": ratio,
              "actual_min_slack": float(slack.min()) if slack is not None else None,
              "rounding_bound": tol + round_norm}
    if ideal_norm == 0:
        valid = torch.equal(after, before)
    else:
        valid = ratio <= ROUND_REL_TOL and (slack is None or float(slack.min()) >= -tol - round_norm)
    if not valid:
        error = NumericalContractError(f"Design numeric requirements incompatible with implementation conditions: {detail}")
        error.detail = detail
        raise error
    return after, actual, detail


def assign_flat(params, vector):
    with torch.no_grad():
        offset = 0
        for p in params:
            p.copy_(vector[offset:offset + p.numel()].view_as(p))
            offset += p.numel()


def training_update(model, x, y, adam, variant, four, rng, lr, projection=True):
    params = list(model.parameters())
    before = flatten([p.detach() for p in params])
    z = model(x)
    losses, detail = objectives(x, y, z, four)
    raw = objective_gradients(losses, params)
    if four:
        pc_rows, combined, orders = pcgrad(raw, rng)
    else:
        pc_rows, combined, orders = raw.clone(), raw[0].clone(), []
    clipped, norm = clip_for_params(combined, params)
    d0 = adam.propose(clipped, lr)
    tensors = {"parameters_before": before, "raw_objective_gradients": raw, "pcgrad_objective_gradients": pc_rows,
               "pcgrad_sum_before_clip": combined, "gradient_after_clip": clipped,
               "adam_candidate_step": d0}
    detail.update({"losses": [float(l.detach()) for l in losses], "pcgrad_orders": orders,
                   "gradient_norm_before_clip": norm, "lr": lr, "adam_step": adam.step_count})
    if variant == "C" and four and projection:
        ideal, H, qp = project_step(d0, raw)
        tensors["projected_ideal_step"] = ideal
        detail["projection"] = qp
        try:
            after, actual, rounding = rounded_candidate(before, ideal, H, qp["tol"])
        except NumericalContractError as error:
            error.tensors = tensors
            error.update_detail = detail
            raise
        detail["rounding"] = rounding
    else:
        ideal = d0
        after = before - d0
        actual = before.double() - after.double()
        detail["projection"] = None
        detail["rounding"] = {"actual_norm": float(torch.linalg.vector_norm(actual))}
        tensors["projected_ideal_step"] = ideal
    tensors["final_actual_parameter_step"] = actual
    detail.update({"nonzero_update": bool(torch.any(actual != 0)),
                   "relative_projection_change": float(torch.linalg.vector_norm(ideal.double() - d0.double())) /
                                                max(float(torch.linalg.vector_norm(d0.double())), 1e-12)})
    if not bool(torch.isfinite(after).all()):
        raise NumericalContractError("Nonfinite final parameters")
    assign_flat(params, after)
    return detail, tensors
