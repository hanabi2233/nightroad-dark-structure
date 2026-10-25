"""SUM4 only: individual raw gradients -> sum -> one clip -> one Adam proposal.

The imported core is the immutable original pilot core, not C numerical V2.
No architecture, data, checkpoint or device initialization occurs on import.
"""
import torch
import core


def four(variant, step):
    if variant != 'SUM4' or not 1 <= step <= 2000:
        raise ValueError('SUM4 identity / attempt contract violation')
    return step > 500


def combine(raw, use_four):
    if raw.ndim != 2 or len(raw) != (4 if use_four else 1):
        raise ValueError('Expected R,S,over,under rows, or warmup R only')
    # Exactly the reduction used after PCGrad in B; no averaging or row clipping.
    return raw.sum(0) if use_four else raw[0].clone()


def training_update(model, x, y, adam, variant, four, rng, lr):
    if variant != 'SUM4':
        raise ValueError('SUM4 cannot be labelled A, B or C')
    params = list(model.parameters())
    before = core.flatten([p.detach() for p in params])
    z = model(x)
    losses, detail = core.objectives(x, y, z, four)
    raw = core.objective_gradients(losses, params)
    combined = combine(raw, four)
    clipped, norm = core.clip_for_params(combined, params)
    d0 = adam.propose(clipped, lr)
    # Same FP32 subtraction as B, not a double-precision reimplementation.
    after = before - d0
    actual = before.double() - after.double()
    tensors = dict(parameters_before=before, raw_objective_gradients=raw,
                   summed_raw_gradient_before_clip=combined,
                   gradient_after_clip=clipped, adam_candidate_step=d0,
                   unprojected_ideal_step=d0, final_actual_parameter_step=actual)
    detail.update(variant='SUM4', combination='SUM4_RAW_SUM' if four else 'L1_WARMUP',
                  pcgrad_applied=False, optimizer_updates=1,
                  losses=[float(v.detach()) for v in losses], pcgrad_orders=[],
                  gradient_norm_before_clip=norm, lr=lr, adam_step=adam.step_count,
                  projection=None, rounding={'actual_norm':float(torch.linalg.vector_norm(actual))},
                  nonzero_update=bool(torch.any(actual != 0)), relative_projection_change=0.)
    if not bool(torch.isfinite(after).all()):
        error = core.NumericalContractError('SUM4 nonfinite final parameters')
        error.tensors = tensors
        raise error
    core.assign_flat(params, after)
    return detail, tensors
