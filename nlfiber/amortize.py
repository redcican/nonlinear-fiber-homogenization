"""Amortized conditional generator (M5): a flow prior conditioned on (c*, v, d) with d the nonlinear descriptor
(nu at strains, stiffening, eps_cr, TOECs). Inference: sample -> exact projection -> a few null-space steps.
The distillation gap is measured by `gap`: success, diversity and solver calls of the amortized designs
against the steered ones that trained it."""
import numpy as np
import jax
import jax.numpy as jnp

from . import project as P, topology as T
from .prior.flow import FlowPrior


def descriptor(c, v, qois, keys=("nu10", "nu20", "S20", "eps_max")):
    """Condition vector from the stiffness invariants, volume and a QoI dict (nan -> 0 with an indicator)."""
    vals = []
    for k in keys:
        x = qois.get(k, np.nan)
        vals += [0.0 if not np.isfinite(x) else float(x), 0.0 if np.isfinite(x) else 1.0]
    return np.array(list(np.asarray(c, float)) + [float(v)] + vals, np.float32)


class Amortized:
    def __init__(self, n_o, cond_dim, d=3, widths=(16, 32, 64), seed=0):
        self.prior = FlowPrior(n_o, d, widths, cond_dim=cond_dim, seed=seed)

    def fit(self, phis, conds, **kw):
        return self.prior.fit(phis, cond=conds, **kw)

    def design(self, fiber, target, cond, key, K=16, beta=16.0, ns_steps=0, loss_and_grad=None, up=2):
        """One amortized design for the condition vector: sample, project exactly, optional null-space steps."""
        s0 = fiber.solves
        phi = self.prior.sample_phi(key, 1, K, jnp.asarray(cond))[0].astype(jnp.float64)
        phi, info = P.project(fiber, phi, target, betas=(beta,))
        if ns_steps and loss_and_grad is not None:
            from .steer.nullspace import null_space_flow
            phi, _ = null_space_flow(fiber, phi, target, loss_and_grad, beta=beta, iters=ns_steps)
        phi, pol = P.polish_binary(fiber, phi, target, beta, up=up)
        return phi, dict(err_bin=pol["err_bin"], binary=pol["binary"], solves=fiber.solves - s0)


def gap(steered, amortized, success):
    """Distillation gap: success rate, mean pairwise IoU and solver calls of both sets.
    steered/amortized: lists of dicts with 'binary' (cell), 'solves' and the QoI used by success(d) -> bool."""
    def summary(rows):
        cells = [r["binary"]["cell"] for r in rows]
        return dict(n=len(rows), success=float(np.mean([success(r) for r in rows])) if rows else np.nan,
                    iou=T.mean_pairwise_iou(cells) if len(cells) > 1 else np.nan,
                    solves=float(np.mean([r["solves"] for r in rows])) if rows else np.nan)
    s, a = summary(steered), summary(amortized)
    return dict(steered=s, amortized=a, success_ratio=a["success"] / s["success"] if s["success"] else np.nan,
                cost_ratio=a["solves"] / s["solves"] if s["solves"] else np.nan)
