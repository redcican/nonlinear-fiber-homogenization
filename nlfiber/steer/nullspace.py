"""Null-space gradient flow on the fiber (Feppon, Allaire & Dapogny 2020) in a metric Gamma (M4 of the plan):

    phi_dot = -alpha_L Pi_T Gamma grad L - alpha_C Gamma J^T (J Gamma J^T)^-1 h,
    Pi_T = I - Gamma J^T (J Gamma J^T)^-1 J,

with L a loss of nonlinear quantities (its gradient from the finite-strain adjoint through the density map),
h the scaled stiffness residual and J its Jacobian rows. After every step a Gauss-Newton correction restores
|h|_inf <= tol. The loss gradient with respect to phi is obtained by pulling d L / d rho back through the
density map with the chain rule of the exact-volume threshold.
"""
import time

import jax
import jax.numpy as jnp
import numpy as np

from .. import project as P
from ..metric import euclid


def pullback(fiber, phi, beta, g_rho):
    """d L / d phi from d L / d rho through rho(phi) (density map, volume threshold, symmetrization)."""
    _, pull = jax.vjp(lambda p: fiber.dm.rho(p, float(beta)), jnp.asarray(phi))
    return pull(jnp.asarray(g_rho, jnp.float64))[0]


def tangent_project(J, g, w, metric=euclid):
    """Pi_T Gamma g: the Gamma-orthogonal projection of Gamma g onto the tangent space of the fiber."""
    Gg = metric(g)
    Jf = jnp.reshape(J, (J.shape[0], -1)) / jnp.asarray(w)[:, None]
    GJ = jax.vmap(metric)(J / jnp.asarray(w).reshape((-1,) + (1,) * (J.ndim - 1)))
    GJf = jnp.reshape(GJ, Jf.shape)
    A = Jf @ GJf.T
    a = jnp.linalg.solve(A + 1e-12 * jnp.trace(A) * jnp.eye(A.shape[0]), Jf @ jnp.ravel(g))
    return Gg - jnp.tensordot(a, GJ, axes=1)


def null_space_flow(fiber, phi, target, loss_and_grad, beta=16.0, metric=euclid, iters=60, step=0.05,
                    tol=1e-4, typicality=None, lam_typ=0.0, verbose=False, callback=None):
    """Minimize L on the fiber. loss_and_grad(rho) -> (L, dL/drho) (finite-strain adjoint).
    typicality(phi) -> score s_theta(phi) (optional prior-likelihood term, added as lam_typ * s).
    The step is the largest change of phi per iteration; it halves when L does not decrease and grows by 1.2
    otherwise. Returns (phi, history)."""
    w = P.scales(target)
    phi = jnp.asarray(phi)
    phi, _ = P.gn_project(fiber, phi, target, beta, metric, tol)
    hist = []
    t0 = time.time()
    L, g_rho = loss_and_grad(fiber.rho(phi, beta))
    for it in range(iters):
        g = pullback(fiber, phi, beta, g_rho)
        if typicality is not None and lam_typ:
            g = g - lam_typ * typicality(phi)
        c, J = fiber.value_jac(phi, beta)
        d = -tangent_project(J, g, w, metric)
        dmax = float(jnp.abs(d).max()) + 1e-30
        trial = phi + d * (step / dmax)
        trial, info = P.gn_project(fiber, trial, target, beta, metric, tol, maxit=20)
        L1, g1 = loss_and_grad(fiber.rho(trial, beta))
        accepted = L1 < L
        hist.append(dict(it=it, L=float(L), L_trial=float(L1), step=step, res=info["res_inf"], seconds=time.time() - t0,
                         accepted=bool(accepted)))
        if verbose:
            print(f"  ns it {it}: L {L:.5f} -> {L1:.5f} step {step:.3g} |h| {info['res_inf']:.1e} {'ok' if accepted else 'rej'}",
                  flush=True)
        if callback is not None:
            callback(it, trial, L1)
        if accepted:
            phi, L, g_rho = trial, L1, g1
            step = min(0.5, 1.2 * step)
        else:
            step *= 0.5
            if step < 1e-4:
                break
    return phi, hist
