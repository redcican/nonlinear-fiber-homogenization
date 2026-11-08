"""Gauss-Newton projection onto the fiber, sharpness continuation and binary polish.

One step in metric G (Sec. 3.1 / M2 of the plan), with the scaled residual h = (c - c*) / w:
    phi <- phi - G J^T (J G J^T + lam I)^-1 h
The step is halved until the residual decreases. `project` runs the beta continuation; `polish_binary`
shifts the smooth target by the measured binary error (as in the P1 driver) and keeps the best binary cell.
"""
import time

import jax
import jax.numpy as jnp
import numpy as np

from .fiber import rel_error
from .metric import euclid


def scales(target):
    t = np.abs(np.asarray(target, dtype=np.float64))
    return np.maximum(t, 0.1 * t.max())


def gn_direction(J, h, w, metric=euclid, lam=1e-10):
    """Minimum-G-norm step that cancels the linearized scaled residual h."""
    Jf = jnp.reshape(J, (J.shape[0], -1)) / jnp.asarray(w)[:, None]
    GJ = jax.vmap(metric)(J / jnp.asarray(w).reshape((-1,) + (1,) * (J.ndim - 1)))
    GJf = jnp.reshape(GJ, Jf.shape)
    A = Jf @ GJf.T
    a = jnp.linalg.solve(A + lam * jnp.trace(A) / A.shape[0] * jnp.eye(A.shape[0]), jnp.asarray(h))
    return -jnp.tensordot(a, GJ, axes=1)


def gn_project(fiber, phi, target, beta, metric=euclid, tol=1e-4, maxit=60, move=0.5, max_move=2.0,
               min_move=1e-4, verbose=False):
    """Project phi onto c = target at sharpness beta. Gauss-Newton with a trust region on the largest change of
    phi (`move`), merit ||h||_2: a step that lowers the merit is taken and the region grows by 1.5, otherwise
    it shrinks by 2. Returns (phi, info)."""
    w = scales(target)
    tgt = np.asarray(target, dtype=np.float64)
    phi = jnp.asarray(phi)
    c, J = fiber.value_jac(phi, beta)
    h = (c - tgt) / w
    hist = [float(np.abs(h).max())]
    it = 0
    for it in range(maxit):
        if np.abs(h).max() <= tol or move < min_move:
            break
        step = gn_direction(J, h, w, metric)
        smax = float(jnp.abs(step).max())
        trial = phi + (step * (move / smax) if smax > move else step)
        c1, J1 = fiber.value_jac(trial, beta)
        h1 = (c1 - tgt) / w
        if np.linalg.norm(h1) < np.linalg.norm(h):
            phi, c, J, h = trial, c1, J1, h1
            move = min(max_move, 1.5 * move) if smax > move else move
            hist.append(float(np.abs(h).max()))
            if verbose:
                print(f"  gn it {it}: |h|inf {hist[-1]:.2e} (step {min(smax, move):.2e})")
        else:
            move *= 0.5
    return phi, dict(c=c, J=J, res_inf=float(np.abs(h).max()), err=rel_error(c, tgt), iters=it, hist=hist,
                     converged=bool(np.abs(h).max() <= tol))


def project(fiber, phi, target, betas=(2, 4, 8, 16), metric=euclid, tol=1e-4, maxit=60, verbose=False):
    """Beta continuation: project at each sharpness in turn. Returns (phi, info of the last stage)."""
    info = None
    for b in betas:
        phi, info = gn_project(fiber, phi, target, b, metric, tol, maxit, verbose=verbose)
        info["beta"] = b
        if verbose:
            print(f" beta {b}: err {info['err']:.2e} iters {info['iters']}")
    return phi, info


def polish_binary(fiber, phi, target, beta, up=2, metric=euclid, steps=6, damp=0.8, tol_bin=2e-3, tol=1e-4,
                  verbose=False):
    """Target shifting against the binary re-solve at `up` times the design resolution.
    Returns (phi_best, dict(binary=..., err_bin, history))."""
    tgt = np.asarray(target, dtype=np.float64)
    ts = tgt.copy()
    best, best_phi, best_chk = np.inf, phi, None
    hist = []
    for k in range(steps + 1):
        chk = fiber.binary_check(phi, up=up)
        err = rel_error(chk["c"], tgt)
        hist.append(err)
        if verbose:
            print(f"  polish {k}: binary err {err:.2e} vol {chk['vol']:.4f}")
        if err < best:
            best, best_phi, best_chk = err, phi, chk
        if best <= tol_bin or k == steps:
            break
        ts = ts + damp * (tgt - chk["c"])
        phi, _ = gn_project(fiber, phi, ts, beta, metric, tol)
    best_chk = dict(best_chk)
    return best_phi, dict(binary=best_chk, err_bin=best, history=hist)


def project_to_fiber(fiber, phi, target, betas=(2, 4, 8, 16), metric=euclid, up=2, polish_steps=6, tol=1e-4,
                     verbose=False):
    """Full protocol: continuation, grey projection to tol, binary polish at `up` x resolution."""
    t0 = time.time()
    s0 = fiber.solves
    phi, info = project(fiber, phi, target, betas, metric, tol, verbose=verbose)
    grey_err = info["err"]
    phi, pol = polish_binary(fiber, phi, target, betas[-1], up, metric, polish_steps, tol=tol, verbose=verbose)
    c_grey = fiber.value(phi, betas[-1])
    return phi, dict(err_grey=rel_error(c_grey, target), err_grey_before_polish=grey_err, err_bin=pol["err_bin"],
                     binary=pol["binary"], solves=fiber.solves - s0, seconds=time.time() - t0,
                     converged=info["converged"])
