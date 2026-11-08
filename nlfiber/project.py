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
    """Damped Gauss-Newton step in metric G: -G J^T (J G J^T + lam diag(J G J^T))^-1 h (lam -> 0: the minimum-
    G-norm step that cancels the linearized scaled residual h)."""
    Jf = jnp.reshape(J, (J.shape[0], -1)) / jnp.asarray(w)[:, None]
    GJ = jax.vmap(metric)(J / jnp.asarray(w).reshape((-1,) + (1,) * (J.ndim - 1)))
    GJf = jnp.reshape(GJ, Jf.shape)
    A = Jf @ GJf.T
    D = jnp.diag(jnp.diag(A)) + 1e-300 * jnp.eye(A.shape[0])
    a = jnp.linalg.solve(A + lam * D, jnp.asarray(h))
    return -jnp.tensordot(a, GJ, axes=1)


def gn_project(fiber, phi, target, beta, metric=euclid, tol=1e-4, maxit=100, lam=1e-3, max_step=0.5,
               verbose=False):
    """Project phi onto c = target at sharpness beta by Levenberg-Marquardt on the scaled residual h:
    a step that lowers ||h||_2 is taken and the damping shrinks by 3, otherwise it grows by 4; the largest
    change of phi per step is capped at max_step times the standard deviation of phi (the density map is
    invariant under a rescaling of phi, so the cap is relative). Returns (phi, info)."""
    w = scales(target)
    tgt = np.asarray(target, dtype=np.float64)
    phi = jnp.asarray(phi)
    cap = max_step * float(jnp.std(phi)) + 1e-12
    c, J = fiber.value_jac(phi, beta)
    h = (c - tgt) / w
    hist = [float(np.abs(h).max())]
    it = 0
    for it in range(maxit):
        if np.abs(h).max() <= tol or lam > 1e8:
            break
        step = gn_direction(J, h, w, metric, lam)
        smax = float(jnp.abs(step).max())
        if smax > cap:
            step = step * (cap / smax)
        c1, J1 = fiber.value_jac(phi + step, beta)
        h1 = (c1 - tgt) / w
        if np.linalg.norm(h1) < np.linalg.norm(h):
            phi, c, J, h = phi + step, c1, J1, h1
            lam = max(lam / 3, 1e-12)
            hist.append(float(np.abs(h).max()))
            if verbose:
                print(f"  lm it {it}: |h|inf {hist[-1]:.2e} (step {min(smax, cap):.2e}, lam {lam:.1e})")
        else:
            lam *= 4
    return phi, dict(c=c, J=J, res_inf=float(np.abs(h).max()), err=rel_error(c, tgt), iters=it, hist=hist,
                     converged=bool(np.abs(h).max() <= tol))


def homotopy_targets(c0, target, max_ratio=2.0):
    """Intermediate targets between c0 and target when the stiffness gap is large: geometric interpolation of
    the invariants whose signs agree (linear otherwise), with at most a factor max_ratio per stage on the
    dominant invariant. Returns the list of targets ending with `target`."""
    c0, t = np.asarray(c0, float), np.asarray(target, float)
    gap = max(abs(np.log(abs(t[0]) / max(abs(c0[0]), 1e-30))), 0.0)
    n = int(np.ceil(gap / np.log(max_ratio)))
    if n <= 1:
        return [t]
    out = []
    for k in range(1, n + 1):
        a = k / n
        same = np.sign(c0) == np.sign(t)
        geo = np.sign(t) * np.abs(c0) ** (1 - a) * np.abs(t) ** a
        out.append(np.where(same & (np.abs(c0) > 0), geo, (1 - a) * c0 + a * t))
    return out


def project(fiber, phi, target, betas=(2, 4, 8, 16), metric=euclid, tol=1e-4, maxit=100, verbose=False,
            homotopy=False):
    """Beta continuation: project at each sharpness in turn. With homotopy=True a large stiffness gap is
    closed through intermediate targets at the first sharpness. Returns (phi, info of the last stage)."""
    info = None
    first = True
    for b in betas:
        stages = [np.asarray(target, float)]
        if first and homotopy:
            c0 = fiber.value(phi, b)
            stages = homotopy_targets(c0, target)
        first = False
        for k, tk in enumerate(stages):
            last = k == len(stages) - 1
            phi, info = gn_project(fiber, phi, tk, b, metric, tol if last else 10 * tol, maxit if last else maxit // 2,
                                   verbose=verbose)
            if verbose and not last:
                print(f" beta {b} homotopy stage {k + 1}/{len(stages)}: err {info['err']:.2e}")
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
        phi, _ = gn_project(fiber, phi, ts, beta, metric, tol, maxit=40)
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
