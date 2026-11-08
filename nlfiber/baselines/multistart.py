"""B1 multi-start constrained optimization and B2 deflation (both exact in C^H by projection)."""
import numpy as np
import jax.numpy as jnp

from .. import fiber as Fb, project as P
from ..steer.nullspace import null_space_flow


def random_linear_loss(shape, seed):
    """L(rho) = <a, rho> with a random unit a (sampling by multi-start optimization)."""
    a = jnp.asarray(np.random.default_rng(seed).standard_normal(shape))
    a = a / jnp.linalg.norm(a)
    return lambda rho: (float(jnp.vdot(a, rho)), a)


def b1_multistart(fiber, target, starts, loss_and_grad=None, beta=16.0, iters=30, up=2, seed=0, verbose=False):
    """From each start: project, minimize the loss on the fiber, polish binary. Returns a list of dicts."""
    out = []
    for i, phi0 in enumerate(starts):
        lg = loss_and_grad if loss_and_grad is not None else random_linear_loss(fiber.dm.shape, seed + i)
        phi, _ = P.project(fiber, phi0, target, betas=(2, 4, 8, beta))
        phi, hist = null_space_flow(fiber, phi, target, lg, beta=beta, iters=iters, verbose=verbose)
        phi, pol = P.polish_binary(fiber, phi, target, beta, up=up)
        out.append(dict(phi=phi, err_bin=pol["err_bin"], binary=pol["binary"], L=hist[-1]["L"] if hist else np.nan,
                        solves=fiber.solves))
    return out


def b2_deflation(fiber, target, starts, loss_and_grad, beta=16.0, iters=30, up=2, alpha=1.0, power=2, verbose=False):
    """Deflation: the loss is multiplied by prod_k (1 / |rho - rho_k|^power + alpha) over found designs, so
    the flow is pushed away from them."""
    found = []
    out = []
    for phi0 in starts:
        def deflated(rho):
            L, g = loss_and_grad(rho)
            L, g = jnp.asarray(L), jnp.asarray(g)
            m, gm = 1.0, jnp.zeros(rho.shape)
            for rk in found:
                d = rho - rk
                n2 = jnp.vdot(d, d) + 1e-12
                f = n2 ** (-power / 2) + alpha
                df = -power * n2 ** (-power / 2 - 1) * d
                gm = gm * f + m * df
                m = m * f
            return float(L * m), g * m + L * gm

        phi, _ = P.project(fiber, phi0, target, betas=(2, 4, 8, beta))
        phi, hist = null_space_flow(fiber, phi, target, deflated, beta=beta, iters=iters, verbose=verbose)
        phi, pol = P.polish_binary(fiber, phi, target, beta, up=up)
        found.append(fiber.rho(phi, beta))
        out.append(dict(phi=phi, err_bin=pol["err_bin"], binary=pol["binary"], L=loss_and_grad(found[-1])[0]))
    return out
