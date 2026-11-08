"""S1: projected flow (TC-Flow adapted to the fiber), and the PCFM-style Euclidean variant (baseline B5).

At each step from t >= t0 the Tweedie endpoint D = x + (1 - t) v(x, t) is moved by n_gn Gauss-Newton steps
towards the fiber in the metric Gamma_t (Gaussian-Tweedie DCT metric of the data for t < t_sw, the learned
Tweedie metric after it, or the identity for B5), at a sharpness beta(t) that grows to the final value; then
x moves with the conditional velocity (D - x) / (1 - t). The end point is projected exactly (continuation)
and polished against the binary re-solve.
"""
import time

import jax
import jax.numpy as jnp
import numpy as np

from .. import project as P
from ..metric import DCTMetric, LearnedTweedie, euclid


def beta_schedule(t, t0, b0=2.0, b1=16.0):
    a = min(1.0, max(0.0, (t - t0) / max(1e-9, 1.0 - t0)))
    return float(b0 * (b1 / b0) ** a)


def s1_sample(prior, fiber, target, z, K=16, t0=0.4, t_sw=0.8, metric="tweedie", spectrum=None, n_gn=1,
              cond=None, final=True, betas=(2, 4, 8, 16), up=2, polish_steps=6, verbose=False):
    """One S1 sample from latent z. metric: 'tweedie' (Gamma_G, then learned after t_sw), 'gaussian' (Gamma_G
    only), 'euclid' (B5). Returns (phi, info)."""
    t_start = time.time()
    s0 = fiber.solves
    dct = DCTMetric(spectrum) if spectrum is not None else None
    vel = jax.jit(lambda x, t: prior.velocity(x, t, cond))
    x = jnp.asarray(z, jnp.float32)
    dt = 1.0 / K
    for k in range(K):
        t = k * dt
        v = vel(x, t)
        D = x + (1 - t) * v
        if t >= t0 and k < K:
            beta = beta_schedule(t, t0, betas[0], betas[-1])
            if metric == "euclid" or dct is None:
                G = euclid
            elif metric == "tweedie" and t >= t_sw:
                G = LearnedTweedie(lambda y, tt, c: prior.velocity(y, tt, c), cond).at(x, t)
            else:
                G = dct.at(t)
            phi = prior.to_phi(D).astype(jnp.float64)
            for _ in range(n_gn):
                c, J = fiber.value_jac(phi, beta)
                h = (c - np.asarray(target)) / P.scales(target)
                step = P.gn_direction(J, h, P.scales(target), lambda u: G(u.astype(jnp.float32)).astype(jnp.float64))
                smax = float(jnp.abs(step).max())
                phi = phi + step * min(1.0, 2.0 / max(smax, 1e-12))  # cap the in-loop move
            D = prior.to_model(phi).astype(jnp.float32)
            if verbose:
                print(f"  t {t:.3f} beta {beta:.1f} |h| {np.abs(h).max():.2e}")
        x = x + dt * (D - x) / (1 - t)
    phi = prior.to_phi(x).astype(jnp.float64)
    info = dict(in_loop_solves=fiber.solves - s0)
    if final:
        fmet = euclid if (metric == "euclid" or dct is None) else dct.at(None)
        phi, pinfo = P.project_to_fiber(fiber, phi, target, betas, fmet, up=up, polish_steps=polish_steps)
        info.update(pinfo)
    info["seconds"] = time.time() - t_start
    info["solves"] = fiber.solves - s0
    return phi, info
