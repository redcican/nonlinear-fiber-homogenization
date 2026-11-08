"""Tilted sequential Monte Carlo on the fiber: pi_beta ∝ pi* exp(-beta L) (M4 of the plan).

Particles are latent states on Z (the fiber in the generator's latent space). Tempering follows the
effective sample size (ESS >= ess_min * N chooses the next beta by bisection); after each resampling the
particles move with tilted ZHG steps. Values-only: L is evaluated, never differentiated, so bifurcations do
not break it. The output is a set of designs at exact stiffness with weights.
"""
import time

import numpy as np
import jax.numpy as jnp

from ..sampler.zhg import ZHG


def ess(logw):
    w = np.exp(logw - logw.max())
    w /= w.sum()
    return 1.0 / np.sum(w**2)


def next_beta(L, beta, ess_min, beta_max):
    """Largest beta' <= beta_max with ESS of the incremental weights >= ess_min."""
    lo, hi = beta, beta_max
    if ess(-(hi - beta) * L) >= ess_min:
        return hi
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if ess(-(mid - beta) * L) >= ess_min:
            lo = mid
        else:
            hi = mid
    return lo


def tilted_smc(latent, particles, loss, beta_max=1.0, ess_frac=0.5, moves=5, step=0.3, seed=0, gram=True,
               verbose=False):
    """latent: LatentFiber (cj, project); particles: list of latent states on Z; loss(z) -> L (float).
    Returns dict(particles, L, betas, acceptance, seconds)."""
    rng = np.random.default_rng(seed)
    N = len(particles)
    Z = list(particles)
    L = np.array([loss(z) for z in Z])
    beta, betas, t0 = 0.0, [0.0], time.time()
    acc = []
    cache = {}  # tilt values of the proposed states

    def tilt(z):
        key = hash(np.asarray(z).tobytes())
        if key not in cache:
            cache[key] = float(loss(z))
        return beta * cache[key]

    sampler = ZHG(latent.cj, gram=gram, tilt=tilt)
    while beta < beta_max:
        beta_new = next_beta(L, beta, ess_frac * N, beta_max)
        logw = -(beta_new - beta) * L
        w = np.exp(logw - logw.max())
        w /= w.sum()
        idx = rng.choice(N, N, p=w)  # multinomial resampling
        Z = [Z[i] for i in idx]
        L = L[idx]
        beta = beta_new
        betas.append(beta)
        cache.clear()
        for i in range(N):
            cache[hash(np.asarray(Z[i]).tobytes())] = float(L[i])
        n_acc = 0
        for i in range(N):
            z = Z[i]
            _, J = latent.cj(z)
            for _ in range(moves):
                z, J, a = sampler.step(z, J, step, rng)
                n_acc += a
            Z[i] = z
            L[i] = cache[hash(np.asarray(z).tobytes())]
        acc.append(n_acc / (N * moves))
        if verbose:
            print(f"  smc beta {beta:.3f}  L mean {L.mean():.4f} min {L.min():.4f}  acc {acc[-1]:.2f}", flush=True)
    return dict(particles=Z, L=L, betas=betas, acceptance=acc, seconds=time.time() - t0, stats=sampler.stats)
