"""B4 DPS-style soft guidance and B6 values-only SMC with the learned prior."""
import numpy as np
import jax
import jax.numpy as jnp

from .. import project as P
from ..metric import euclid


def b4_dps(prior, fiber, target, z, K=16, strength=1.0, t0=0.0, beta=8.0, cond=None, final=True, up=2):
    """Guidance: at each step the endpoint D moves by -strength * grad_D |h(D)|^2 / |h| (DPS normalization),
    Euclidean, with the stiffness Jacobian of the fiber. Returns (phi, info)."""
    vel = jax.jit(lambda x, t: prior.velocity(x, t, cond))
    x = jnp.asarray(z, jnp.float32)
    dt = 1.0 / K
    w = P.scales(target)
    s0 = fiber.solves
    for k in range(K):
        t = k * dt
        D = x + (1 - t) * vel(x, t)
        if t >= t0:
            phi = prior.to_phi(D).astype(jnp.float64)
            c, J = fiber.value_jac(phi, beta)
            h = (c - np.asarray(target)) / w
            g = jnp.tensordot(jnp.asarray(2 * h / w), J, axes=1)  # grad of |h|^2 in phi
            step = -strength * g / (jnp.linalg.norm(g) + 1e-30) * float(np.linalg.norm(h))
            D = prior.to_model(phi + step).astype(jnp.float32)
        x = x + dt * (D - x) / (1 - t)
    phi = prior.to_phi(x).astype(jnp.float64)
    c = fiber.value(phi, beta)
    info = dict(err_soft=float(np.linalg.norm(c - target) / np.linalg.norm(target)))
    if final:
        phi, pinfo = P.project_to_fiber(fiber, phi, target, betas=(beta,), metric=euclid, up=up)
        info.update(pinfo)
    info["solves"] = fiber.solves - s0
    return phi, info


def b6_values_only(prior, fiber, target, key, n_particles=16, K=16, beta=8.0, temps=(1e2, 1e3, 1e4), cond=None,
                   final=True, up=2):
    """Feynman-Kac steering without Jacobians: prior samples are reweighted by exp(-T |h|^2) and resampled at
    each temperature T; the survivors are then projected exactly. Returns (list of phi, info)."""
    rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 2**31 - 1)))
    z = prior.latent(key, n_particles)
    phis = [p.astype(jnp.float64) for p in prior.sample_phi(key, n_particles, K, cond)]
    w = P.scales(target)
    h2 = np.array([np.sum(((fiber.value(p, beta) - target) / w) ** 2) for p in phis])
    for T in temps:
        lw = -T * h2
        pw = np.exp(lw - lw.max())
        pw /= pw.sum()
        idx = rng.choice(n_particles, n_particles, p=pw)
        phis = [phis[i] for i in idx]
        h2 = h2[idx]
        # one fresh prior sample for each duplicate (keeps diversity; values-only)
        seen = set()
        for j, i in enumerate(idx):
            if i in seen:
                phis[j] = prior.sample_phi(jax.random.fold_in(key, int(T) + j), 1, K, cond)[0].astype(jnp.float64)
                h2[j] = np.sum(((fiber.value(phis[j], beta) - target) / w) ** 2)
            seen.add(i)
    info = dict(err_soft=[float(np.sqrt(x)) for x in h2])
    if final:
        out = []
        for p in phis:
            p, pinfo = P.project_to_fiber(fiber, p, target, betas=(beta,), up=up)
            out.append((p, pinfo))
        phis = [p for p, _ in out]
        info["err_bin"] = [i["err_bin"] for _, i in out]
    return phis, info
