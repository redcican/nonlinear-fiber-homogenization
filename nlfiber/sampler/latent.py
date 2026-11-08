"""The fiber in the generator's latent space: Z = {z : c(G_K(z)) = c*} (Sec. 3.2, latent form).

h(z) = (c(rho(phi(G_K(z)))) - c*) / w, and J_z = J_phi dG_K/dz from three VJPs through the generator and the
density map (correctors held fixed: exact). The generator is a fixed deterministic map (fixed K, method,
parameters), so the pushforward of the latent co-area target is pi* on the fiber.
"""
import jax
import jax.numpy as jnp
import numpy as np

from ..project import scales


class LatentFiber:
    def __init__(self, prior, fiber, target, beta, K=8, method="euler", cond=None, float64=True):
        """float64: run the generator in double precision (its parameters are cast once), so that the Newton
        solves of the manifold MCMC reach tight tolerances; float32 is faster for projection only."""
        self.prior, self.fiber, self.beta, self.K = prior, fiber, float(beta), K
        self.target = jnp.asarray(target)
        self.w = jnp.asarray(scales(target))
        self.calls = 0
        self.dtype = jnp.float64 if float64 else jnp.float32
        params = prior.params64() if float64 else None

        def phi_of(z):
            return prior.to_phi(prior.generate(z, K, cond, method, params)).astype(jnp.float64)

        def h_of(z, u0):
            c, (u, it) = fiber.cfun(phi_of(z), self.beta, u0)
            return (c - self.target) / self.w, u

        def hj(z, u0):
            h, pull, u = jax.vjp(lambda zz: h_of(zz, u0), z, has_aux=True)
            J = jax.vmap(lambda e: pull(e)[0])(jnp.eye(3, dtype=h.dtype))
            return h, J, u

        self.phi_of = jax.jit(phi_of)
        self._hj = jax.jit(hj)
        self.u = None

    def cj(self, z):
        u0 = self.u if self.u is not None else jnp.zeros((2,) + self.fiber.lh.shape + (3,))
        h, J, u = self._hj(jnp.asarray(z, self.dtype), u0)
        self.u = u
        self.calls += 1
        return np.asarray(h), J

    def project(self, z, tol=1e-6, maxit=200, lam=1e-2, homotopy=False):
        """Levenberg-Marquardt in z (Euclidean latent metric) onto Z: a = (J J^T + lam I)^-1 h, dz = -J^T a;
        lam grows by 10 on a rejected step and shrinks by 3 on an accepted one. A large stiffness gap is
        closed through intermediate targets first (homotopy, as in project.project)."""
        if homotopy:
            from ..project import homotopy_targets
            h0, _ = self.cj(z)
            c0 = np.asarray(h0) * np.asarray(self.w) + np.asarray(self.target)
            stages = homotopy_targets(c0, np.asarray(self.target))
            if len(stages) > 1:
                final = self.target
                for tk in stages[:-1]:
                    self.target = jnp.asarray(tk)
                    z, _ = self.project(z, tol=10 * tol, maxit=maxit // 2, lam=lam, homotopy=False)
                self.target = final
        h, J = self.cj(z)
        for _ in range(maxit):
            if np.abs(h).max() <= tol or lam > 1e8:
                break
            Jf = np.asarray(J, np.float64).reshape(3, -1)
            A = Jf @ Jf.T
            a = np.linalg.solve(A + lam * np.trace(A) / 3 * np.eye(3), h)
            trial = z - jnp.asarray((a @ Jf).reshape(z.shape), z.dtype)
            h1, J1 = self.cj(trial)
            if np.linalg.norm(h1) < np.linalg.norm(h):
                z, h, J = trial, h1, J1
                lam = max(lam / 3, 1e-10)
            else:
                lam *= 10
        return z, dict(res_inf=float(np.abs(h).max()), converged=bool(np.abs(h).max() <= tol))
