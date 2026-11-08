"""Constrained-manifold MCMC of Zappa, Holmes-Cerfon & Goodman (2018) with an exact reversibility check.

Target on Z = {z : h(z) = 0} (m constraints):  pi(z) ∝ exp(logp(z)) det(J J^T)^(-1/2) dH(z) [x exp(-tilt(z))].
With logp the latent standard normal and Z the preimage of the fiber under a deterministic generator, the
pushforward is the co-area target pi* of the plan (Sec. 3.2); gram=False drops the Gram factor (ablation).

One step:  tangent proposal v ~ N(0, s^2 P_T) at z;  y = z + v + Q a with h(y) = 0 (Newton in a, Q an
orthonormal basis of the normal space at z);  reverse move from y and the check that it returns to z;
Metropolis-Hastings acceptance with the Gram factor. States live in a linear subspace (symmetric octants):
`sym` projects onto it.
"""
import numpy as np
import jax.numpy as jnp


class ZHG:
    def __init__(self, cj, logp=None, sym=None, gram=True, newton_tol=1e-9, newton_maxit=40, rev_tol=1e-6,
                 tilt=None):
        """cj(z) -> (h (m,), J (m, *z.shape)); logp(z) -> float (default standard normal); sym: projector."""
        self.cj = cj
        self.logp = logp if logp is not None else (lambda z: -0.5 * float(jnp.vdot(z, z)))
        self.sym = sym if sym is not None else (lambda u: u)
        self.gram, self.tol, self.maxit, self.rev_tol = gram, newton_tol, newton_maxit, rev_tol
        self.tilt = tilt
        self.stats = dict(steps=0, accepted=0, fwd_fail=0, rev_fail=0)

    @staticmethod
    def basis(J):
        """Orthonormal basis Q (n, m) of the row space of J and log det(J J^T)."""
        Jf = np.asarray(J).reshape(J.shape[0], -1)
        Q, R = np.linalg.qr(Jf.T)
        return Q, 2.0 * float(np.sum(np.log(np.abs(np.diag(R)))))

    def newton(self, y0, Q, shape):
        """Solve h(y0 + Q a) = 0 for a (chord Newton with the Jacobian at each iterate projected on Q)."""
        a = np.zeros(Q.shape[1])
        for _ in range(self.maxit):
            y = y0 + jnp.asarray((Q @ a).reshape(shape))
            h, J = self.cj(y)
            h = np.asarray(h)
            if np.linalg.norm(h) <= self.tol:
                return y, h, J, True
            M = np.asarray(J).reshape(J.shape[0], -1) @ Q
            try:
                a = a - np.linalg.solve(M, h)
            except np.linalg.LinAlgError:
                return y, h, J, False
            if not np.all(np.isfinite(a)) or np.linalg.norm(a) > 1e6:
                return y, h, J, False
        y = y0 + jnp.asarray((Q @ a).reshape(shape))
        h, J = self.cj(y)
        return y, np.asarray(h), J, bool(np.linalg.norm(np.asarray(h)) <= self.tol)

    def tangent_noise(self, Q, shape, s, rng):
        v = np.asarray(self.sym(jnp.asarray(rng.standard_normal(shape)))).ravel()
        v = v - Q @ (Q.T @ v)
        return s * v

    def step(self, z, J, s, rng):
        """One move from z (on Z, with its Jacobian J). Returns (z', J', accepted)."""
        self.stats["steps"] += 1
        shape = z.shape
        Q, ld = self.basis(J)
        v = self.tangent_noise(Q, shape, s, rng)
        y, h1, J1, ok = self.newton(z + jnp.asarray(v.reshape(shape)), Q, shape)
        if not ok:
            self.stats["fwd_fail"] += 1
            return z, J, False
        Q1, ld1 = self.basis(J1)
        d = np.asarray(z - y).ravel()
        vr = d - Q1 @ (Q1.T @ d)  # reverse tangent move
        zb, _, _, okr = self.newton(y + jnp.asarray(vr.reshape(shape)), Q1, shape)
        if not okr or float(jnp.linalg.norm(zb - z)) > self.rev_tol * max(1.0, float(jnp.linalg.norm(z))):
            self.stats["rev_fail"] += 1
            return z, J, False
        logA = self.logp(y) - self.logp(z) - (vr @ vr - v @ v) / (2 * s * s)
        if self.gram:
            logA -= 0.5 * (ld1 - ld)
        if self.tilt is not None:
            logA -= self.tilt(y) - self.tilt(z)
        if np.log(rng.random()) < logA:
            self.stats["accepted"] += 1
            return y, J1, True
        return z, J, False

    def run(self, z, n, s, seed=0, adapt=0, target_acc=0.3, callback=None):
        """n steps from z (on Z). During the first `adapt` steps the step size follows the acceptance rate.
        Returns (chain list of states, final step size)."""
        rng = np.random.default_rng(seed)
        _, J = self.cj(z)
        chain = []
        acc_hist = []
        for k in range(n):
            z, J, a = self.step(z, J, s, rng)
            acc_hist.append(a)
            if k < adapt and (k + 1) % 10 == 0:
                rate = np.mean(acc_hist[-10:])
                s *= np.exp(rate - target_acc)
            chain.append(z)
            if callback is not None:
                callback(k, z, a)
        return chain, s
