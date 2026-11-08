"""The constraint map of the fiber: phi (symmetric octant) -> normalized stiffness invariants c(phi).

    M(c*, v) = { phi : c(rho(phi)) = c*, mean rho = v (exact by the threshold), symmetric by construction }

c = (C11, C12, C44) / E in 3D and (C11, C12, C66) / E in 2D (plane strain). The Jacobian rows are three
VJPs through the density map with the correctors held fixed (no adjoint solve).
"""
import jax
import jax.numpy as jnp
import numpy as np

from . import symmetry as S
from .density import DensityMap
from .linear import LinearHomogenization, cubic_invariants, cubic_leakage


class Fiber:
    def __init__(self, n_o, d=3, v=0.3, r=1.5, E=1.0, nu=0.3, emin=1e-3, p=3.0, tol=1e-9):
        self.n_o, self.d, self.v = n_o, d, v
        self.dm = DensityMap(n_o, d, r, v)
        self.lh = LinearHomogenization(self.dm.shape, E=E, nu=nu, emin=emin, p=p, tol=tol)
        self._mat = dict(E=E, nu=nu, emin=emin, p=p, tol=tol)
        self._lh_bin = {}
        self.u = None  # warm start of the correctors
        self.solves = 0

        def cfun(phi, beta, u0):
            c, u, it = self.lh.invariants(self.dm.rho(phi, beta), u0)
            return c, (u, it)

        self.cfun = cfun  # pure, differentiable in phi (for composition with a generator)
        self._val = jax.jit(cfun)

        def value_and_jac(phi, beta, u0):
            c, pull, aux = jax.vjp(lambda p: cfun(p, beta, u0), phi, has_aux=True)
            J = jax.vmap(lambda e: pull(e)[0])(jnp.eye(3, dtype=c.dtype))
            return c, J, aux

        self._vj = jax.jit(value_and_jac)

    def _u0(self):
        if self.u is None:
            return jnp.zeros((2,) + self.lh.shape + (3,), self.lh.dtype)
        return self.u

    def value(self, phi, beta, warm=True):
        c, (u, it) = self._val(jnp.asarray(phi), float(beta), self._u0() if warm else jnp.zeros_like(self._u0()))
        self.u = u
        self.solves += 1
        return np.asarray(c)

    def value_jac(self, phi, beta, warm=True):
        c, J, (u, it) = self._vj(jnp.asarray(phi), float(beta), self._u0() if warm else jnp.zeros_like(self._u0()))
        self.u = u
        self.solves += 1
        return np.asarray(c), J

    def rho(self, phi, beta):
        return self.dm.rho(jnp.asarray(phi), float(beta))

    # --- binary check ------------------------------------------------------------------
    def lh_for(self, shape):
        if shape not in self._lh_bin:
            self._lh_bin[shape] = LinearHomogenization(shape if self.d == 3 else shape[:2], **self._mat)
        return self._lh_bin[shape]

    def binary_check(self, phi, up=1, full=False):
        """Binarize (upsampled by `up`), drop floating pieces, re-solve. Returns dict(c, vol, cell[, C, leak])."""
        b = self.dm.binary(np.asarray(phi), up=up)
        lh = self.lh_for(b.shape)
        rho = jnp.asarray(b, dtype=jnp.float64)
        out = dict(vol=float(b.mean()), cell=b)
        if full:
            C, _, _ = lh.C_full(rho)
            C = np.asarray(C)
            out.update(C=C, c=cubic_invariants(C) / lh.E, leak=cubic_leakage(C))
        else:
            c, _, _ = lh.invariants(rho)
            out["c"] = np.asarray(c)
        self.solves += 1
        return out


def rel_error(c, target):
    c, t = np.asarray(c), np.asarray(target)
    return float(np.linalg.norm(c - t) / np.linalg.norm(t))


def gram_logdet(J, metric=None):
    """log det(J G J^T) for Jacobian rows J (3, ...) in a metric G (identity if None)."""
    Jf = jnp.reshape(J, (J.shape[0], -1))
    GJ = Jf if metric is None else jnp.reshape(jax.vmap(metric)(J), Jf.shape)
    return float(jnp.linalg.slogdet(Jf @ GJ.T)[1])


def sigma_min_normalized(J):
    """Smallest singular value of the row-normalized Jacobian (rank check of Prop. 1)."""
    Jf = np.asarray(jnp.reshape(J, (J.shape[0], -1)))
    Jf = Jf / np.linalg.norm(Jf, axis=1, keepdims=True)
    return float(np.linalg.svd(Jf, compute_uv=False).min())


def random_start(n_o, d=3, seed=0, kmax=3, nterms=40):
    """Random smooth symmetric pre-density (low-frequency cosine series), standardized."""
    rng = np.random.default_rng(seed)
    n = 2 * n_o
    x = (np.arange(n_o) + 0.5) / n
    f = np.zeros((n_o,) * d)
    for _ in range(nterms):
        k = rng.integers(0, kmax + 1, size=d)
        if not k.any():
            continue
        a = rng.standard_normal() / np.linalg.norm(k)
        term = np.ones(())
        for ax in range(d):
            shp = [1] * d
            shp[ax] = n_o
            term = term * np.cos(2 * np.pi * k[ax] * x).reshape(shp)
        f = f + a * term
    f = np.asarray(S.sym_perm(jnp.asarray(f)))
    return jnp.asarray((f - f.mean()) / (f.std() + 1e-12))
