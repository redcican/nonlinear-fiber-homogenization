"""Projection metrics on symmetric octant fields.

A metric is a symmetric positive operator G applied as G(u) to arrays of the octant's shape. The Gauss-Newton
step in metric G moves along range(G J^T).

  euclid        G = I (PCFM-style projection)
  DCTMetric     stationary covariance of a mirror-symmetric periodic field: diagonal in the octant's DCT-II
                basis. With t given, the Gaussian-Tweedie posterior covariance of x1 given x_t for the path
                x_t = t x1 + (1 - t) x0, x0 ~ N(0, I):  gamma_k(t) = s_k (1-t)^2 / ((1-t)^2 + t^2 s_k).
  LearnedTweedie  ((1-t)^2 / t) * sym(dD/dx_t) for the network endpoint D(x) = x + (1 - t) v(x, t),
                symmetrized as (J + J^T) / 2 with one JVP and one VJP.
"""
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.fft import dctn, idctn

from . import symmetry as S


def euclid(u):
    return u


def dct(x):
    return dctn(x, type=2, norm="ortho")


def idct(x):
    return idctn(x, type=2, norm="ortho")


def spectrum_from_data(X):
    """Per-mode variance s_k of octant fields X (N, n_o, ...) in the DCT-II basis, symmetrized over axis
    permutations and floored at 1e-6 of its maximum."""
    X = jnp.asarray(X)
    Xc = X - X.mean(axis=0, keepdims=True)
    s = jnp.mean(jax.vmap(dct)(Xc) ** 2, axis=0)
    s = S.sym_perm(s)
    return jnp.maximum(s, 1e-6 * s.max())


def power_law_spectrum(n_o, d=3, alpha=2.0, k0=1.0):
    """s_k = (k0^2 + |k|^2)^(-alpha) (a smooth Gaussian prior when no data are available)."""
    k = np.meshgrid(*[np.arange(n_o)] * d, indexing="ij")
    kk = sum(x.astype(float) ** 2 for x in k)
    s = (k0**2 + kk) ** (-alpha)
    return jnp.asarray(s / s.max())


class DCTMetric:
    def __init__(self, spectrum):
        self.s = jnp.asarray(spectrum)

    def gamma(self, t=None):
        if t is None:
            return self.s
        a = (1.0 - t) ** 2
        return self.s * a / (a + t**2 * self.s)

    def at(self, t=None):
        g = self.gamma(t)
        return lambda u: idct(g * dct(u))

    def __call__(self, u):
        return self.at(None)(u)

    def sqrt_at(self, t=None):
        g = jnp.sqrt(self.gamma(t))
        return lambda u: idct(g * dct(u))


class LearnedTweedie:
    """Gamma_theta(t) for a velocity model v(x, t, cond)."""

    def __init__(self, velocity, cond=None):
        self.v = velocity
        self.cond = cond

    def at(self, x, t):
        D = lambda y: y + (1.0 - t) * self.v(y, t, self.cond)  # noqa: E731
        _, pull = jax.vjp(D, x)
        scale = (1.0 - t) ** 2 / max(t, 1e-6)

        def G(u):
            jv = jax.jvp(D, (x,), (u,))[1]
            vj = pull(u)[0]
            return S.sym_perm(0.5 * scale * (jv + vj))

        return G
