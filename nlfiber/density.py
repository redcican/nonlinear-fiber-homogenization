"""Pre-density phi -> physical density rho with exact volume.

    u   = F mirror(sym_perm(phi))          periodic filter on the full cell
    rho = H_beta(u - eta),  H_beta(t) = sigmoid(beta t)
    eta : mean(rho) = v exactly (bisection, then Newton)

The derivative of eta follows from the volume condition: d eta = d^T du / 1^T d with d = H'_beta(u - eta),
so J_phi = J_rho D (I - 1 d^T / 1^T d) F (the projection of Sec. 3.1 of the plan). `threshold` implements
this with a stop-gradient construction whose value is exact and whose first derivative is the implicit one.
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from . import symmetry as S


def hat_kernel_fft(shape, r):
    """rfftn of the normalized periodic hat (cone) filter of radius r voxels on a grid of `shape`."""
    grids = np.meshgrid(*[np.minimum(np.arange(n), n - np.arange(n)) for n in shape], indexing="ij")
    dist = np.sqrt(sum(g.astype(np.float64) ** 2 for g in grids))
    k = np.maximum(0.0, r - dist)
    if k.sum() == 0:
        k[(0,) * len(shape)] = 1.0
    k /= k.sum()
    return jnp.asarray(np.fft.rfftn(k))


def filt(x, kfft):
    """Periodic convolution of a full-cell field with a kernel given by its rfftn."""
    return jnp.fft.irfftn(jnp.fft.rfftn(x) * kfft, s=x.shape)


def _vol(u, eta, beta):
    return jnp.mean(jax.nn.sigmoid(beta * (u - eta)))


@partial(jax.jit, static_argnums=())
def solve_eta(u, v, beta):
    """eta with mean(sigmoid(beta (u - eta))) = v: bisection to the bracket width 1e-12, then 3 Newton steps."""
    lo = jnp.min(u) - 40.0 / beta
    hi = jnp.max(u) + 40.0 / beta

    def body(_, b):
        lo, hi = b
        mid = 0.5 * (lo + hi)
        big = _vol(u, mid, beta) > v  # volume too large -> raise eta
        return jnp.where(big, mid, lo), jnp.where(big, hi, mid)

    lo, hi = jax.lax.fori_loop(0, 200, body, (lo, hi))
    eta = 0.5 * (lo + hi)

    def newton(_, eta):
        s = jax.nn.sigmoid(beta * (u - eta))
        dv = -jnp.mean(beta * s * (1 - s))
        return eta - (jnp.mean(s) - v) / jnp.where(dv == 0, -1.0, dv)

    return jax.lax.fori_loop(0, 3, newton, eta)


def threshold(u, v, beta):
    """rho = H_beta(u - eta(u)) with mean(rho) = v; differentiable with the implicit d eta."""
    eta0 = jax.lax.stop_gradient(solve_eta(jax.lax.stop_gradient(u), v, beta))
    s0 = jax.nn.sigmoid(beta * (jax.lax.stop_gradient(u) - eta0))
    d = s0 * (1 - s0)
    eta = eta0 + jnp.sum(d * (u - jax.lax.stop_gradient(u))) / jnp.sum(d)
    return jax.nn.sigmoid(beta * (u - eta))


class DensityMap:
    """phi (symmetric octant, (n_o)^d) -> rho (full cell, (2 n_o)^d) with volume v at sharpness beta."""

    def __init__(self, n_o, d=3, r=1.5, v=0.3):
        self.n_o, self.d, self.r, self.v = n_o, d, r, v
        self.n = 2 * n_o
        self.shape = (self.n,) * d
        self.kfft = hat_kernel_fft(self.shape, r)

    def smooth(self, phi):
        """Filtered full-cell field u = F mirror(sym_perm(phi))."""
        return S.exact_symmetrize(filt(S.mirror(S.sym_perm(phi)), self.kfft))

    def rho(self, phi, beta):
        return threshold(self.smooth(phi), self.v, beta)

    def eta(self, phi, beta):
        return float(solve_eta(self.smooth(phi), self.v, beta))

    def binary(self, phi, up=1, v=None, remove_floating=True):
        """Binary cell: u upsampled by `up` (periodic trilinear), thresholded so that the solid fraction is v
        (to one voxel), floating pieces removed. Returns a bool array of side up * 2 n_o."""
        from .topology import remove_floating as rf

        v = self.v if v is None else v
        u = np.asarray(self.smooth(phi), dtype=np.float64)
        if up > 1:
            u = np.asarray(S.exact_symmetrize(jnp.asarray(upsample_periodic(u, up))))
        b = u > np.quantile(u, 1.0 - v)
        if remove_floating:
            # removing floating pieces lowers the volume: lower the threshold until the kept volume is v again
            target = v
            for _ in range(6):
                b = rf(b)
                missing = target - b.mean()
                if missing * b.size < 1 or not b.any():
                    break
                cand = np.sort(u[~b])[::-1]
                k = int(round(missing * b.size))
                b = b | (u >= cand[min(k, cand.size) - 1])
            b = rf(b)
        return b


def upsample_periodic(c, f):
    """Periodic multilinear resampling of a cell-centred field by a factor f (integer or not, e.g. 1.5);
    keeps the symmetry when the new side is even."""
    out = np.asarray(c, dtype=np.float64)
    for ax in range(out.ndim):
        n = out.shape[ax]
        m = int(round(n * f))
        x = (np.arange(m) + 0.5) * n / m - 0.5
        i0 = np.floor(x).astype(int)
        w = x - i0
        a = np.take(out, i0 % n, axis=ax)
        b = np.take(out, (i0 + 1) % n, axis=ax)
        shp = [1] * out.ndim
        shp[ax] = m
        w = w.reshape(shp)
        out = (1 - w) * a + w * b
    return out
