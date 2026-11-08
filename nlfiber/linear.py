"""Small-strain periodic homogenization on voxel grids (JAX), with exact sensitivities.

Unit cell of nx x ny x nz trilinear hexahedra (Q1, 2x2x2 Gauss points) of unit size, periodic.
2D cells (n, n) are solved as (n, n, 1): one periodic element layer gives plane strain exactly.
Stiffness of element e: s_e E with s = emin + (1 - emin) rho^p (SIMP, as in the GPU solver of Zhang et al.).
Voigt order xx yy zz yz xz xy, engineering shear (homo3d's convention).

For a macroscopic strain E_I the periodic corrector u_I solves K u_I = -f_I, and
    C_IJ = (1/N_e) sum_e s_e b_e(E_I + u_I, E_J + u_J),
with b_e the element bilinear form. Because u_I minimizes the energy, dC/ds_e is the element term with u fixed,
so the first derivative of C (and of any function of C) comes from autodiff with the correctors held fixed
(`stop_gradient`): one solve, no adjoint. Functions that depend on the correctors in other ways (third-order
constants) use `correctors`, whose custom VJP solves the adjoint system.
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

VOIGT = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
NODES = np.array([[a, b, c] for c in (0, 1) for b in (0, 1) for a in (0, 1)])  # x fastest
_g = 0.5 + np.array([-1.0, 1.0]) / (2 * np.sqrt(3.0))
GPTS = np.array([[x, y, z] for z in _g for y in _g for x in _g])


def clip01(x):
    """Clamp to [0, 1] with derivative 1 at the bounds (jnp.clip splits its gradient at a tie, which halves the
    derivative on exactly binary voxels)."""
    return jnp.where(x > 1.0, 1.0, jnp.where(x < 0.0, 0.0, x))


def dN_ref():
    """Shape-function gradients of the unit Q1 element: [gauss, node, dim]."""
    d = np.zeros((8, 8, 3))
    for gi, (x, y, z) in enumerate(GPTS):
        for a, (i, j, k) in enumerate(NODES):
            fx, fy, fz = (x if i else 1 - x), (y if j else 1 - y), (z if k else 1 - z)
            sx, sy, sz = (1 if i else -1), (1 if j else -1), (1 if k else -1)
            d[gi, a] = [sx * fy * fz, fx * sy * fz, fx * fy * sz]
    return d


def lame(E, nu):
    return E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))


def iso_D(E, nu):
    lam, mu = lame(E, nu)
    D = np.zeros((6, 6))
    D[:3, :3] = lam
    D[[0, 1, 2], [0, 1, 2]] = lam + 2 * mu
    D[[3, 4, 5], [3, 4, 5]] = mu
    return D


def B_matrices():
    """Strain-displacement matrices (8 Gauss points, 6, 24), engineering shear."""
    dN = dN_ref()
    B = np.zeros((8, 6, 24))
    for g in range(8):
        for a in range(8):
            dx, dy, dz = dN[g, a]
            c = 3 * a
            B[g, 0, c] = dx
            B[g, 1, c + 1] = dy
            B[g, 2, c + 2] = dz
            B[g, 3, c + 1], B[g, 3, c + 2] = dz, dy
            B[g, 4, c + 0], B[g, 4, c + 2] = dz, dx
            B[g, 5, c + 0], B[g, 5, c + 1] = dy, dx
    return B


def unit_strain(I):
    e = np.zeros(6)
    e[I] = 1.0
    return e


def gather(u):
    """Nodal field (..., nx, ny, nz, 3) -> element vectors (..., nx, ny, nz, 24)."""
    ax = (-4, -3, -2)
    return jnp.concatenate([jnp.roll(u, shift=tuple(-o for o in off), axis=ax) for off in NODES], axis=-1)


def scatter(fe):
    """Element vectors (..., nx, ny, nz, 24) -> assembled nodal field (..., nx, ny, nz, 3)."""
    ax = (-4, -3, -2)
    out = 0
    for a, off in enumerate(NODES):
        out = out + jnp.roll(fe[..., 3 * a:3 * a + 3], shift=tuple(off), axis=ax)
    return out


def _proj(u):
    return u - jnp.mean(u, axis=(-4, -3, -2), keepdims=True)


class LinearHomogenization:
    """Linear homogenization on a fixed grid. `shape` is (nx, ny, nz) or (n, n) for plane strain."""

    def __init__(self, shape, E=1.0, nu=0.3, emin=1e-3, p=3.0, dtype=jnp.float64, tol=1e-9, maxit=4000):
        shape = tuple(shape)
        self.plane = len(shape) == 2
        self.shape = shape + (1,) if self.plane else shape
        self.E, self.nu, self.emin, self.p = E, nu, emin, p
        self.dtype, self.tol, self.maxit = dtype, tol, maxit
        B = B_matrices()
        D = iso_D(E, nu)
        self.Dm = jnp.asarray(D, dtype)
        self.Ke = jnp.asarray(sum(B[g].T @ D @ B[g] for g in range(8)) / 8.0, dtype)
        self.G = jnp.asarray(sum(D @ B[g] for g in range(8)) / 8.0, dtype)  # (6, 24)
        self._precond_symbol()
        self._solve = jax.jit(self._solve_impl)

        lh = self

        @jax.custom_vjp
        def correctors(s, Es, u0):
            u, _ = lh._solve(s, Es, u0)
            return u

        def fwd(s, Es, u0):
            u, _ = lh._solve(s, Es, u0)
            return u, (s, Es, u)

        def bwd(res, ubar):
            s, Es, u = res
            lam, _ = lh._solve_rhs(s, -_proj(ubar))  # K lam = ubar
            _, vjp = jax.vjp(lambda s_: lh.residual(s_, Es, u), s)
            return (vjp(lam)[0], jnp.zeros_like(Es), jnp.zeros_like(u))

        correctors.defvjp(fwd, bwd)
        self.correctors = correctors

    # --- operators ---------------------------------------------------------------------
    def stiffness_scale(self, rho):
        rho = jnp.asarray(rho)
        if self.plane and rho.ndim == 2:
            rho = rho[..., None]
        return self.emin + (1.0 - self.emin) * clip01(rho) ** self.p

    def apply_K(self, s, u):
        return scatter(s[..., None] * (gather(u) @ self.Ke))

    def load(self, s, Es):
        """Assembled a(E_I, .) for macroscopic strains Es (m, 6): (m, nx, ny, nz, 3)."""
        fe = jnp.einsum("mi,ij->mj", Es, self.G)  # (m, 24)
        return scatter(s[None, ..., None] * fe[:, None, None, None, :])

    def residual(self, s, Es, u):
        return _proj(jax.vmap(lambda x: self.apply_K(s, x))(u) + self.load(s, Es))

    def _precond_symbol(self):
        nx, ny, nz = self.shape
        s1 = jnp.ones(self.shape, self.dtype)
        cols = []
        for c in range(3):
            e = jnp.zeros(self.shape + (3,), self.dtype).at[0, 0, 0, c].set(1.0)
            cols.append(jnp.fft.rfftn(self.apply_K(s1, e), axes=(0, 1, 2)))  # (nx, ny, nz//2+1, 3)
        Kh = jnp.stack(cols, axis=-1)  # [..., a, c]
        eye = jnp.eye(3, dtype=Kh.dtype)
        Kh = Kh.at[0, 0, 0].set(eye)
        Kinv = jnp.linalg.inv(Kh + 1e-12 * jnp.abs(jnp.trace(Kh, axis1=-2, axis2=-1))[..., None, None] * eye)
        self.Kinv = Kinv.at[0, 0, 0].set(0.0)

    def precond(self, r):
        rh = jnp.fft.rfftn(r, axes=(-4, -3, -2))
        zh = jnp.einsum("...ac,...c->...a", self.Kinv, rh)
        return jnp.fft.irfftn(zh, s=self.shape, axes=(-4, -3, -2)).astype(r.dtype)

    # --- solver ------------------------------------------------------------------------
    def _cg(self, s, b, x0):
        """Batched PCG for K x_m = b_m on the mean-free subspace. Returns (x, iterations)."""
        A = jax.vmap(lambda x: self.apply_K(s, x))
        b = _proj(b)
        bn = jnp.sqrt(jnp.sum(b * b, axis=(1, 2, 3, 4))) + 1e-300
        x = _proj(x0)
        r = b - _proj(A(x))
        z = _proj(self.precond(r))
        p = z
        rz = jnp.sum(r * z, axis=(1, 2, 3, 4))
        bc = lambda a: a[:, None, None, None, None]  # noqa: E731

        def cond(st):
            x, r, z, p, rz, it = st
            rn = jnp.sqrt(jnp.sum(r * r, axis=(1, 2, 3, 4)))
            return (it < self.maxit) & jnp.any(rn > self.tol * bn)

        def body(st):
            x, r, z, p, rz, it = st
            Ap = _proj(A(p))
            pAp = jnp.sum(p * Ap, axis=(1, 2, 3, 4))
            a = rz / jnp.where(pAp == 0, 1.0, pAp)
            x = x + bc(a) * p
            r = r - bc(a) * Ap
            z = _proj(self.precond(r))
            rz1 = jnp.sum(r * z, axis=(1, 2, 3, 4))
            p = z + bc(rz1 / jnp.where(rz == 0, 1.0, rz)) * p
            return x, r, z, p, rz1, it + 1

        x, r, z, p, rz, it = jax.lax.while_loop(cond, body, (x, r, z, p, rz, 0))
        return x, it

    def _solve_impl(self, s, Es, u0):
        return self._cg(s, -self.load(s, Es), u0)

    @partial(jax.jit, static_argnums=(0,))
    def _solve_rhs(self, s, b):
        return self._cg(s, b, jnp.zeros_like(b))

    def solve(self, s, Es, u0=None):
        Es = jnp.asarray(Es, self.dtype)
        if u0 is None:
            u0 = jnp.zeros((Es.shape[0],) + self.shape + (3,), self.dtype)
        u, it = self._solve(jnp.asarray(s, self.dtype), Es, u0)
        return u, it  # it is a traced value inside jit

    # --- homogenized stiffness ---------------------------------------------------------
    def bilinear(self, s, Es, u):
        """C_IJ = mean_e s_e b_e(E_I + u_I, E_J + u_J) for the cases in Es (m, 6) with correctors u."""
        ue = gather(u)  # (m, nx, ny, nz, 24)
        KU = ue @ self.Ke
        GU = ue @ self.G.T  # (m, ..., 6)
        EDE = Es @ self.Dm @ Es.T
        t1 = jnp.einsum("ie,jxyze->ijxyz", Es, GU)  # E_I . G u_J
        t2 = jnp.einsum("ixyze,jxyze->ijxyz", ue, KU)
        b = EDE[:, :, None, None, None] + t1 + jnp.swapaxes(t1, 0, 1) + t2
        return jnp.mean(s[None, None] * b, axis=(2, 3, 4))

    def C_full(self, rho, u0=None):
        """6x6 C^H (all six unit strains). Returns (C, u, cg iterations)."""
        s = self.stiffness_scale(rho)
        Es = jnp.eye(6, dtype=self.dtype)
        u, it = self.solve(s, Es, u0)
        return self.bilinear(s, Es, u), u, it

    # cubic (3D) or square (2D) symmetric cells: solve xx and the shear, get yy by the x<->y swap
    def _shear(self):
        return 5 if self.plane else 3

    def sym_cases(self):
        return jnp.stack([jnp.asarray(unit_strain(0)), jnp.asarray(unit_strain(self._shear()))]).astype(self.dtype)

    @staticmethod
    def swap_xy(u):
        """Corrector of E_yy from the corrector of E_xx of a cell symmetric under x<->y."""
        return jnp.swapaxes(u, -4, -3)[..., jnp.array([1, 0, 2])]

    def invariants_from(self, s, u2):
        """(C11, C12, C44) / E (C66 in 2D) from the xx and shear correctors u2 (2, nx, ny, nz, 3)."""
        uyy = self.swap_xy(u2[0])
        u3 = jnp.stack([u2[0], uyy, u2[1]])
        Es = jnp.stack([jnp.asarray(unit_strain(0)), jnp.asarray(unit_strain(1)),
                        jnp.asarray(unit_strain(self._shear()))]).astype(self.dtype)
        C = self.bilinear(s, Es, u3)
        return jnp.stack([C[0, 0], C[1, 0], C[2, 2]]) / self.E

    def invariants(self, rho, u0=None):
        """Normalized invariants of a symmetric cell. Returns (c, u2, cg iterations).
        c is differentiable in rho (correctors held fixed, which is exact)."""
        s = self.stiffness_scale(rho)
        u2, it = self.solve(jax.lax.stop_gradient(s), self.sym_cases(), u0)
        return self.invariants_from(s, jax.lax.stop_gradient(u2)), u2, it


def cubic_invariants(C):
    """(C11, C12, C44) as averages over the cubic-equivalent entries of a 6x6 C."""
    C = np.asarray(C)
    return np.array([np.mean([C[0, 0], C[1, 1], C[2, 2]]), np.mean([C[0, 1], C[0, 2], C[1, 2]]),
                     np.mean([C[3, 3], C[4, 4], C[5, 5]])])


def cubic_leakage(C):
    """Relative distance of C from its cubic projection (full-tensor leakage)."""
    C = np.asarray(C)
    c11, c12, c44 = cubic_invariants(C)
    Cc = np.zeros((6, 6))
    Cc[:3, :3] = c12
    Cc[[0, 1, 2], [0, 1, 2]] = c11
    Cc[[3, 4, 5], [3, 4, 5]] = c44
    return float(np.linalg.norm(C - Cc) / np.linalg.norm(C))


def hashin_shtrikman(v, E=1.0, nu=0.3):
    """HS upper bounds (K_HS, G_HS) of a porous isotropic solid at volume fraction v (void phase 0)."""
    lam, mu = lame(E, nu)
    K = lam + 2 * mu / 3
    K_hs = K + (1 - v) / (-1 / K + 3 * v / (3 * K + 4 * mu)) if v < 1 else K
    G_hs = mu + (1 - v) / (-1 / mu + 6 * v * (K + 2 * mu) / (5 * mu * (3 * K + 4 * mu))) if v < 1 else mu
    return K_hs, G_hs
