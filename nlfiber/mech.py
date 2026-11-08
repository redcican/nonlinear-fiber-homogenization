"""Periodic finite-strain homogenization on voxel grids (JAX, float64).

Adapted from the reference solver of the companion code (Q1 hexahedra, 2x2x2 Gauss points, energy
minimization by Newton-CG), generalized to any grid nx x ny x nz (2D cells as one periodic element layer:
exact plane strain) and extended with what the fiber study needs:

  * stiffness scale s = emin + (1 - emin) rho^p, the same as the linear model, so the small-strain limit
    of the cell is exactly its linear C^H (the fiber and the evaluator describe the same solid);
  * void interpolation of Wang, Lazarov, Sigmund & Jensen (2014): psi = psi_nl(F_g) - psi_lin(F_g) + psi_lin(F),
    F_g = I + gamma (F - I), gamma(rho) a smooth step (void elements cannot invert);
  * mixed macroscopic control (free components of Fbar with zero conjugate stress) with a stored Gauss-point
    tangent, an FFT preconditioner (the homogeneous linear operator) and Eisenstat-Walker forcing;
  * adjoint gradients of any scalar of the equilibrium state with respect to rho;
  * stability: lowest eigenvalues of the scaled cell Hessian (LOBPCG), the effective tangent A = d Pbar / d Fbar
    and its rank-one (strong) ellipticity over directions on the sphere;
  * third-order elastic constants from the linear correctors (2n+1 rule).

Units: voxel edge = 1; energies and stresses are cell averages (Pbar = d Wbar / d Fbar).
"""
import numpy as np
import jax
import jax.numpy as jnp
from scipy.sparse.linalg import LinearOperator, lobpcg

from .linear import NODES, dN_ref, lame, VOIGT, unit_strain as unit_voigt

DN = jnp.asarray(dN_ref())
I3 = jnp.eye(3)


def psi_linear(F, lam, mu):
    H = F - I3
    eps = 0.5 * (H + jnp.swapaxes(H, -1, -2))
    tr = jnp.trace(eps, axis1=-2, axis2=-1)
    return 0.5 * lam * tr**2 + mu * jnp.sum(eps * eps, axis=(-2, -1))


def psi_neohooke(F, lam, mu):
    J = jnp.linalg.det(F)
    I1 = jnp.sum(F * F, axis=(-2, -1))
    lnJ = jnp.log(J)
    return 0.5 * mu * (I1 - 3.0) - mu * lnJ + 0.5 * lam * lnJ**2


MATERIALS = {"linear": psi_linear, "neohooke": psi_neohooke}
STALL_RTOL = 1e-6


def voigt_tensor(I):
    """Symmetric 3x3 tensor of the Voigt unit strain I (engineering shear: off-diagonals 1/2)."""
    H = np.zeros((3, 3))
    i, j = VOIGT[I]
    if i == j:
        H[i, i] = 1.0
    else:
        H[i, j] = H[j, i] = 0.5
    return H


def to_voigt(M):
    """Voigt vector (engineering shear) of the symmetric part of a 3x3 matrix."""
    S = 0.5 * (M + M.T)
    return np.array([S[0, 0], S[1, 1], S[2, 2], 2 * S[1, 2], 2 * S[0, 2], 2 * S[0, 1]])


def _linesearch(E, gnorm, x, dx, E0, slope, gn):
    tol = 64 * np.finfo(np.float64).eps * max(abs(E0), 1e-300)
    t = 1.0
    while t > 1e-6:
        Et = E(x + t * dx)
        if jnp.isfinite(Et):
            Et = float(Et)
            if Et <= E0 + 1e-4 * t * slope:
                return t
            if Et <= E0 + tol and gnorm(x + t * dx) < gn:
                return t
        t *= 0.5
    return None


class Cell:
    """A periodic voxel cell. rho: design density (nx, ny, nz) or (n, n) in [0, 1]."""

    def __init__(self, rho, E=1.0, nu=0.3, material="neohooke", wls=(500.0, 0.01), emin=1e-3, p=3.0):
        rho = jnp.asarray(rho, dtype=jnp.float64)
        if rho.ndim == 2:
            rho = rho[..., None]
        self.rho, self.shape = rho, tuple(rho.shape)
        self.N = int(np.prod(self.shape))
        self.E, self.nu, self.material, self.wls, self.emin, self.p = E, nu, material, wls, emin, p
        self.lam, self.mu = lame(E, nu)
        self.psi = MATERIALS[material]
        self.wq = 1.0 / (8.0 * self.N)
        self.s = self.scale_of(rho)
        self.gamma = self.gamma_of(rho)
        self._jit()

    def scale_of(self, rho):
        return self.emin + (1.0 - self.emin) * jnp.clip(rho, 0.0, 1.0) ** self.p

    def gamma_of(self, rho):
        if self.wls is None:
            return None
        b, r0 = self.wls
        return (jnp.tanh(b * r0) + jnp.tanh(b * (rho - r0))) / (jnp.tanh(b * r0) + jnp.tanh(b * (1 - r0)))

    def with_rho(self, rho):
        return Cell(rho, self.E, self.nu, self.material, self.wls, self.emin, self.p)

    def tiled(self, k):
        """k x k x k supercell (2D: k x k x 1)."""
        reps = (k, k, 1) if self.shape[2] == 1 else (k, k, k)
        return Cell(jnp.tile(self.rho, reps), self.E, self.nu, self.material, self.wls, self.emin, self.p)

    # --- discretization ----------------------------------------------------------------
    def nodal(self, w):
        return jnp.stack([jnp.roll(w, shift=tuple(-o for o in off), axis=(0, 1, 2)) for off in NODES])

    def grad_w(self, w):
        """(8, nx, ny, nz, 3, 3): H[..., i, j] = d w_i / d x_j at each Gauss point."""
        return jnp.einsum("aijkc,gad->gijkcd", self.nodal(w), DN)

    def grad_w_g(self, w, g):
        return jnp.einsum("aijkc,ad->ijkcd", self.nodal(w), DN[g])

    def psi_eff(self, F, gamma):
        if gamma is None:
            return self.psi(F, self.lam, self.mu)
        g = gamma[..., None, None]
        Fg = I3 + g * (F - I3)
        return self.psi(Fg, self.lam, self.mu) - psi_linear(Fg, self.lam, self.mu) + psi_linear(F, self.lam, self.mu)

    def energy(self, w, Fbar, rho=None):
        if rho is None:
            s, gamma = self.s, self.gamma
        else:
            s, gamma = self.scale_of(rho), self.gamma_of(rho)

        def eg(w_, Fbar_, s_, gamma_, g):
            return jnp.sum(s_ * self.psi_eff(Fbar_ + self.grad_w_g(w_, g), gamma_))

        eg_c = jax.checkpoint(eg)
        tot, _ = jax.lax.scan(lambda acc, g: (acc + eg_c(w, Fbar, s, gamma, g), None), jnp.zeros(()), jnp.arange(8))
        return self.wq * tot

    def tangent_g(self, w, Fbar, g):
        """wq s d^2 psi / dF dF at Gauss point g: (nx, ny, nz, 3, 3, 3, 3)."""
        F = Fbar + self.grad_w_g(w, g)
        gam = jnp.ones(self.shape) if self.gamma is None else self.gamma
        if self.gamma is None:
            f = lambda F1, g1: self.psi(F1, self.lam, self.mu)  # noqa: E731
        else:
            f = lambda F1, g1: self.psi_eff(F1[None, None, None], g1[None, None, None])[0, 0, 0]  # noqa: E731
        hf = jax.vmap(jax.hessian(f, argnums=0))
        sc = (self.wq * self.s).reshape(-1, 1, 1, 1, 1)
        return (sc * hf(F.reshape(-1, 3, 3), gam.reshape(-1))).reshape(-1, 81).reshape(F.shape + (3, 3))

    def _jit(self):
        self._E = jax.jit(self.energy)
        self._g = jax.jit(jax.grad(self.energy, argnums=0))
        self._P = jax.jit(jax.grad(self.energy, argnums=1))
        self._fft_symbol()

    def _fft_symbol(self):
        """Inverse symbol of the homogeneous linear-elastic operator (unit stiffness), for preconditioning."""
        def e_lin(w):
            H = self.grad_w(w)
            return self.wq * jnp.sum(psi_linear(I3 + H, self.lam, self.mu))
        Kop = jax.grad(e_lin)
        cols = []
        for c in range(3):
            e = jnp.zeros(self.shape + (3,)).at[0, 0, 0, c].set(1.0)
            cols.append(jnp.fft.rfftn(Kop(e), axes=(0, 1, 2)))
        Kh = jnp.stack(cols, axis=-1)
        Kh = Kh.at[0, 0, 0].set(jnp.eye(3))
        Kinv = jnp.linalg.inv(Kh + 1e-12 * jnp.abs(jnp.trace(Kh, axis1=-2, axis2=-1))[..., None, None] * jnp.eye(3))
        self.Kinv = Kinv.at[0, 0, 0].set(0.0)

    def precond_w(self, r):
        rh = jnp.fft.rfftn(r, axes=(0, 1, 2))
        return jnp.fft.irfftn(jnp.einsum("...ac,...c->...a", self.Kinv, rh), s=self.shape, axes=(0, 1, 2))

    def mean_stress(self, w, Fbar):
        return self._P(w, jnp.asarray(Fbar, dtype=jnp.float64))


class Mixed:
    """Mixed macroscopic control: Fbar components in `free` are unknowns with zero conjugate stress, the rest
    are prescribed. free=[] is full control. Unknown z = (w, Fbar_free); solve min_z W(z)."""

    def __init__(self, cell, free=((1, 1), (2, 2)), forcing="ew"):
        self.c = cell
        self.free = [tuple(f) for f in free]
        self.forcing = forcing
        self.shape = cell.shape
        self.nw = 3 * cell.N
        k = self.k = len(self.free)
        self.idx = tuple(np.array(self.free).T) if k else None
        nw, shape, idx = self.nw, self.shape, self.idx

        def unpack(z, Fb):
            w = z[:nw].reshape(shape + (3,))
            return w, (Fb.at[idx].set(z[nw:]) if k else Fb)

        self.unpack = jax.jit(unpack)

        def Ez(z, Fb, rho=None):
            w, F = unpack(z, Fb)
            return cell.energy(w, F, rho)

        self.Ez = Ez
        self._E = jax.jit(lambda z, Fb: Ez(z, Fb))
        self._g = jax.jit(jax.grad(lambda z, Fb: Ez(z, Fb)))
        Esel = np.zeros((max(k, 1), 3, 3))
        for j, (a, b) in enumerate(self.free):
            Esel[j, a, b] = 1.0
        Esel = jnp.asarray(Esel[:k]) if k else jnp.zeros((0, 3, 3))

        def Lmap(v):
            dF = jnp.tensordot(v[nw:], Esel, axes=1) if k else jnp.zeros((3, 3))
            return dF + cell.grad_w(v[:nw].reshape(shape + (3,)))

        LT = jax.linear_transpose(Lmap, jnp.zeros(nw + k))

        def hvp_A(A, v):
            dF = Lmap(v)
            dP = jnp.stack([jnp.einsum("ijkabcd,ijkcd->ijkab", A[g], dF[g]) for g in range(8)])
            return LT(dP)[0]

        self.hvp_A = jax.jit(hvp_A)
        tg = jax.jit(lambda z, Fb, g: cell.tangent_g(*unpack(z, Fb), g))
        self.tangent = lambda z, Fb: tuple(tg(z, Fb, g) for g in range(8))
        self._cgA = jax.jit(self._cg_impl, static_argnums=(5,))

    def z0(self, Fb):
        Fb = jnp.asarray(Fb, dtype=jnp.float64)
        return jnp.concatenate([jnp.zeros(self.nw), Fb[self.idx] if self.k else jnp.zeros(0)])

    def proj(self, v):
        w = v[: self.nw].reshape(self.shape + (3,))
        w = w - jnp.mean(w, axis=(0, 1, 2))
        return jnp.concatenate([w.ravel(), v[self.nw:]])

    def macro_diag(self, A, Fb):
        """Diagonal of the Hessian in the macro unknowns (for the preconditioner)."""
        d = []
        for j in range(self.k):
            e = jnp.zeros(self.nw + self.k).at[self.nw + j].set(1.0)
            d.append(self.hvp_A(A, e)[self.nw + j])
        return jnp.abs(jnp.array(d)) + 1e-12

    def precond(self, r, dmac, swscale):
        w = self.c.precond_w(r[: self.nw].reshape(self.shape + (3,))) / swscale
        return jnp.concatenate([w.ravel(), r[self.nw:] / dmac])

    def _cg_impl(self, A, b, dmac, sws, tol, maxit):
        proj = self.proj
        M = lambda r: proj(self.precond(r, dmac, sws))  # noqa: E731
        b = proj(b)
        bn = jnp.linalg.norm(b) + 1e-300
        z0 = M(b)
        init = (jnp.zeros_like(b), b, z0, z0, jnp.vdot(b, z0), 0, False, False)

        def cond(st):
            return (~st[6]) & (st[5] < maxit)

        def body(st):
            x, r, zz, p, rz, it, done, conv = st
            Ap = proj(self.hvp_A(A, p))
            pAp = jnp.vdot(p, Ap)
            neg = pAp <= 0
            a = jnp.where(neg, 0.0, rz / jnp.where(neg, 1.0, pAp))
            x1, r1 = x + a * p, r - a * Ap
            conv1 = jnp.linalg.norm(r1) <= tol * bn
            z1 = M(r1)
            rz1 = jnp.vdot(r1, z1)
            p1 = z1 + (rz1 / jnp.where(rz == 0, 1.0, rz)) * p
            x1 = jnp.where(neg & (it == 0), z0, jnp.where(neg, x, x1))
            return (x1, r1, z1, p1, rz1, it + 1, neg | conv1, conv1 & ~neg)

        x, r, zz, p, rz, it, done, conv = jax.lax.while_loop(cond, body, init)
        return x, it, conv, done & ~conv

    def _sws(self):
        # scale of the preconditioner: geometric mean of the stiffness (contrast is handled by CG)
        return float(jnp.exp(jnp.mean(jnp.log(self.c.s))))

    def solve_linear(self, A, b, tol=1e-10, maxit=20000):
        """Solve H x = b with the stored tangent A (mean-free in w). Returns (x, iterations, converged)."""
        dmac = self.macro_diag(A, None) if self.k else jnp.zeros(0)
        x, it, conv, neg = self._cgA(A, b, dmac, self._sws(), tol, int(maxit))
        return x, int(it), bool(conv)

    def solve(self, Fbar, z0=None, rtol=1e-9, maxnewton=30, cg_tol=1e-6, cg_maxit=20000, g_ref=None):
        """Newton-CG. Fbar: prescribed components (free ones are the initial guess if z0 is None).
        Returns (w, Fbar_full, log) with log['z'] the state."""
        Fb = jnp.asarray(Fbar, dtype=jnp.float64)
        z = self.z0(Fb) if z0 is None else z0
        g0, log = g_ref, dict(newton=0, cg=0, converged=False, gnorm=[], negative_curvature=False)
        eta, gprev = None, None
        sws = self._sws()
        for k in range(maxnewton):
            g = self.proj(self._g(z, Fb))
            gn = float(jnp.linalg.norm(g))
            g0 = gn if g0 is None else g0
            log["gnorm"].append(gn)
            if gn <= rtol * max(g0, 1e-300) or gn < 1e-14:
                log["converged"] = True
                break
            tol = cg_tol
            if self.forcing == "ew":
                e_new = 0.1 if gprev is None else 0.9 * (gn / gprev) ** 2
                if eta is not None and 0.9 * eta**2 > 0.1:
                    e_new = max(e_new, 0.9 * eta**2)
                eta = min(e_new, 0.1)
                tol = max(eta, cg_tol, 0.5 * rtol * g0 / gn) if gprev is not None else max(eta, cg_tol)
            gprev = gn
            A = self.tangent(z, Fb)
            dmac = self.macro_diag(A, Fb) if self.k else jnp.zeros(0)
            dz, its, ok, neg = self._cgA(A, -g, dmac, sws, tol, int(cg_maxit))
            log["cg"] += int(its)
            log["negative_curvature"] |= bool(neg)
            E0, slope = float(self._E(z, Fb)), float(jnp.vdot(g, dz))
            if slope >= 0:
                dz = -self.proj(self.precond(g, dmac, sws))
                slope = float(jnp.vdot(g, dz))
            t = _linesearch(lambda x: self._E(x, Fb), lambda x: float(jnp.linalg.norm(self.proj(self._g(x, Fb)))),
                            z, dz, E0, slope, gn)
            log["newton"] = k + 1
            if t is None:
                log["linesearch_failed"] = True
                if gn <= STALL_RTOL * g0:
                    log["converged"] = log["roundoff_floor"] = True
                break
            z = z + t * dz
        w, F = self.unpack(z, Fb)
        log["z"], log["g0"] = z, g0
        return w, F, log

    # --- adjoint ------------------------------------------------------------------------
    def grad_rho(self, z, Fb, q, cg_tol=1e-10, cg_maxit=20000):
        """d q / d rho for a scalar q(z, Fb, rho) of the equilibrium state z(rho) at prescribed Fb.
        H lam = dq/dz;  dq/drho = dq/drho|_z - d/drho <lam, grad_z W(z; rho)>."""
        Fb = jnp.asarray(Fb, dtype=jnp.float64)
        rho = self.c.rho
        dq_dz = self.proj(jax.grad(q, argnums=0)(z, Fb, rho))
        A = self.tangent(z, Fb)
        lam, its, ok = self.solve_linear(A, dq_dz, cg_tol, cg_maxit)
        dq_dr = jax.grad(q, argnums=2)(z, Fb, rho)
        gz = lambda r_: jax.grad(lambda zz: self.Ez(zz, Fb, r_))(z)  # noqa: E731
        coupling = jax.grad(lambda r_: jnp.vdot(lam, gz(r_)))(rho)
        return dq_dr - coupling, dict(cg=its, converged=ok)

    # --- stability ------------------------------------------------------------------------
    def lowest_eigs(self, z, Fb, k=4, X=None, tol=1e-5, maxiter=2000):
        """Lowest eigenvalues of D^-1/2 H D^-1/2 (D the Hessian diagonal at the undeformed state, macro
        unknowns included), rigid translations excluded. Same inertia as H (Sylvester).
        Returns (values, vectors in z coordinates, iterations)."""
        Fb = jnp.asarray(Fb, dtype=jnp.float64)
        A = self.tangent(z, Fb)
        dsc = self._diag_scale()
        nz = self.nw + self.k
        T = np.zeros((nz, 3))
        for i in range(3):
            T[i:self.nw:3, i] = 1.0
        Y, _ = np.linalg.qr(T / dsc[:, None])
        hv = jax.jit(lambda V: jax.vmap(lambda v: self.hvp_A(A, v), in_axes=1, out_axes=1)(V))

        def mm(v):
            v = np.asarray(v, dtype=np.float64)
            V = jnp.asarray(v.reshape(nz, -1)) * jnp.asarray(dsc)[:, None]
            return (np.asarray(hv(V)) * dsc[:, None]).reshape(v.shape)

        op = LinearOperator((nz, nz), matvec=mm, matmat=mm, dtype=np.float64)
        if X is None:
            X = np.random.default_rng(0).standard_normal((nz, k))
        vals, vecs, hist = lobpcg(op, X, Y=Y, largest=False, tol=tol, maxiter=maxiter, retLambdaHistory=True)
        o = np.argsort(vals)
        return vals[o], vecs[:, o] * dsc[:, None], len(hist)

    def _diag_scale(self):
        if not hasattr(self, "_dsc"):
            Fb = jnp.eye(3)
            z = self.z0(Fb)
            A = self.tangent(z, Fb)
            # Jacobi diagonal of the w block by probing with the 3 * 8 node colours of the 2x2x2 coloring
            d = np.zeros(self.nw + self.k)
            idx = np.arange(self.c.N).reshape(self.shape)
            col = (np.indices(self.shape) % 2)
            col = col[0] + 2 * col[1] + 4 * col[2]
            for cc in range(8):
                for comp in range(3):
                    e = np.zeros(self.shape + (3,))
                    e[..., comp] = (col == cc)
                    v = jnp.concatenate([jnp.asarray(e.ravel()), jnp.zeros(self.k)])
                    hv = np.asarray(self.hvp_A(A, v))[: self.nw].reshape(self.shape + (3,))
                    d[: self.nw].reshape(self.shape + (3,))[..., comp][col == cc] = hv[..., comp][col == cc]
            for j in range(self.k):
                e = jnp.zeros(self.nw + self.k).at[self.nw + j].set(1.0)
                d[self.nw + j] = float(self.hvp_A(A, e)[self.nw + j])
            self._dsc = 1.0 / np.sqrt(np.abs(d) + 1e-30)
        return self._dsc


def effective_tangent(cell, w, Fbar, cg_tol=1e-10):
    """A_abcd = d Pbar_ab / d Fbar_cd at the equilibrium (w, Fbar) under full control (9 linear solves)."""
    full = Mixed(cell, free=[])
    Fb = jnp.asarray(Fbar, dtype=jnp.float64)
    z = jnp.asarray(w).ravel()
    Atan = full.tangent(z, Fb)
    P = lambda zz, F: jax.grad(lambda F_: cell.energy(zz.reshape(cell.shape + (3,)), F_))(F)  # noqa: E731
    gz = lambda zz, F: jax.grad(lambda z_: cell.energy(z_.reshape(cell.shape + (3,)), F))(zz)  # noqa: E731
    A = np.zeros((3, 3, 3, 3))
    for c in range(3):
        for d in range(3):
            E = jnp.zeros((3, 3)).at[c, d].set(1.0)
            dPF = jax.jvp(lambda F: P(z, F), (Fb,), (E,))[1]
            rhs = -jax.jvp(lambda F: gz(z, F), (Fb,), (E,))[1]  # H dz = -d(grad_z)/dF . E
            dz, _, _ = full.solve_linear(Atan, rhs, cg_tol)
            dPz = jax.jvp(lambda zz: P(zz, Fb), (z,), (dz,))[1]
            A[:, :, c, d] = np.asarray(dPF + dPz)
    return A


def fibonacci_sphere(m):
    i = np.arange(m) + 0.5
    phi = np.arccos(1 - 2 * i / m)
    th = np.pi * (1 + 5**0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], axis=1)


def rank_one_min(A, m=1000, plane=False):
    """min over unit n of the smallest eigenvalue of the acoustic tensor Q_ik = A_ijkl n_j n_l.
    Strong ellipticity is lost when it is <= 0. 2D cells: n in the x-y plane. Returns (value, n)."""
    if plane:
        th = np.linspace(0, np.pi, m, endpoint=False)
        ns = np.stack([np.cos(th), np.sin(th), np.zeros_like(th)], 1)
    else:
        ns = fibonacci_sphere(m)
    Q = np.einsum("ijkl,nj,nl->nik", A, ns, ns)
    Q = 0.5 * (Q + np.swapaxes(Q, 1, 2))
    if plane:
        Q = Q[:, :2, :2]
    ev = np.linalg.eigvalsh(Q)[:, 0]
    i = int(np.argmin(ev))
    return float(ev[i]), ns[i]


# --- third-order elastic constants ------------------------------------------------------------
def toec(cell, chi):
    """C (6x6) and Lagrangian-strain third-order constants T (6x6x6) of the cell, from the linear correctors
    chi (6, nx, ny, nz, 3) of unit Voigt strains (same stiffness scale and E, nu).
    f(s) = <psi(I + sum_I s_I (e_I + grad chi_I))> equals Wbar to O(s^4) (2n+1 rule), so
    C_IJ = f''_IJ and T_IJK = f'''_IJK - 3 Sym C[e_I, e_J e_K]. Differentiable in cell.rho via chi."""
    Hs = jnp.stack([jnp.asarray(voigt_tensor(I)) + cell.grad_w(chi[I]) for I in range(6)])  # (6,8,nx,ny,nz,3,3)

    def f(s):
        F = I3 + jnp.tensordot(s, Hs, axes=1)
        return cell.wq * jnp.sum(cell.s[None] * cell.psi_eff(F, cell.gamma[None] if cell.gamma is not None else None))

    z6 = jnp.zeros(6)
    eye = jnp.eye(6)
    C = jax.hessian(f)(z6)

    def d3(a, b, c):
        g2 = lambda s: jax.jvp(lambda s2: jax.jvp(f, (s2,), (a,))[1], (s,), (b,))[1]  # noqa: E731
        return jax.jvp(g2, (z6,), (c,))[1]

    T3 = jnp.zeros((6, 6, 6))
    for I in range(6):
        for J in range(I, 6):
            for K in range(J, 6):
                v = d3(eye[I], eye[J], eye[K])
                for p in {(I, J, K), (I, K, J), (J, I, K), (J, K, I), (K, I, J), (K, J, I)}:
                    T3 = T3.at[p].set(v)
    # geometric correction: 3 Sym_{IJK} C(e_I, voigt(e_J e_K))
    X = np.zeros((6, 6, 6))
    for J in range(6):
        for K in range(6):
            X[:, J, K] = to_voigt(voigt_tensor(J) @ voigt_tensor(K))
    X = jnp.asarray(X)
    CX = jnp.einsum("ia,ajk->ijk", C, X)
    sym = (CX + jnp.transpose(CX, (0, 2, 1)) + jnp.transpose(CX, (1, 0, 2)) + jnp.transpose(CX, (1, 2, 0))
           + jnp.transpose(CX, (2, 0, 1)) + jnp.transpose(CX, (2, 1, 0))) / 6.0
    return C, T3 - 3.0 * sym


CUBIC_TOEC = {
    "C111": [(0, 0, 0), (1, 1, 1), (2, 2, 2)],
    "C112": [(0, 0, 1), (0, 0, 2), (1, 1, 0), (1, 1, 2), (2, 2, 0), (2, 2, 1)],
    "C123": [(0, 1, 2)],
    "C144": [(0, 3, 3), (1, 4, 4), (2, 5, 5)],
    "C155": [(0, 4, 4), (0, 5, 5), (1, 3, 3), (1, 5, 5), (2, 3, 3), (2, 4, 4)],
    "C456": [(3, 4, 5)],
}


def cubic_toec_jnp(T):
    """Traceable version of cubic_toec (dict of scalars)."""
    return {k: jnp.mean(jnp.stack([T[i] for i in v])) for k, v in CUBIC_TOEC.items()}


def cubic_toec(T):
    """The six cubic third-order constants (C111, C112, C123, C144, C155, C456), averaged over equivalent entries
    (Voigt indices 1..6 = xx yy zz yz xz xy)."""
    T = np.asarray(T)
    return {k: float(np.mean([T[i] for i in v])) for k, v in CUBIC_TOEC.items()}
