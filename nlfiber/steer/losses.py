"""Nonlinear losses L(rho) with adjoint gradients, for null-space flow and tilted SMC (T3 targets).

  PoissonPath   sum over strains e_k of (nu(e_k) - nu_target(e_k))^2 under uniaxial stress along x;
                one adjoint solve per strain (nu is a function of the macro unknown Fbar_yy, Fbar_zz)
  StressPath    sum_k (Pbar_xx(e_k) / (E0 e_k) - r_target(e_k))^2: secant-ratio (stiffening/softening) target
  ToecTarget    squared distance of chosen cubic third-order constants to targets; gradient by autodiff through
                the linear correctors (custom VJP of the corrector solve)
Each loss returns (L, dL/drho) and keeps the last path states as warm starts.
"""
import numpy as np
import jax
import jax.numpy as jnp

from .. import mech as Me, qoi as Q
from ..linear import LinearHomogenization, cubic_invariants


class _Path:
    def __init__(self, strains, axis=0, E=1.0, nu=0.3, wls=(500.0, 0.01), emin=1e-3, p=3.0):
        self.strains = [float(e) for e in strains]
        self.axis = axis
        self.mat = dict(E=E, nu=nu, wls=wls, emin=emin, p=p)
        self.z = {}
        self.lh = None

    def states(self, rho):
        """Solve the path through the strains (warm-started). Returns {e: (mix, z, F)}."""
        cell = Me.Cell(rho, **self.mat)
        mix = Me.Mixed(cell, Q.LATERAL[self.axis])
        out = {}
        z = None
        for e in sorted(self.strains, key=abs):
            Fb = np.eye(3)
            Fb[self.axis, self.axis] = 1 + e
            if z is not None and mix.k:
                Fb = np.array(mix.unpack(z, jnp.asarray(Fb))[1])
                Fb[self.axis, self.axis] = 1 + e
            w, F, log = mix.solve(Fb, z0=self.z.get(e, z))
            for _ in range(3):
                if log["converged"] or log.get("linesearch_failed"):
                    break
                w, F, log = mix.solve(Fb, z0=log["z"], g_ref=log["g0"])
            # a state whose gradient fell by 1e-5 is accurate enough for the loss; only a stalled solve fails
            ok = bool(log["converged"]) or log["gnorm"][-1] <= 1e-5 * max(log["g0"], 1e-300)
            z = log["z"]
            self.z[e] = z
            out[e] = (mix, z, np.asarray(F), ok)
        return cell, out


class PoissonPath(_Path):
    def __init__(self, strains, nu_target, **kw):
        super().__init__(strains, **kw)
        self.nu_t = nu_target if callable(nu_target) else (lambda e: float(nu_target))

    def __call__(self, rho):
        rho = jnp.asarray(rho)
        cell, st = self.states(rho)
        L, g = 0.0, jnp.zeros(rho.shape)
        lat = [i for i in range(3) if i != self.axis]
        for e, (mix, z, F, ok) in st.items():
            if not ok:
                return float("inf"), g
            nu = np.mean([-(F[i, i] - 1) / e for i in lat])
            r = nu - self.nu_t(e)
            L += r * r

            def q(zz, Fb, rr, e=e, mix=mix):  # mean secant Poisson's ratio from the macro unknowns
                return -jnp.mean((zz[mix.nw:] - 1.0)) / e

            gq, _ = mix.grad_rho(z, jnp.asarray(F), q)
            g = g + 2 * r * gq
        return float(L), g


class StressPath(_Path):
    def __init__(self, strains, r_target, E0, **kw):
        super().__init__(strains, **kw)
        self.r_t = r_target if callable(r_target) else (lambda e: float(r_target))
        self.E0 = E0

    def __call__(self, rho):
        rho = jnp.asarray(rho)
        cell, st = self.states(rho)
        L, g = 0.0, jnp.zeros(rho.shape)
        a = self.axis
        for e, (mix, z, F, ok) in st.items():
            if not ok:
                return float("inf"), g
            P = np.asarray(cell.mean_stress(*mix.unpack(z, jnp.asarray(F))))
            r = P[a, a] / (self.E0 * e) - self.r_t(e)
            L += r * r

            def q(zz, Fb, rr, mix=mix, e=e):
                w, Ff = mix.unpack(zz, Fb)
                return jax.grad(cell.energy, argnums=1)(w, Ff, rr)[a, a] / (self.E0 * e)

            gq, _ = mix.grad_rho(z, jnp.asarray(F), q)
            g = g + 2 * r * gq
        return float(L), g


class ToecTarget:
    """L = sum_k w_k (T_k(rho) / E - t_k)^2 over named cubic constants, by autodiff through the correctors."""

    def __init__(self, targets, E=1.0, nu=0.3, emin=1e-3, p=3.0, wls=(500.0, 0.01)):
        self.targets = dict(targets)
        self.mat = dict(E=E, nu=nu, emin=emin, p=p)
        self.wls = wls
        self.lh = None

        def loss(rho):
            lh = self.lh
            s = lh.stiffness_scale(rho)
            chi = lh.correctors(s, jnp.eye(6), jnp.zeros((6,) + lh.shape + (3,)))
            cell = Me.Cell(rho, E=E, nu=nu, wls=wls, emin=emin, p=p)
            C, T = Me.toec(cell, chi)
            vals = Me.cubic_toec_jnp(T)
            return sum((vals[k] / E - t) ** 2 for k, t in self.targets.items())

        self._vg = jax.value_and_grad(loss)

    def __call__(self, rho):
        rho = jnp.asarray(rho)
        if self.lh is None or self.lh.shape != tuple(rho.shape):
            self.lh = LinearHomogenization(rho.shape, **self.mat)
        L, g = self._vg(rho)
        return float(L), g
