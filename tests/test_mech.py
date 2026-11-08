"""Finite-strain evaluator: small-strain limit, third-order constants, adjoint, stability, paths."""
import os
import sys
import unittest

import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nlfiber import linear as L, mech as Me, qoi as Q  # noqa: E402
from helpers import grey_cell, binary_cell  # noqa: E402

REF = os.environ.get("NLFIBER_REFERENCE")


def voigt_of(A):
    """6x6 Voigt matrix of a 4th-order tangent at F = I (engineering shear)."""
    C = np.zeros((6, 6))
    for I, (i, j) in enumerate(L.VOIGT):
        for J, (k, l) in enumerate(L.VOIGT):
            C[I, J] = A[i, j, k, l]
    return C


class TestSmallStrainLimit(unittest.TestCase):
    def setUp(self):
        self.rho = grey_cell(3, beta=3.0)  # 6^3 cell
        self.cell = Me.Cell(self.rho)
        self.lh = L.LinearHomogenization(self.rho.shape)
        C, self.chi, _ = self.lh.C_full(self.rho)
        self.C = np.asarray(C)

    def test_toec_second_order_is_linear_C(self):
        C2, T = Me.toec(self.cell, self.chi)
        np.testing.assert_allclose(np.asarray(C2), self.C, atol=1e-9 * np.abs(self.C).max())
        T = np.asarray(T)
        np.testing.assert_allclose(T, np.transpose(T, (1, 0, 2)), atol=1e-12)
        np.testing.assert_allclose(T, np.transpose(T, (0, 2, 1)), atol=1e-12)

    def test_effective_tangent_at_identity(self):
        A = Me.effective_tangent(self.cell, jnp.zeros(self.rho.shape + (3,)), np.eye(3))
        np.testing.assert_allclose(voigt_of(A), self.C, atol=1e-7 * np.abs(self.C).max())
        val, _ = Me.rank_one_min(A, m=200)
        self.assertGreater(val, 0)

    def test_toec_against_nonlinear_solves(self):
        """g(t) = Wbar(I + t n): g''' from finite differences of Pbar along the path vs T and C."""
        C2, T = Me.toec(self.cell, self.chi)
        C2, T = np.asarray(C2), np.asarray(T)
        n = np.array([1.0, -0.3, 0.2, 0.4, -0.2, 0.1])  # Voigt, engineering shear
        Hn = sum(n[I] * Me.voigt_tensor(I) for I in range(6))
        X = Me.to_voigt(Hn @ Hn)
        pred = np.einsum("ijk,i,j,k", T, n, n, n) + 3 * n @ C2 @ X  # f'''(0) along n
        full = Me.Mixed(self.cell, free=[])
        h = 0.004

        def dg(t):  # <Pbar(I + t Hn), Hn> = g'(t)
            w, F, log = full.solve(np.eye(3) + t * Hn, rtol=1e-12, cg_tol=1e-10)
            self.assertTrue(log["converged"])
            return float(np.sum(np.asarray(self.cell.mean_stress(w, F)) * Hn))

        g3 = (dg(h) - 2 * dg(0.0) + dg(-h)) / h**2
        self.assertAlmostEqual(g3, pred, delta=2e-3 * abs(pred))


class TestAdjointAndStability(unittest.TestCase):
    def test_grad_rho_of_poisson_ratio(self):
        rho = grey_cell(3, beta=3.0)
        cell = Me.Cell(rho)
        mix = Me.Mixed(cell, Q.LATERAL[0])
        e = 0.04
        Fb = np.diag([1 + e, 1.0, 1.0])

        def nu_of(c):
            m = Me.Mixed(c, Q.LATERAL[0])
            w, F, log = m.solve(Fb, rtol=1e-12, cg_tol=1e-10)
            return -(float(F[1, 1]) - 1) / e, log

        _, log = nu_of(cell)
        q = lambda z, F_, r: -(z[mix.nw] - 1.0) / e  # noqa: E731  (z[nw] is Fbar_yy)
        g, info = mix.grad_rho(log["z"], Fb, q)
        d = jnp.asarray(np.random.default_rng(0).standard_normal(rho.shape)) * (rho * (1 - rho))
        hstep = 1e-5
        fd = (nu_of(cell.with_rho(rho + hstep * d))[0] - nu_of(cell.with_rho(rho - hstep * d))[0]) / (2 * hstep)
        self.assertAlmostEqual(float(jnp.vdot(g, d)), fd, delta=1e-4 * abs(fd) + 1e-9)

    def test_lobpcg_matches_dense_spectrum(self):
        rho = grey_cell(2, beta=2.0)  # 4^3 cell: dense Hessian is small
        cell = Me.Cell(rho)
        mix = Me.Mixed(cell, Q.LATERAL[0])
        Fb = np.diag([0.95, 1.0, 1.0])
        w, F, log = mix.solve(Fb, rtol=1e-12)
        z = log["z"]
        vals, _, _ = mix.lowest_eigs(z, F, k=3, tol=1e-8)
        H = np.asarray(jax.hessian(lambda zz: mix.Ez(zz, jnp.asarray(F)))(z))
        dsc = mix._diag_scale()
        Hs = H * dsc[:, None] * dsc[None, :]
        # remove the rigid translations (they are the constraint space of LOBPCG)
        Tm = np.zeros((H.shape[0], 3))
        for i in range(3):
            Tm[i:mix.nw:3, i] = 1.0
        Y, _ = np.linalg.qr(Tm / dsc[:, None])
        Pm = np.eye(H.shape[0]) - Y @ Y.T
        ev = np.linalg.eigvalsh(Pm @ Hs @ Pm)
        ev = ev[np.abs(ev) > 1e-12 * np.abs(ev).max()]
        np.testing.assert_allclose(vals, np.sort(ev)[:3], rtol=1e-5)


class TestPaths(unittest.TestCase):
    def test_tension_path_qois(self):
        rho = binary_cell(4, v=0.35)
        cell = Me.Cell(rho)
        C, _, _ = L.LinearHomogenization(rho.shape).C_full(jnp.asarray(rho, float))
        rows = Q.run_path(cell, "tension", emax=0.06, de=0.02, stability=True)
        self.assertTrue(all(r["converged"] for r in rows))
        out = Q.path_qois(rows, Q.young_x(np.asarray(C)), at=(0.02, 0.06))
        self.assertAlmostEqual(out["nu2"], Q.nu0(np.asarray(C)), delta=0.02)
        self.assertTrue(np.isnan(out["eps_cell"]))  # stable in tension

    @unittest.skipUnless(REF, "set NLFIBER_REFERENCE to the companion reference solver directory")
    def test_path_against_reference_solver(self):
        sys.path.insert(0, REF)
        import oracle as O
        rho = np.asarray(grey_cell(3, beta=3.0))
        cell = Me.Cell(rho, wls=None)
        s = np.asarray(cell.s)
        ocell = O.Cell(s - 1e-9, E=1.0, nu=0.3, material="neohooke")
        omix = O.Mixed(ocell, [(1, 1), (2, 2)])
        mix = Me.Mixed(cell, Q.LATERAL[0])
        for e in (0.05, -0.05):
            Fb = np.diag([1 + e, 1.0, 1.0])
            w, F, _ = mix.solve(Fb, rtol=1e-11, cg_tol=1e-10)
            wo, Fo, _ = omix.solve(Fb, rtol=1e-11, cg_tol=1e-10)
            P = np.asarray(cell.mean_stress(w, F))
            Po = np.asarray(ocell.mean_stress(wo, Fo))
            # the reference solver works on the unit cube; stresses are per unit volume in both
            np.testing.assert_allclose(P, Po, atol=1e-7 * np.abs(Po).max())
            np.testing.assert_allclose(np.asarray(F), np.asarray(Fo), atol=1e-8)


if __name__ == "__main__":
    unittest.main()
