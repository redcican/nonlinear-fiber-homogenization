"""Symmetry, density map, linear homogenization, topology and projection."""
import os
import sys
import unittest

import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nlfiber import symmetry as S, density as D, linear as L, topology as T, fiber as Fb, project as P, metric as M  # noqa
from helpers import schwarz_p_phi, binary_cell, grey_cell  # noqa: E402

REF = os.environ.get("NLFIBER_REFERENCE")  # optional: path to the companion reference solver (oracle.py)


class TestSymmetry(unittest.TestCase):
    def test_mirror_fold_and_free_count(self):
        rng = np.random.default_rng(0)
        for d, n_o in ((3, 5), (2, 6)):
            x = S.sym_perm(jnp.asarray(rng.standard_normal((n_o,) * d)))
            full = S.mirror(x)
            self.assertLess(S.sym_residual(full), 1e-14)
            np.testing.assert_allclose(S.fold(full), x, atol=1e-14)
            v = S.to_free(x)
            self.assertEqual(v.size, S.n_free(n_o, d))
            np.testing.assert_allclose(S.from_free(v, n_o, d), x, atol=1e-14)
            # sym_perm is an orthogonal projection
            y = jnp.asarray(rng.standard_normal((n_o,) * d))
            self.assertAlmostEqual(float(jnp.vdot(S.sym_perm(y), x)), float(jnp.vdot(y, x)), places=10)

    def test_full_cell_has_all_symmetries(self):
        x = S.sym_perm(jnp.asarray(np.random.default_rng(1).standard_normal((4, 4, 4))))
        f = np.asarray(S.mirror(x))
        np.testing.assert_allclose(f, f[::-1], atol=0)
        np.testing.assert_allclose(f, np.transpose(f, (2, 0, 1)), atol=0)
        np.testing.assert_allclose(f, np.roll(f[::-1], 0, 0), atol=0)


class TestDensity(unittest.TestCase):
    def test_exact_volume_and_implicit_derivative(self):
        dm = D.DensityMap(4, r=1.5, v=0.27)
        phi = S.sym_perm(jnp.asarray(np.random.default_rng(2).standard_normal((4, 4, 4))))
        for beta in (2.0, 8.0, 32.0):
            self.assertAlmostEqual(float(dm.rho(phi, beta).mean()), 0.27, places=12)
        f = lambda p: jnp.sum(dm.rho(p, 4.0) ** 3)  # noqa: E731
        g = jax.grad(f)(phi)
        d = S.sym_perm(jnp.asarray(np.random.default_rng(3).standard_normal((4, 4, 4))))
        h = 1e-6
        fd = (f(phi + h * d) - f(phi - h * d)) / (2 * h)
        self.assertAlmostEqual(float(jnp.vdot(g, d)), float(fd), delta=1e-7 * abs(float(fd)) + 1e-10)

    def test_binary_volume_after_floating_removal(self):
        # smooth fields at a large filter radius leave floating pieces at the threshold; the kept solid must
        # grow back to v instead of re-adding the removed pieces
        fib = Fb.Fiber(8, v=0.25, r=2.5)
        kept = 0
        for seed in range(8):
            b = fib.dm.binary(Fb.random_start(8, seed=seed, kmax=3), up=1)
            if not b.any():  # no spanning solid at the threshold: nothing to keep
                continue
            kept += 1
            self.assertLessEqual(abs(b.mean() - 0.25) * b.size, 48)  # to one symmetry orbit (48 voxels)
            self.assertEqual(T.validity(b)["floating"], 0)
        self.assertGreaterEqual(kept, 2)

    def test_binary_volume_and_symmetry(self):
        dm = D.DensityMap(6, r=1.5, v=0.3)
        b = dm.binary(np.asarray(schwarz_p_phi(6)), up=2)
        self.assertEqual(b.shape, (24,) * 3)
        self.assertLess(abs(b.mean() - 0.3), 48.0 / b.size)  # one orbit of the 48 symmetries
        self.assertLess(S.sym_residual(jnp.asarray(b, float)), 1e-14)


class TestLinear(unittest.TestCase):
    def test_full_vs_cubic_path_and_symmetry(self):
        rho = grey_cell(4)
        lh = L.LinearHomogenization(rho.shape)
        C, _, _ = lh.C_full(rho)
        C = np.asarray(C)
        np.testing.assert_allclose(C, C.T, atol=1e-12)
        self.assertLess(L.cubic_leakage(C), 1e-8)
        c, _, _ = lh.invariants(rho)
        np.testing.assert_allclose(np.asarray(c), L.cubic_invariants(C), rtol=1e-8)

    def test_uniform_solid_is_isotropic_material(self):
        lh = L.LinearHomogenization((4, 4, 4), E=2.0, nu=0.25)
        C, _, _ = lh.C_full(jnp.ones((4, 4, 4)))
        np.testing.assert_allclose(np.asarray(C), L.iso_D(2.0, 0.25), atol=1e-10)

    def test_plane_strain_equals_extruded_3d(self):
        rho2 = np.asarray(grey_cell(4, d=2))
        c2, _, _ = L.LinearHomogenization(rho2.shape).invariants(rho2)
        rho3 = np.repeat(rho2[..., None], 3, axis=2)
        C3, _, _ = L.LinearHomogenization(rho3.shape).C_full(rho3)
        C3 = np.asarray(C3)
        np.testing.assert_allclose(np.asarray(c2), [C3[0, 0], C3[1, 0], C3[5, 5]], rtol=1e-8)

    def test_invariant_gradient(self):
        fib = Fb.Fiber(4, v=0.3, r=1.5)
        phi = S.sym_perm(Fb.random_start(4, seed=4, kmax=1) + schwarz_p_phi(4))
        c, J = fib.value_jac(phi, 4.0)
        d = S.sym_perm(jnp.asarray(np.random.default_rng(5).standard_normal((4, 4, 4))))
        h = 1e-5
        fd = (fib.value(phi + h * d, 4.0, warm=False) - fib.value(phi - h * d, 4.0, warm=False)) / (2 * h)
        np.testing.assert_allclose(np.asarray(jnp.tensordot(J, d, 3)), fd, rtol=1e-5, atol=1e-10)
        self.assertGreater(Fb.sigma_min_normalized(J), 1e-3)

    def test_corrector_custom_vjp(self):
        """d/ds of a nonlinear function of the correctors: custom VJP (adjoint solve) vs finite differences."""
        lh = L.LinearHomogenization((8, 8, 8))
        s0 = lh.stiffness_scale(grey_cell(4))
        Es = jnp.eye(6)[:2]
        u0 = jnp.zeros((2, 8, 8, 8, 3))
        f = lambda s: jnp.sum(lh.correctors(s, Es, u0) ** 2)  # noqa: E731
        g = jax.grad(f)(s0)
        d = jnp.asarray(np.random.default_rng(6).standard_normal(s0.shape))
        h = 1e-6
        fd = (f(s0 + h * d) - f(s0 - h * d)) / (2 * h)
        self.assertAlmostEqual(float(jnp.vdot(g, d)), float(fd), delta=1e-4 * abs(float(fd)))  # FD noise of the CG tolerance

    @unittest.skipUnless(REF, "set NLFIBER_REFERENCE to the companion reference solver directory")
    def test_against_reference_solver(self):
        sys.path.insert(0, REF)
        import oracle as O
        rho = grey_cell(4)
        lh = L.LinearHomogenization(rho.shape)
        C, _, _ = lh.C_full(rho)
        s = np.asarray(lh.stiffness_scale(rho))
        Co, _ = O.C_linear(s - 1e-9, E=1.0, nu=0.3)
        np.testing.assert_allclose(np.asarray(C), Co, atol=1e-9 * np.abs(Co).max())

    def test_hashin_shtrikman_bounds_cell(self):
        rho = binary_cell(6, v=0.3)
        C, _, _ = L.LinearHomogenization(rho.shape).C_full(jnp.asarray(rho, float))
        c11, c12, c44 = L.cubic_invariants(np.asarray(C))
        K = (c11 + 2 * c12) / 3
        Khs, Ghs = L.hashin_shtrikman(rho.mean())
        self.assertLess(K, Khs)
        self.assertLess(c44, Ghs)


class TestTopology(unittest.TestCase):
    def test_periodic_components(self):
        b = np.zeros((6, 6, 6), bool)
        b[:, 2, 2] = True  # a rod along x through the periodic boundary: one component
        lab, n = T.periodic_label(b)
        self.assertEqual(n, 1)
        self.assertFalse(T.spanning(lab, 1))  # does not touch every y slice
        b[2, :, 2] = b[2, 2, :] = True  # cross: spans all axes
        self.assertTrue(T.remove_floating(b).sum() == b.sum())
        b2 = b.copy()
        b2[5, 5, 5] = True  # isolated voxel: floating
        self.assertEqual(T.remove_floating(b2).sum(), b.sum())

    def test_euler_characteristic(self):
        b = np.zeros((6, 6, 6), bool)
        b[1:3, 1:3, 1:3] = True  # a cube: chi = 1
        self.assertEqual(T.euler_characteristic(b), 1)
        c = np.zeros((6, 6, 6), bool)
        c[:, 2, 2] = True  # a closed loop around the torus: chi = 0
        self.assertEqual(T.euler_characteristic(c), 0)

    def test_validity_and_iou(self):
        b = binary_cell(6, v=0.35)
        self.assertTrue(T.validity(b)["valid"])
        self.assertEqual(T.iou(b, b), 1.0)


class TestProjection(unittest.TestCase):
    def test_homotopy_targets(self):
        st = P.homotopy_targets([0.01, 0.003, 0.002], [0.1, -0.01, 0.02], max_ratio=2.0)
        self.assertEqual(len(st), 4)  # log(10) / log(2) -> 4 stages
        np.testing.assert_allclose(st[-1], [0.1, -0.01, 0.02])
        self.assertTrue(all(st[k][0] < st[k + 1][0] for k in range(3)))  # geometric in C11
        self.assertEqual(len(P.homotopy_targets([0.1, 0.01, 0.02], [0.12, 0.01, 0.02])), 1)

    def test_reaches_target_from_random_starts(self):
        fib = Fb.Fiber(6, v=0.3, r=1.5)
        target = fib.value(schwarz_p_phi(6), 4.0)
        for seed in range(2):
            phi0 = Fb.random_start(6, seed=seed, kmax=1)
            phi, info = P.project(fib, phi0, target, betas=(2, 4), tol=1e-5)
            self.assertLess(info["err"], 1e-4, f"seed {seed}: {info['err']}")
            self.assertAlmostEqual(float(fib.rho(phi, 4).mean()), 0.3, places=10)

    def test_metric_changes_the_landing_point(self):
        fib = Fb.Fiber(6, v=0.3, r=1.5)
        target = fib.value(schwarz_p_phi(6), 4.0)
        phi0 = Fb.random_start(6, seed=0, kmax=1)
        p1, i1 = P.project(fib, phi0, target, betas=(4,), tol=1e-6)
        met = M.DCTMetric(M.power_law_spectrum(6, alpha=3.0))
        p2, i2 = P.project(fib, phi0, target, betas=(4,), metric=met, tol=1e-6)
        self.assertLess(i2["err"], 1e-5)
        self.assertGreater(float(jnp.linalg.norm(p1 - p2)), 1e-3)


if __name__ == "__main__":
    unittest.main()
