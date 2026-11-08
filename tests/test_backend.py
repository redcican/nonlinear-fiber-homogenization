"""The fs3d back end against the JAX evaluator on a 16^3 binary cell (needs NLFIBER_FS3D=<build/fs3d>)."""
import os
import sys
import unittest

import numpy as np
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nlfiber import linear as L, mech as Me, qoi as Q  # noqa: E402
from helpers import binary_cell  # noqa: E402

FS3D = os.environ.get("NLFIBER_FS3D")


@unittest.skipUnless(FS3D, "set NLFIBER_FS3D to the fs3d binary")
class TestFS3D(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from nlfiber.backend_fs3d import FS3D as B
        cls.rho = binary_cell(8, v=0.3).astype(np.float64)  # 16^3
        cls.be = B(FS3D, workdir=os.path.join(os.path.dirname(__file__), "..", "runs", "fs3d_test"))

    def test_moduli(self):
        C, T = self.be.moduli(self.rho)
        Cj, chi, _ = L.LinearHomogenization(self.rho.shape).C_full(jnp.asarray(self.rho))
        Cj = np.asarray(Cj)
        np.testing.assert_allclose(C, Cj, atol=1e-5 * np.abs(Cj).max())
        _, Tj = Me.toec(Me.Cell(self.rho), chi)
        Tj = np.asarray(Tj)
        np.testing.assert_allclose(T, Tj, atol=1e-4 * np.abs(Tj).max())

    def test_compression_path(self):
        rows = self.be.run_path(self.rho, "compression", emax=0.06, de=0.02, crit="1")
        self.assertEqual(len(rows), 3)
        cell = Me.Cell(self.rho)
        rj = Q.run_path(cell, "compression", emax=0.06, de=0.02, kick=0.0, stability=True, k_eigs=1)
        for a, b in zip(rows, rj):
            self.assertAlmostEqual(a["e"], b["e"], places=12)
            np.testing.assert_allclose(a["P"][0, 0], b["P"][0, 0], rtol=1e-4)
            np.testing.assert_allclose(a["nu"], b["nu"], atol=1e-4)
        # the scaled eigenvalues use different diagonal scalings; the sign and the trend must agree
        self.assertEqual(np.sign(rows[-1]["lam"][0]), np.sign(rj[-1]["lam"][0]))

    def test_gradient(self):
        g = self.be.grad(self.rho, obj="P11", path="tension", e=0.04, steps=2)
        cell = Me.Cell(self.rho)
        mix = Me.Mixed(cell, Q.LATERAL[0])
        Fb = np.diag([1.04, 1.0, 1.0])
        w, F, log = mix.solve(Fb, rtol=1e-11, cg_tol=1e-10)
        q = lambda z, Fb_, r: jax_grad_P11(cell, mix, z, Fb_, r)  # noqa: E731
        gj, _ = mix.grad_rho(log["z"], F, q)
        gj = np.asarray(gj)
        m = self.rho > 0.5  # the gradient is compared on the solid voxels (void voxels: 1e-3 stiffness floor)
        cos = np.vdot(g[m], gj[m]) / (np.linalg.norm(g[m]) * np.linalg.norm(gj[m]))
        self.assertGreater(cos, 0.999)
        self.assertLess(np.linalg.norm(g[m] - gj[m]) / np.linalg.norm(gj[m]), 1e-2)


def jax_grad_P11(cell, mix, z, Fb, r):
    import jax
    w, Ff = mix.unpack(z, Fb)
    return jax.grad(cell.energy, argnums=1)(w, Ff, r)[0, 0]


if __name__ == "__main__":
    unittest.main()
