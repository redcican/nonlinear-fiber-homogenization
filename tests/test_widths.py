"""Width statistics and hypothesis decisions on synthetic numbers."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nlfiber import widths as W  # noqa: E402


class TestWidths(unittest.TestCase):
    def test_spread_and_ratio(self):
        x = np.linspace(0, 1, 101)
        self.assertAlmostEqual(W.spread(x), 0.9)
        self.assertAlmostEqual(W.spread([0.1, 0.4, 0.2]), 0.3)  # max - min below 10 samples
        self.assertAlmostEqual(W.ratio(np.linspace(1, 3, 201)), 2.8 / 1.2)

    def test_floors(self):
        self.assertAlmostEqual(W.path_floor([1, 2, 3], [[1.1, 2, 3], [1, 1.95, 3]]), 0.1)
        # a common shift does not count, a relative change does
        self.assertAlmostEqual(W.difference_floor([1, 2, 3], [1.5, 2.5, 3.5]), 0.0)
        self.assertAlmostEqual(W.difference_floor([1, 2, 3], [1, 2.2, 3]), 0.2)
        self.assertAlmostEqual(W.log_floor([0.1], [[0.11]]), np.log(1.1))

    def test_h1_decision(self):
        rng = np.random.default_rng(0)
        nu = rng.uniform(-0.2, 0.1, 40)
        eps = np.where(rng.random(40) < 0.5, rng.uniform(0.05, 0.2, 40), np.nan)
        r = rng.uniform(0.8, 0.9, 40)
        d = W.h1_target(nu, eps, r, floor_nu=0.01, floor_lneps=0.05, floor_r=0.01)
        self.assertTrue(d["evaluable"] and d["nu_pass"] and d["H1"])
        self.assertTrue(d["r_pass"] and d["H1s"])
        d2 = W.h1_target(nu * 0.1, eps, r, floor_nu=0.01, floor_lneps=1.0, floor_r=0.05)
        self.assertFalse(d2["nu_pass"])
        self.assertFalse(d2["eps_pass"])  # ratio large but below 3 log floors
        self.assertFalse(d2["H1s"])
        q = W.quantifier([d, d2, d, d2], "H1")
        self.assertTrue(q["passed"] and q["k"] == 2)

    def test_distances(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((200, 2))
        Y = rng.standard_normal((200, 2))
        Z = Y + np.array([1.0, 0.0])
        self.assertGreater(W.energy_test(X, Y, n_perm=200)[1], 0.01)
        self.assertLess(W.energy_test(X, Z, n_perm=200)[1], 0.01)
        w = W.sliced_w1(Y, Z, n_proj=2000)
        self.assertAlmostEqual(w, 2 / np.pi, delta=0.05)  # E|theta_1| for a unit shift in 2D
        cov = W.quantile_coverage(rng.standard_normal(5000), rng.standard_normal(5000))
        self.assertAlmostEqual(cov[0.5], 0.5, delta=0.03)
        tau, ci = W.kendall_ci(np.arange(30), np.arange(30) + rng.standard_normal(30), B=300)
        self.assertGreater(ci[0], 0)


if __name__ == "__main__":
    unittest.main()
