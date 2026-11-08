"""Constrained-manifold MCMC and the ABC arbiter on a curve with a known co-area density; S1 and S2 smoke runs."""
import os
import sys
import unittest

import numpy as np
import jax
import jax.numpy as jnp
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nlfiber.sampler.zhg import ZHG  # noqa: E402
from nlfiber.sampler import abc  # noqa: E402

# curve z0 + z1^3 = 1 in R^2, parametrized by u = z1. The Gram factor |grad h| = sqrt(1 + 9 u^4) varies
# strongly along it, so the co-area and Hausdorff densities are far apart.


def cj(z):
    z = jnp.asarray(z)
    h = jnp.array([z[0] + z[1] ** 3 - 1.0])
    J = jnp.array([[1.0, 3 * z[1] ** 2]])
    return h, J


def exact_cdf(gram=True):
    u = np.linspace(-4, 4, 400001)
    z0 = 1.0 - u**3
    ds = np.sqrt(1 + 9 * u**4)
    dens = np.exp(-0.5 * (z0**2 + u**2)) * ds
    if gram:
        dens /= ds  # the co-area target on this curve is the Gaussian restricted to it, in u
    cdf = np.cumsum(dens)
    cdf /= cdf[-1]
    return lambda x: np.interp(x, u, cdf)


CPU = jax.devices("cpu")[0]  # tiny arrays: CPU dispatch is far cheaper than GPU synchronization


class TestCoarea(unittest.TestCase):
    def test_zhg_samples_the_coarea_density(self):
        with jax.default_device(CPU):
            sampler = ZHG(cj, newton_tol=1e-12)
            chain, s = sampler.run(jnp.array([0.0, 1.0]), 6000, 0.8, seed=1, adapt=1000)
        u = np.array([float(z[1]) for z in chain[1500::3]])
        p = stats.kstest(u, exact_cdf(True)).pvalue
        self.assertGreater(p, 1e-3, f"KS p {p}, acceptance {sampler.stats}")
        self.assertLess(sampler.stats["rev_fail"], 0.2 * sampler.stats["steps"])

    def test_without_gram_factor_is_biased(self):
        with jax.default_device(CPU):
            sampler = ZHG(cj, newton_tol=1e-12, gram=False)
            chain, _ = sampler.run(jnp.array([0.0, 1.0]), 6000, 0.8, seed=2, adapt=1000)
        u = np.array([float(z[1]) for z in chain[1500::3]])
        self.assertGreater(stats.kstest(u, exact_cdf(False)).pvalue, 1e-3)  # it samples the Hausdorff density
        self.assertLess(stats.kstest(u, exact_cdf(True)).pvalue, 1e-3)

    def test_abc_converges_to_coarea_density(self):
        rng = np.random.default_rng(0)
        Z = rng.standard_normal((4_000_000, 2))
        c = Z[:, 0] + Z[:, 1] ** 3
        acc = abc.abc_accept(c[:, None], [1.0], eps_list=(0.005,))
        u = Z[acc[0.005], 1]
        self.assertGreater(stats.kstest(u, exact_cdf(True)).pvalue, 1e-3)
        self.assertAlmostEqual(abc.extrapolate({0.01: 1.1, 0.02: 1.2, 0.04: 1.4}), 1.0)


if __name__ == "__main__":
    unittest.main()
