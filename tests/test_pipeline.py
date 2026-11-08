"""Prior, S1/S2 samplers, steering, baselines, amortization and the runner on small cells (2D quadrant 8 ->
16^2 cell for speed where the geometry does not matter, 3D 4^3 octant elsewhere)."""
import json
import os
import sys
import tempfile
import unittest

import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nlfiber import fiber as Fb, project as P, symmetry as S, topology as T, runner, amortize  # noqa: E402
from nlfiber.prior import data as Dd, flow as Fl  # noqa: E402
from nlfiber.sampler import s1, zhg, latent  # noqa: E402
from nlfiber.steer import nullspace, smc, losses  # noqa: E402
from nlfiber import baselines as B  # noqa: E402
from helpers import schwarz_p_phi  # noqa: E402


def small_prior(n_o=4, d=3, count=48, steps=150, seed=0):
    X, fam, vol = Dd.make_dataset(n_o, count, d=d, seed=seed)
    fp = Fl.FlowPrior(n_o, d, widths=(8, 16), seed=seed)
    fp.fit(X, steps=steps, batch=16, lr=1e-3, verbose=False)
    return fp, X


class TestPrior(unittest.TestCase):
    def test_data_and_training(self):
        fp, X = small_prior()
        self.assertEqual(X.shape[1:], (4, 4, 4))
        phi = fp.sample_phi(jax.random.PRNGKey(0), 4, K=8)
        self.assertLess(float(jnp.abs(phi - S.sym_perm(phi, batch=1)).max()), 1e-5)
        self.assertTrue(np.all(np.isfinite(np.asarray(phi))))
        with tempfile.TemporaryDirectory() as d:
            fp.save(os.path.join(d, "p.npz"))
            fp2 = Fl.FlowPrior.load(os.path.join(d, "p.npz"))
            phi2 = fp2.sample_phi(jax.random.PRNGKey(0), 4, K=8)
            self.assertLess(float(jnp.abs(phi - phi2).max()), 1e-4)

    def test_generator_is_differentiable(self):
        fp, _ = small_prior(steps=20)
        fp.as_float64()
        z = fp.latent(jax.random.PRNGKey(1)).astype(jnp.float64)
        f = lambda zz: jnp.sum(fp.generate(zz, K=4) ** 2)  # noqa: E731
        g = jax.grad(f)(z)
        d = fp.latent(jax.random.PRNGKey(2)).astype(jnp.float64)
        h = 1e-5
        fd = (f(z + h * d) - f(z - h * d)) / (2 * h)
        self.assertAlmostEqual(float(jnp.vdot(g, d)), float(fd), delta=1e-6 * abs(float(fd)) + 1e-8)


class TestSamplers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fib = Fb.Fiber(4, v=0.3, r=1.0)
        cls.target = cls.fib.value(schwarz_p_phi(4), 4.0)
        cls.fp, cls.X = small_prior(steps=100)
        cls.spec = __import__("nlfiber.metric", fromlist=["spectrum_from_data"]).spectrum_from_data(cls.fp.to_model(jnp.asarray(cls.X)))

    def test_s1_reaches_target(self):
        for metric in ("gaussian", "euclid"):
            z = self.fp.latent(jax.random.PRNGKey(3))
            phi, info = s1.s1_sample(self.fp, self.fib, self.target, z, K=8, t0=0.5, metric=metric,
                                     spectrum=self.spec, betas=(2, 4), up=1, polish_steps=2)
            self.assertLess(info["err_grey_before_polish"], 1e-3, metric)
            self.assertAlmostEqual(float(self.fib.rho(phi, 4).mean()), 0.3, places=10)

    def test_s2_latent_chain_stays_on_fiber(self):
        lat = latent.LatentFiber(self.fp, self.fib, self.target, beta=4.0, K=4)
        z0, info = lat.project(self.fp.latent(jax.random.PRNGKey(4)), tol=1e-7)
        self.assertTrue(info["converged"], info)
        sampler = zhg.ZHG(lat.cj, sym=lambda u: S.sym_perm(u), newton_tol=1e-8)
        chain, s = sampler.run(z0, 12, 0.1, seed=0)
        h, _ = lat.cj(chain[-1])
        self.assertLess(np.abs(h).max(), 1e-7)
        self.assertGreater(sampler.stats["accepted"], 0, sampler.stats)
        self.assertLess(sampler.stats["rev_fail"], sampler.stats["steps"])


class TestSteering(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fib = Fb.Fiber(4, v=0.3, r=1.0)
        cls.target = cls.fib.value(schwarz_p_phi(4), 4.0)
        cls.phi0, _ = P.project(cls.fib, Fb.random_start(4, seed=1, kmax=1), cls.target, betas=(2, 4))

    def test_null_space_flow_lowers_a_linear_loss(self):
        lg = B.multistart.random_linear_loss(self.fib.dm.shape, 0)
        L0 = lg(self.fib.rho(self.phi0, 4.0))[0]
        phi, hist = nullspace.null_space_flow(self.fib, self.phi0, self.target, lg, beta=4.0, iters=8, step=0.2)
        self.assertLess(hist[-1]["L"], L0)
        self.assertLess(hist[-1]["res"], 1e-4)

    def test_poisson_loss_gradient(self):
        rho = self.fib.rho(self.phi0, 4.0)
        loss = losses.PoissonPath([0.02], nu_target=0.0)
        L, g = loss(rho)
        d = jnp.asarray(np.random.default_rng(0).standard_normal(rho.shape)) * rho * (1 - rho)
        h = 1e-4
        fd = (loss(rho + h * d)[0] - loss(rho - h * d)[0]) / (2 * h)
        self.assertAlmostEqual(float(jnp.vdot(g, d)), fd, delta=2e-3 * abs(fd) + 1e-8)

    def test_toec_loss_gradient(self):
        rho = self.fib.rho(self.phi0, 4.0)
        loss = losses.ToecTarget({"C111": -1.0, "C456": 0.0})
        L, g = loss(rho)
        d = jnp.asarray(np.random.default_rng(1).standard_normal(rho.shape)) * rho * (1 - rho)
        h = 1e-5
        fd = (loss(rho + h * d)[0] - loss(rho - h * d)[0]) / (2 * h)
        self.assertAlmostEqual(float(jnp.vdot(g, d)), fd, delta=1e-3 * abs(fd) + 1e-8)

    def test_tilted_smc_lowers_loss(self):
        fp, _ = small_prior(steps=60)
        lat = latent.LatentFiber(fp, self.fib, self.target, beta=4.0, K=4)
        parts = []
        for i in range(4):
            z, info = lat.project(fp.latent(jax.random.PRNGKey(10 + i)), tol=1e-7)
            if info["converged"]:
                parts.append(z)
        self.assertGreaterEqual(len(parts), 2)
        a = jnp.asarray(np.random.default_rng(0).standard_normal(self.fib.dm.shape))
        loss = lambda z: float(jnp.vdot(a, self.fib.rho(lat.phi_of(z), 4.0)))  # noqa: E731
        L0 = np.mean([loss(z) for z in parts])
        out = smc.tilted_smc(lat, parts, loss, beta_max=50.0, moves=2, step=0.2)
        self.assertLessEqual(out["L"].mean(), L0 + 1e-9)
        self.assertEqual(out["betas"][-1], 50.0)


class TestBaselines(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fib = Fb.Fiber(4, v=0.3, r=1.0)
        cls.target = cls.fib.value(schwarz_p_phi(4), 4.0)

    def test_b1_b2(self):
        starts = [Fb.random_start(4, seed=s, kmax=1) for s in range(2)]
        r1 = B.b1_multistart(self.fib, self.target, starts, beta=4.0, iters=3, up=1)
        self.assertEqual(len(r1), 2)
        lg = B.multistart.random_linear_loss(self.fib.dm.shape, 5)
        r2 = B.b2_deflation(self.fib, self.target, starts, lg, beta=4.0, iters=3, up=1)
        self.assertEqual(len(r2), 2)

    def test_b4_b6(self):
        fp, X = small_prior(steps=60)
        phi, info = B.b4_dps(fp, self.fib, self.target, fp.latent(jax.random.PRNGKey(0)), K=6, beta=4.0, up=1)
        self.assertLess(info["err_grey"], 1e-3)
        phis, info = B.b6_values_only(fp, self.fib, self.target, jax.random.PRNGKey(1), n_particles=3, K=4,
                                      beta=4.0, temps=(10.0,), up=1)
        self.assertEqual(len(phis), 3)

    def test_b7_b8(self):
        best = B.b7_parametric(self.fib, self.target, n_eval=12, up=1)
        self.assertTrue(np.isfinite(best["err_bin"]))
        acc, tried = B.b8_random(self.fib, self.target, 6, eps=2.0)
        self.assertEqual(tried, 6)


class TestAmortizeAndRunner(unittest.TestCase):
    def test_descriptor_and_gap(self):
        d = amortize.descriptor([0.1, 0.02, 0.03], 0.3, dict(nu10=0.2, nu20=np.nan))
        self.assertEqual(d.shape, (3 + 1 + 8,))
        rows = [dict(binary=dict(cell=np.eye(4) > 0), solves=10, ok=True),
                dict(binary=dict(cell=np.eye(4) < 1), solves=20, ok=False)]
        g = amortize.gap(rows, rows, lambda r: r["ok"])
        self.assertAlmostEqual(g["success_ratio"], 1.0)

    def test_runner_queue(self):
        with tempfile.TemporaryDirectory() as q:
            for i in range(4):
                (lambda p: p.write_text(json.dumps(dict(module="math", function="sqrt", args={}))))(
                    __import__("pathlib").Path(q) / f"t{i}.json")
            # math.sqrt(dict) fails: both outcomes are recorded
            n0 = runner.work(q, shard=0, n_shards=2, once=True)
            n1 = runner.work(q, shard=1, n_shards=2, once=True)
            files = sorted(os.listdir(q))
            self.assertEqual(sum(f.endswith(".failed.json") for f in files), 4)
            self.assertEqual(n0 + n1, 0)
            self.assertFalse(any(f.endswith(".lock") for f in files))


if __name__ == "__main__":
    unittest.main()
