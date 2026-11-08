"""Flow-matching geometry prior on symmetric octant pre-densities, and its deterministic generator G_K.

Path x_t = t x1 + (1 - t) x0 with x0 a standard normal on the symmetric subspace, velocity target x1 - x0
(rectified flow). Data are standardized by a scalar mean and standard deviation. G_K integrates the learned
velocity from t = 0 to 1 with K Euler (or Heun) steps; it is a fixed deterministic map, so the prior density
is the pushforward of N(0, I) through it (the latent sampler S2 conditions in z).
Optionally conditional on a vector (volume fraction; or stiffness and nonlinear targets for amortization).
"""
import jax
import jax.numpy as jnp
import numpy as np

from .. import symmetry as S
from .unet import apply_unet, init_unet, param_count


def adam_init(p):
    z = jax.tree_util.tree_map(jnp.zeros_like, p)
    return dict(m=z, v=jax.tree_util.tree_map(jnp.zeros_like, p), t=jnp.zeros((), jnp.int32))


def adam_update(p, g, st, lr, b1=0.9, b2=0.999, eps=1e-8, clip=1.0):
    gn = jnp.sqrt(sum(jnp.sum(x**2) for x in jax.tree_util.tree_leaves(g)))
    g = jax.tree_util.tree_map(lambda x: x * jnp.minimum(1.0, clip / (gn + 1e-12)), g)
    t = st["t"] + 1
    tf = t.astype(jnp.float32)
    c1, c2 = 1 - jnp.float32(b1) ** tf, 1 - jnp.float32(b2) ** tf
    m = jax.tree_util.tree_map(lambda a, b: (b1 * a + (1 - b1) * b).astype(a.dtype), st["m"], g)
    v = jax.tree_util.tree_map(lambda a, b: (b2 * a + (1 - b2) * b * b).astype(a.dtype), st["v"], g)
    p = jax.tree_util.tree_map(lambda a, b, c: (a - lr * (b / c1) / (jnp.sqrt(c / c2) + eps)).astype(a.dtype), p, m, v)
    return p, dict(m=m, v=v, t=t), gn


class FlowPrior:
    def __init__(self, n_o, d=3, widths=(16, 32, 64), cond_dim=0, seed=0):
        self.n_o, self.d, self.widths, self.cond_dim = n_o, d, tuple(widths), cond_dim
        self.params = init_unet(jax.random.PRNGKey(seed), d, widths, cond_dim)
        self.ema = self.params
        self.mu, self.sd = 0.0, 1.0
        self.cmu, self.csd = None, None  # condition standardization
        self._loss_grad = jax.jit(jax.value_and_grad(self._loss))
        self._vel = jax.jit(lambda p, x, t, c: apply_unet(p, x, t, c))

    def as_float64(self):
        """Cast the averaged parameters to float64 (derivative checks, exact MCMC)."""
        self.ema = self.params64()
        return self

    def params64(self):
        return jax.tree_util.tree_map(lambda a: a.astype(jnp.float64), self.ema)

    def nparams(self):
        return param_count(self.params)

    # --- normalization ---------------------------------------------------------------------
    def to_model(self, phi):
        return (phi - self.mu) / self.sd

    def to_phi(self, x):
        return x * self.sd + self.mu

    def norm_cond(self, c):
        if c is None or self.cmu is None:
            return c
        return (jnp.asarray(c, jnp.float32) - self.cmu) / self.csd

    # --- training --------------------------------------------------------------------------
    def _loss(self, p, x1, key, cond):
        B = x1.shape[0]
        k1, k2 = jax.random.split(key)
        x0 = S.sym_perm(jax.random.normal(k1, x1.shape, jnp.float32), batch=1)
        t = jax.nn.sigmoid(jax.random.normal(k2, (B,)))  # logit-normal times
        tb = t.reshape((B,) + (1,) * self.d)
        xt = tb * x1 + (1 - tb) * x0
        v = apply_unet(p, xt, t, cond)
        return jnp.mean((v - (x1 - x0)) ** 2)

    def fit(self, phi, steps=1000, batch=16, lr=2e-4, cond=None, ema=0.999, seed=0, log_every=100, verbose=True):
        """phi: (N, n_o^d) data; cond: (N, cond_dim) or None. Returns the loss history."""
        X = np.asarray(phi, np.float32)
        self.mu, self.sd = float(X.mean()), float(X.std() + 1e-8)
        X = (X - self.mu) / self.sd
        X = np.asarray(S.sym_perm(jnp.asarray(X), batch=1))
        if cond is not None:
            Cn = np.asarray(cond, np.float32)
            self.cmu, self.csd = Cn.mean(0), Cn.std(0) + 1e-6
            Cn = (Cn - self.cmu) / self.csd
        st = adam_init(self.params)
        key = jax.random.PRNGKey(seed)
        rng = np.random.default_rng(seed)
        upd = jax.jit(lambda p, g, s: adam_update(p, g, s, lr))
        ema_upd = jax.jit(lambda e, p, k: jax.tree_util.tree_map(lambda a, b: (k * a + (1 - k) * b).astype(a.dtype), e, p))
        hist = []
        p = self.params
        for it in range(steps):
            idx = rng.integers(0, X.shape[0], batch)
            xb = X[idx]
            # half-period shift of the cell along all three axes (keeps the mirror planes and the permutation
            # symmetry): the octant of the shifted cell is the original octant reversed along every axis
            flip = rng.random(batch) < 0.5
            xb[flip] = xb[flip][(slice(None),) + (slice(None, None, -1),) * self.d]
            key, sub = jax.random.split(key)
            c = None if cond is None else jnp.asarray(Cn[idx])
            loss, g = self._loss_grad(p, jnp.asarray(xb), sub, c)
            p, st, gn = upd(p, g, st)
            k = min(ema, (1.0 + it) / (10.0 + it))  # warm-up of the average
            self.ema = ema_upd(self.ema, p, jnp.float32(k)) if it > 0 else p
            hist.append(float(loss))
            if verbose and (it % log_every == 0 or it == steps - 1):
                print(f"  step {it}: loss {np.mean(hist[-log_every:]):.4f}", flush=True)
        self.params = p
        return np.array(hist)

    # --- generator -------------------------------------------------------------------------
    def velocity(self, x, t, cond=None, params=None):
        """Velocity for a batch (B, n^d) or a single field (n^d), t scalar."""
        p = self.ema if params is None else params
        single = x.ndim == self.d
        xb = x[None] if single else x
        tb = jnp.full((xb.shape[0],), t, jnp.float32)
        cb = None
        if cond is not None:
            cb = self.norm_cond(cond)
            cb = jnp.broadcast_to(cb, (xb.shape[0], cb.shape[-1]))
        dt = p["inp"]["w"].dtype
        v = apply_unet(p, xb.astype(dt), tb.astype(dt), cb)
        return v[0] if single else v

    def latent(self, key, n=None):
        shape = (self.n_o,) * self.d if n is None else (n,) + (self.n_o,) * self.d
        return S.sym_perm(jax.random.normal(key, shape, jnp.float32), batch=0 if n is None else 1)

    def generate(self, z, K=16, cond=None, method="euler", params=None):
        """G_K(z): x at t = 1 in model units (apply to_phi for the pre-density). Differentiable in z."""
        x = z
        dt = 1.0 / K
        for k in range(K):
            t = k * dt
            v = self.velocity(x, t, cond, params)
            if method == "heun" and k < K - 1:
                v2 = self.velocity(x + dt * v, t + dt, cond, params)
                v = 0.5 * (v + v2)
            x = x + dt * v
        return x

    def sample_phi(self, key, n, K=16, cond=None, method="euler"):
        z = self.latent(key, n)
        return self.to_phi(jax.jit(lambda z_: self.generate(z_, K, cond, method))(z))

    # --- persistence -------------------------------------------------------------------------
    def save(self, path):
        """Plain-array checkpoint (.npz): configuration, normalization and the parameter leaves."""
        arrs = {f"p{i}": np.asarray(a) for i, a in enumerate(jax.tree_util.tree_leaves(self.params))}
        arrs.update({f"e{i}": np.asarray(a) for i, a in enumerate(jax.tree_util.tree_leaves(self.ema))})
        cfg = np.array([self.n_o, self.d, self.cond_dim] + list(self.widths), np.int64)
        norm = np.array([self.mu, self.sd], np.float64)
        extra = {} if self.cmu is None else dict(cmu=np.asarray(self.cmu), csd=np.asarray(self.csd))
        np.savez(path, cfg=cfg, norm=norm, **extra, **arrs)

    @staticmethod
    def load(path):
        st = np.load(path, allow_pickle=False)
        cfg = [int(x) for x in st["cfg"]]
        fp = FlowPrior(cfg[0], cfg[1], tuple(cfg[3:]), cfg[2])
        tdef = jax.tree_util.tree_structure(fp.params)
        n = len(jax.tree_util.tree_leaves(fp.params))
        fp.params = jax.tree_util.tree_unflatten(tdef, [jnp.asarray(st[f"p{i}"]) for i in range(n)])
        fp.ema = jax.tree_util.tree_unflatten(tdef, [jnp.asarray(st[f"e{i}"]) for i in range(n)])
        fp.mu, fp.sd = float(st["norm"][0]), float(st["norm"][1])
        if "cmu" in st:
            fp.cmu, fp.csd = jnp.asarray(st["cmu"]), jnp.asarray(st["csd"])
        return fp
