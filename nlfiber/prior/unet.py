"""A small U-Net in plain JAX for fields on the symmetric octant (3D) or quadrant (2D).

Convolutions use 'symmetric' padding: beyond an octant face the field continues as its mirror image, which is
what the mirror planes of the cell prescribe. Time t and an optional condition vector enter through a shared
embedding added in every residual block. The output is projected onto the permutation-symmetric subspace,
so the velocity of the flow keeps the state in it (Oh completion of the plan).
Arrays are channels-last: x (B, n, n, n) -> (B, n, n, n).
"""
import math

import jax
import jax.numpy as jnp
import numpy as np

from .. import symmetry as S


def _dn(d):
    return ("NDHWC", "DHWIO", "NDHWC") if d == 3 else ("NHWC", "HWIO", "NHWC")


def conv(p, x, d):
    pad = [(0, 0)] + [(1, 1)] * d + [(0, 0)]
    if p["w"].shape[0] == 3:
        x = jnp.pad(x, pad, mode="symmetric")
    y = jax.lax.conv_general_dilated(x, p["w"], (1,) * d, "VALID", dimension_numbers=_dn(d))
    return y + p["b"]


def group_norm(p, x, groups):
    B, C = x.shape[0], x.shape[-1]
    g = min(groups, C)
    xs = x.reshape(B, -1, g, C // g)
    mu = xs.mean(axis=(1, 3), keepdims=True)
    var = xs.var(axis=(1, 3), keepdims=True)
    xs = (xs - mu) / jnp.sqrt(var + 1e-5)
    return xs.reshape(x.shape) * p["g"] + p["b"]


def dense(p, x):
    return x @ p["w"] + p["b"]


def _init_conv(key, cin, cout, d, k=3, zero=False):
    shape = (k,) * d + (cin, cout)
    fan = cin * k**d
    w = jnp.zeros(shape) if zero else jax.random.normal(key, shape) * math.sqrt(2.0 / fan)
    return dict(w=w.astype(jnp.float32), b=jnp.zeros(cout, jnp.float32))


def _init_dense(key, cin, cout, zero=False):
    w = jnp.zeros((cin, cout)) if zero else jax.random.normal(key, (cin, cout)) * math.sqrt(1.0 / cin)
    return dict(w=w.astype(jnp.float32), b=jnp.zeros(cout, jnp.float32))


def _init_gn(c):
    return dict(g=jnp.ones(c, jnp.float32), b=jnp.zeros(c, jnp.float32))


def _init_res(key, cin, cout, emb, d):
    k = jax.random.split(key, 4)
    p = dict(n1=_init_gn(cin), c1=_init_conv(k[0], cin, cout, d), e=_init_dense(k[1], emb, cout),
             n2=_init_gn(cout), c2=_init_conv(k[2], cout, cout, d, zero=True))
    if cin != cout:
        p["skip"] = _init_conv(k[3], cin, cout, d, k=1)
    return p


def res_block(p, x, e, d, groups):
    h = conv(p["c1"], jax.nn.silu(group_norm(p["n1"], x, groups)), d)
    h = h + dense(p["e"], e)[:, None, None, None, :] if d == 3 else h + dense(p["e"], e)[:, None, None, :]
    h = conv(p["c2"], jax.nn.silu(group_norm(p["n2"], h, groups)), d)
    return h + (conv(p["skip"], x, d) if "skip" in p else x)


def time_embedding(t, dim=64):
    half = dim // 2
    f = jnp.exp(-math.log(1000.0) * jnp.arange(half) / half)
    a = 1000.0 * t[:, None] * f[None]
    return jnp.concatenate([jnp.sin(a), jnp.cos(a)], axis=-1)


def init_unet(key, d=3, widths=(16, 32, 64), cond_dim=0, emb=64):
    keys = iter(jax.random.split(key, 64))
    p = dict(t1=_init_dense(next(keys), emb, emb), t2=_init_dense(next(keys), emb, emb),
             inp=_init_conv(next(keys), 1, widths[0], d), down=[], up=[])
    if cond_dim:
        p["cond"] = _init_dense(next(keys), cond_dim, emb)
    c = widths[0]
    for w in widths:
        p["down"].append(_init_res(next(keys), c, w, emb, d))
        c = w
    p["mid"] = _init_res(next(keys), c, c, emb, d)
    for w in reversed(widths):
        p["up"].append(_init_res(next(keys), c + w, w, emb, d))
        c = w
    p["out_n"] = _init_gn(c)
    p["out"] = _init_conv(next(keys), c, 1, d, zero=True)
    return p


def _pool(x, d):
    s = x.shape
    if d == 3:
        return x.reshape(s[0], s[1] // 2, 2, s[2] // 2, 2, s[3] // 2, 2, s[4]).mean(axis=(2, 4, 6))
    return x.reshape(s[0], s[1] // 2, 2, s[2] // 2, 2, s[3]).mean(axis=(2, 4))


def _up(x, d):
    for ax in range(1, d + 1):
        x = jnp.repeat(x, 2, axis=ax)
    return x


def apply_unet(p, x, t, cond=None, groups=8):
    """x (B, n^d) float32, t (B,) -> velocity (B, n^d), permutation-symmetric."""
    d = x.ndim - 1
    dt = p["inp"]["w"].dtype  # float32 in training; float64 when the parameters are cast for checks
    x, t = x.astype(dt), t.astype(dt)
    e = time_embedding(t).astype(dt)
    e = dense(p["t2"], jax.nn.silu(dense(p["t1"], e)))
    if cond is not None and "cond" in p:
        e = e + dense(p["cond"], cond.astype(dt))
    e = jax.nn.silu(e)
    h = conv(p["inp"], x[..., None], d)
    skips = []
    for i, blk in enumerate(p["down"]):
        h = res_block(blk, h, e, d, groups)
        skips.append(h)
        if i < len(p["down"]) - 1:
            h = _pool(h, d)
    h = res_block(p["mid"], h, e, d, groups)
    for i, blk in enumerate(p["up"]):
        s = skips.pop()
        if h.shape[1] != s.shape[1]:
            h = _up(h, d)
        h = res_block(blk, jnp.concatenate([h, s], axis=-1), e, d, groups)
    out = conv(p["out"], jax.nn.silu(group_norm(p["out_n"], h, groups)), d)[..., 0]
    return S.sym_perm(out, batch=1)


def param_count(p):
    return int(sum(np.prod(a.shape) for a in jax.tree_util.tree_leaves(p)))
