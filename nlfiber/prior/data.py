"""Procedural symmetric cells for the geometry prior (data mix (c) of the plan), and the pre-density phi of a
binary cell.

All generators return a full-cell scalar field f on a (2 n_o)^d grid that is invariant under the mirror planes
at the cell faces and mid-planes and under axis permutations; the cell is {f > threshold} with the threshold
set by the volume fraction. Families:
  tpms      Schwarz P, I-WP and Neovius surfaces, skeletal (f) or sheet (-|f|)
  spinodal  random cosine series on a shell of wave numbers (Gaussian random field level sets)
  truss     strut lattices (SC, BCC, FCC, octet, edge, re-entrant) of random radius
  plate     plate lattices (SC, BCC planes)
  foam      Voronoi wall foams from seeds in the fundamental domain
  random    low-frequency random fields (as the P1 starts)
"""
import itertools

import numpy as np
from scipy import ndimage

from .. import symmetry as S

FAMILIES = ("tpms", "spinodal", "truss", "plate", "foam", "random")


def _coords(n, d):
    x = (np.arange(n) + 0.5) / n  # cell [0, 1)
    return np.meshgrid(*[x] * d, indexing="ij")


def _sym(f):
    import jax.numpy as jnp
    return np.asarray(S.exact_symmetrize(jnp.asarray(f)))


def orbit_max(f):
    """max of f over the 48 (2D: 8) symmetries: union of the transformed copies of a solid {f > t}."""
    d = f.ndim
    out = None
    for perm in itertools.permutations(range(d)):
        g = np.transpose(f, perm)
        for flips in itertools.product((False, True), repeat=d):
            h = g
            for ax, fl in enumerate(flips):
                if fl:
                    h = np.flip(h, ax)
            out = h if out is None else np.maximum(out, h)
    return out


def tpms(n, d, rng):
    X = _coords(n, d)
    c = [np.cos(2 * np.pi * x) for x in X]
    kind = rng.choice(["P", "IWP", "N"]) if d == 3 else "P"
    if kind == "P":
        f = sum(c)
    elif kind == "IWP":
        c2 = [np.cos(4 * np.pi * x) for x in X]
        f = 2 * (c[0] * c[1] + c[1] * c[2] + c[2] * c[0]) - sum(c2)
    else:
        f = 3 * sum(c) + 4 * c[0] * c[1] * c[2]
    f = f / np.abs(f).max()
    return -np.abs(f) if rng.random() < 0.5 else f * rng.choice([-1, 1])


def spinodal(n, d, rng, k0=None):
    k0 = rng.uniform(1.5, 3.5) if k0 is None else k0
    X = _coords(n, d)
    f = np.zeros_like(X[0])
    kmax = int(np.ceil(k0 + 1))
    for k in itertools.product(range(kmax + 1), repeat=d):
        kn = np.linalg.norm(k)
        if kn == 0 or abs(kn - k0) > 1.0:
            continue
        term = rng.standard_normal()
        for ax in range(d):
            term = term * np.cos(2 * np.pi * k[ax] * X[ax])
        f = f + term
    return f


def _segments(kind, d, rng):
    """Strut segments (pairs of points in [0,1]^d) of a symmetric lattice, before symmetrization."""
    if d == 2:
        base = {"SC": [((0, 0), (1, 0))], "BCC": [((0, 0), (0.5, 0.5))], "edge": [((0, 0.5), (0.5, 0))]}
        kind = kind if kind in base else rng.choice(list(base))
        return base[kind]
    a = 0.5
    base = {
        "SC": [((0, 0, 0), (1, 0, 0))],
        "BCC": [((0, 0, 0), (a, a, a))],
        "FCC": [((0, 0, 0), (a, a, 0))],
        "octet": [((0, 0, 0), (a, a, 0)), ((a, a, 0), (a, 0, a))],
        "edge": [((a, 0, 0), (0, a, 0))],
    }
    if kind == "reentrant":
        t = rng.uniform(0.1, 0.3)  # inward offset of the face nodes
        return [((0, 0, 0), (a, a, t)), ((a, a, t), (a, 0, 0))]
    return base[kind]


def truss(n, d, rng):
    kinds = ["SC", "BCC", "FCC", "octet", "edge", "reentrant"] if d == 3 else ["SC", "BCC", "edge"]
    chosen = rng.choice(kinds, size=rng.integers(1, 3), replace=False)
    segs = []
    for k in chosen:
        segs += _segments(k, d, rng)
    X = np.stack(_coords(n, d), axis=-1)
    dist = np.full(X.shape[:-1], np.inf)
    for p, q in segs:
        p, q = np.array(p, float), np.array(q, float)
        for shift in itertools.product((-1, 0, 1), repeat=d):
            a = p + np.array(shift)
            b = q + np.array(shift)
            ab = b - a
            t = np.clip(((X - a) @ ab) / (ab @ ab), 0, 1)
            dist = np.minimum(dist, np.linalg.norm(X - a - t[..., None] * ab, axis=-1))
    return orbit_max(-dist)


def plate(n, d, rng):
    X = _coords(n, d)
    planes = []
    if rng.random() < 0.6 or d == 2:  # SC plates x = 0
        planes.append(np.minimum(X[0], 1 - X[0]))
    if rng.random() < 0.6:  # diagonal plates x = y
        planes.append(np.abs(X[0] - X[1]) / np.sqrt(2))
        planes.append(np.abs(X[0] + X[1] - 1) / np.sqrt(2))
    if not planes:
        planes.append(np.minimum(X[0], 1 - X[0]))
    return orbit_max(-np.minimum.reduce(planes))


def foam(n, d, rng, nseeds=None):
    nseeds = rng.integers(2, 5) if nseeds is None else nseeds
    seeds = rng.uniform(0, 0.5, size=(nseeds, d))
    pts = []
    for s in seeds:  # orbit under mirrors (x -> -x, x -> 1 - x) and permutations
        for perm in itertools.permutations(range(d)):
            for flips in itertools.product((0, 1), repeat=d):
                q = np.array([s[p] for p in perm])
                q = np.where(np.array(flips) == 1, -q, q) % 1.0
                pts.append(q)
    pts = np.unique(np.round(np.array(pts), 12), axis=0)
    X = np.stack(_coords(n, d), axis=-1)
    dd = []
    for q in pts:
        diff = np.abs(X - q)
        diff = np.minimum(diff, 1 - diff)
        dd.append(np.linalg.norm(diff, axis=-1))
    dd = np.sort(np.stack(dd), axis=0)
    return -(dd[1] - dd[0])  # walls: where the two nearest seeds are equidistant


def random_field(n, d, rng):
    X = _coords(n, d)
    f = np.zeros_like(X[0])
    for _ in range(30):
        k = rng.integers(0, 3, size=d)
        if not k.any():
            continue
        t = rng.standard_normal() / np.linalg.norm(k)
        for ax in range(d):
            t = t * np.cos(2 * np.pi * k[ax] * X[ax])
        f = f + t
    return f


GEN = dict(tpms=tpms, spinodal=spinodal, truss=truss, plate=plate, foam=foam, random=random_field)


def binary_from_field(f, v):
    """Threshold a symmetric field to solid fraction v; keep the spanning solid (None if nothing spans)."""
    from ..topology import remove_floating
    b = f > np.quantile(f, 1 - v)
    b = remove_floating(b)
    return b if b.any() else None


def phi_from_binary(b, clip=4.0):
    """Clipped periodic signed distance (positive in the solid), divided by clip: values in [-1, 1]."""
    b = np.asarray(b, bool)
    pad = int(np.ceil(clip)) + 1
    p = np.pad(b, pad, mode="wrap")
    din = ndimage.distance_transform_edt(p)
    dout = ndimage.distance_transform_edt(~p)
    sd = (din - 0.5) * p - (dout - 0.5) * (~p)
    sd = sd[tuple(slice(pad, -pad) for _ in range(b.ndim))]
    return np.clip(sd, -clip, clip) / clip


def make_dataset(n_o, count, d=3, v_range=(0.1, 0.4), families=FAMILIES, seed=0):
    """count cells: octant pre-densities phi (count, n_o^d) float32, family index and solid fraction."""
    rng = np.random.default_rng(seed)
    n = 2 * n_o
    out, fam, vol = [], [], []
    tries = 0
    while len(out) < count:
        k = tries % len(families)  # round robin over the families
        tries += 1
        f = _sym(GEN[families[k]](n, d, rng))
        v = rng.uniform(*v_range)
        b = binary_from_field(f, v)
        if b is None:
            continue
        phi = _sym(phi_from_binary(b))
        out.append(phi[(slice(0, n_o),) * d].astype(np.float32))
        fam.append(FAMILIES.index(families[k]))
        vol.append(b.mean())
    return np.stack(out), np.array(fam), np.array(vol)
