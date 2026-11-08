"""Periodic connectivity, validity, topological signature and diversity measures of binary cells."""
import itertools

import numpy as np
from scipy import ndimage


def periodic_label(b, connectivity=None):
    """Connected components of a periodic binary array (26-connectivity in 3D, 8 in 2D by default).
    Returns (labels, n) with labels 1..n, 0 for background."""
    b = np.asarray(b, dtype=bool)
    d = b.ndim
    st = np.ones((3,) * d, dtype=bool) if connectivity is None else connectivity
    p = np.pad(b, 1, mode="wrap")
    lab, n = ndimage.label(p, structure=st)
    parent = np.arange(n + 1)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    inner = tuple(slice(1, -1) for _ in range(d))
    core = lab[inner]
    # every padded voxel is a copy of an interior voxel: join their labels
    for ax in range(d):
        for side, src in ((0, -2), (-1, 1)):
            sl_p = [slice(None)] * d
            sl_p[ax] = side
            sl_c = [slice(None)] * d
            sl_c[ax] = src
            a = lab[tuple(sl_p)].ravel()
            c = lab[tuple(sl_c)].ravel()
            for x, y in zip(a[(a > 0) & (c > 0)], c[(a > 0) & (c > 0)]):
                rx, ry = find(x), find(y)
                if rx != ry:
                    parent[max(rx, ry)] = min(rx, ry)
    roots = np.array([find(i) for i in range(n + 1)])
    core = roots[core]
    uniq = np.unique(core[core > 0])
    remap = np.zeros(n + 1, dtype=np.int64)
    remap[uniq] = np.arange(1, len(uniq) + 1)
    return remap[core], len(uniq)


def spanning(labels, k):
    """True if component k touches every slice along every axis (it can carry load across the cell)."""
    m = labels == k
    for ax in range(m.ndim):
        other = tuple(a for a in range(m.ndim) if a != ax)
        if not np.all(m.any(axis=other)):
            return False
    return True


def remove_floating(b):
    """Keep only the spanning solid components."""
    lab, n = periodic_label(b)
    keep = np.zeros(n + 1, dtype=bool)
    for k in range(1, n + 1):
        keep[k] = spanning(lab, k)
    return keep[lab]


def euler_characteristic(b):
    """Euler characteristic of the union of closed voxels on the torus (cubical complex)."""
    b = np.asarray(b, dtype=bool)
    d = b.ndim
    chi = 0
    # a cell of dimension q is spanned by q axes; it belongs to the complex if any adjacent voxel is solid.
    # cells are indexed by their lower corner; adjacent voxels differ by -1 in the axes not spanned.
    for q in range(d + 1):
        for axes in itertools.combinations(range(d), q):
            fixed = [a for a in range(d) if a not in axes]
            occ = np.zeros_like(b)
            for shifts in itertools.product((0, 1), repeat=len(fixed)):
                s = [0] * d
                for a, t in zip(fixed, shifts):
                    s[a] = t
                occ |= np.roll(b, shift=s, axis=tuple(range(d)))
            chi += (-1) ** q * int(occ.sum())
    return chi


def signature(b):
    """Topological signature (solid components, void components, Euler characteristic)."""
    b = np.asarray(b, dtype=bool)
    d = b.ndim
    _, ns = periodic_label(b)
    face = ndimage.generate_binary_structure(d, 1)  # dual connectivity for the void
    _, nv = periodic_label(~b, connectivity=face)
    return ns, nv, euler_characteristic(b)


def opening_change(b, radius=1):
    """Fraction of solid voxels removed by a periodic morphological opening with a ball of `radius`."""
    b = np.asarray(b, dtype=bool)
    d = b.ndim
    r = int(radius)
    g = np.mgrid[tuple(slice(-r, r + 1) for _ in range(d))]
    ball = sum(x**2 for x in g) <= radius**2 + 1e-9
    p = np.pad(b, r + 1, mode="wrap")
    o = ndimage.binary_opening(p, structure=ball)
    o = o[tuple(slice(r + 1, -(r + 1)) for _ in range(d))]
    return float((b & ~o).sum() / max(b.sum(), 1))


def validity(b, feature_radius=1, max_opening_change=0.01):
    """Validity checks of the plan (Sec. 5.4): one spanning periodic solid component, no floating parts,
    solid on every face pair, void connected, minimum feature size."""
    b = np.asarray(b, dtype=bool)
    lab, n = periodic_label(b)
    span = [spanning(lab, k) for k in range(1, n + 1)]
    face = ndimage.generate_binary_structure(b.ndim, 1)
    _, nv = periodic_label(~b, connectivity=face)
    faces = all(np.take(b, 0, axis=ax).any() for ax in range(b.ndim))
    oc = opening_change(b, feature_radius)
    out = dict(components=n, spanning=int(sum(span)), floating=int(n - sum(span)), void_components=nv,
               solid_on_faces=bool(faces), opening_change=oc)
    out["valid"] = bool(n == 1 and sum(span) == 1 and faces and oc < max_opening_change)
    return out


def iou(a, b):
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0


def pairwise_iou(cells):
    m = len(cells)
    out = np.full((m, m), np.nan)
    for i in range(m):
        for j in range(i + 1, m):
            out[i, j] = out[j, i] = iou(cells[i], cells[j])
    return out


def mean_pairwise_iou(cells):
    P = pairwise_iou(cells)
    return float(np.nanmean(P[np.triu_indices(len(cells), 1)])) if len(cells) > 1 else float("nan")
