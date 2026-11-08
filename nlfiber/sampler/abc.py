"""Rejection ABC arbiter (T0 and the approximate 3D arbiter): prior samples whose stiffness lies within a
relative tolerance eps of the target. As eps -> 0 the accepted set follows the co-area target pi*."""
import numpy as np


def abc_accept(c_samples, target, eps_list=(0.04, 0.02, 0.01)):
    """c_samples (N, 3) stiffness invariants of prior samples. Returns {eps: indices of accepted samples}."""
    c = np.asarray(c_samples, float)
    t = np.asarray(target, float)
    err = np.linalg.norm(c - t[None], axis=1) / np.linalg.norm(t)
    return {e: np.where(err < e)[0] for e in eps_list}


def extrapolate(stat_by_eps):
    """Linear extrapolation of a statistic to eps = 0 from {eps: value} (least squares in eps)."""
    e = np.array(sorted(stat_by_eps))
    v = np.array([stat_by_eps[x] for x in e])
    ok = np.isfinite(v)
    if ok.sum() < 2:
        return float(v[ok][0]) if ok.any() else float("nan")
    A = np.stack([np.ones(ok.sum()), e[ok]], 1)
    coef, *_ = np.linalg.lstsq(A, v[ok], rcond=None)
    return float(coef[0])
