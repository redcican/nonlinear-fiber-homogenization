"""Fiber-width statistics, noise floors, hypothesis decisions and distribution distances.

Definitions follow the plan (Sec. 5.4 and the G1 amendment of 6 Nov 2026):
  spread      q_0.95 - q_0.05 (linear interpolation of the empirical quantiles); max - min below 10 samples
  ratio       q_0.9 / q_0.1 (instability onset)
  floor       per target: the largest of (a) path re-runs (kick, half step), (b) the change of pairwise
              differences under a resolution change, (c) re-projection to a tighter C^H slack
  H1          nu(20%) spread >= 0.1 and >= 3 floor, or eps_cr ratio >= 1.5 with ln(ratio) >= 3 log-floor
  H1s         r(20%) spread >= 0.05 and >= 3 floor
  quantifier  a hypothesis passes if it passes on >= 50% of the evaluable targets
"""
import itertools

import numpy as np
from scipy import stats


def spread(x, lo=0.05, hi=0.95, min_quantile_n=10):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    if x.size < min_quantile_n:
        return float(x.max() - x.min())
    return float(np.quantile(x, hi) - np.quantile(x, lo))


def ratio(x, lo=0.1, hi=0.9):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    return float(np.quantile(x, hi) / np.quantile(x, lo))


def bootstrap_ci(x, stat=spread, B=2000, alpha=0.05, seed=0):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    rng = np.random.default_rng(seed)
    vals = np.array([stat(x[rng.integers(0, x.size, x.size)]) for _ in range(B)])
    vals = vals[np.isfinite(vals)]
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


# --- noise floor --------------------------------------------------------------------------------
def path_floor(base, reruns):
    """(a): largest |q_rerun - q_base| over the re-runs of the same designs. base: (n,), reruns: (k, n)."""
    base, reruns = np.asarray(base, float), np.atleast_2d(np.asarray(reruns, float))
    return float(np.nanmax(np.abs(reruns - base[None])))


def difference_floor(coarse, fine):
    """(b): largest change of a pairwise difference, max_ij |(q_i - q_j)_fine - (q_i - q_j)_coarse|.
    A shift shared by all designs does not count."""
    c, f = np.asarray(coarse, float), np.asarray(fine, float)
    out = 0.0
    for i, j in itertools.combinations(range(c.size), 2):
        v = abs((f[i] - f[j]) - (c[i] - c[j]))
        if np.isfinite(v):
            out = max(out, v)
    return float(out)


def target_floor(*components):
    vals = [v for v in components if v is not None and np.isfinite(v)]
    return float(max(vals)) if vals else float("nan")


def log_floor(base, reruns):
    """eps_cr floor on the log scale: largest |ln eps_rerun - ln eps_base|."""
    base, reruns = np.asarray(base, float), np.atleast_2d(np.asarray(reruns, float))
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.abs(np.log(np.abs(reruns)) - np.log(np.abs(base))[None])
    return float(np.nanmax(d)) if np.isfinite(d).any() else float("nan")


# --- hypothesis decisions -------------------------------------------------------------------------
def h1_target(nu20, eps_max, r20, floor_nu, floor_lneps, floor_r, n_min=10, n_eps_min=5):
    """Decision of H1 and H1s at one target from its accepted samples."""
    nu20, eps_max, r20 = (np.asarray(a, float) for a in (nu20, eps_max, r20))
    out = dict(n=int(np.isfinite(nu20).sum()))
    out["evaluable"] = out["n"] >= n_min
    out["nu_spread"] = spread(nu20)
    out["nu_ci"] = bootstrap_ci(nu20) if out["n"] >= 3 else (np.nan, np.nan)
    out["nu_pass"] = bool(out["nu_spread"] >= 0.1 and out["nu_spread"] >= 3 * floor_nu)
    has = np.isfinite(eps_max)
    out["eps_evaluable"] = int(has.sum()) >= n_eps_min
    out["eps_ratio"] = ratio(np.abs(eps_max)) if out["eps_evaluable"] else float("nan")
    out["eps_pass"] = bool(out["eps_evaluable"] and out["eps_ratio"] >= 1.5
                           and np.log(out["eps_ratio"]) >= 3 * floor_lneps)
    out["eps_ratio_censored"] = ratio(np.where(has, np.abs(eps_max), 0.25))
    out["r_spread"] = spread(r20)
    out["r_pass"] = bool(out["r_spread"] >= 0.05 and out["r_spread"] >= 3 * floor_r)
    out["H1"] = bool(out["evaluable"] and (out["nu_pass"] or out["eps_pass"]))
    out["H1s"] = bool(out["evaluable"] and out["r_pass"])
    return out


def quantifier(decisions, key, frac=0.5):
    ev = [d for d in decisions if d.get("evaluable", True)]
    if not ev:
        return dict(passed=False, n=0, k=0)
    k = sum(bool(d[key]) for d in ev)
    return dict(passed=bool(k >= frac * len(ev)), n=len(ev), k=k)


# --- distribution distances (measure tests, H2) ---------------------------------------------------
def energy_distance(X, Y):
    """Energy distance between samples X (n, p) and Y (m, p)."""
    X, Y = np.atleast_2d(X).astype(float), np.atleast_2d(Y).astype(float)
    if X.shape[0] == 1 and X.shape[1] > 1 and Y.shape[1] == 1:
        X = X.T
    dxy = np.linalg.norm(X[:, None] - Y[None], axis=-1).mean()
    dxx = np.linalg.norm(X[:, None] - X[None], axis=-1).mean()
    dyy = np.linalg.norm(Y[:, None] - Y[None], axis=-1).mean()
    return float(2 * dxy - dxx - dyy)


def energy_test(X, Y, n_perm=500, seed=0):
    """Permutation p-value of the energy distance."""
    X, Y = np.atleast_2d(X).astype(float), np.atleast_2d(Y).astype(float)
    d0 = energy_distance(X, Y)
    Z = np.concatenate([X, Y])
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    cnt = 0
    for _ in range(n_perm):
        p = rng.permutation(Z.shape[0])
        cnt += energy_distance(Z[p[:n]], Z[p[n:]]) >= d0
    return d0, (cnt + 1) / (n_perm + 1)


def sliced_w1(X, Y, n_proj=256, seed=0):
    """Sliced 1-Wasserstein distance between equal-size or unequal-size samples (quantile matching)."""
    X, Y = np.atleast_2d(X).astype(float), np.atleast_2d(Y).astype(float)
    rng = np.random.default_rng(seed)
    th = rng.standard_normal((n_proj, X.shape[1]))
    th /= np.linalg.norm(th, axis=1, keepdims=True)
    qs = np.linspace(0.005, 0.995, 200)
    a = np.quantile(X @ th.T, qs, axis=0)
    b = np.quantile(Y @ th.T, qs, axis=0)
    return float(np.mean(np.abs(a - b)))


def quantile_coverage(samples, reference, levels=(0.05, 0.25, 0.5, 0.75, 0.95)):
    """Fraction of reference values below each sample quantile (ideal: the level itself)."""
    s, r = np.asarray(samples, float), np.asarray(reference, float)
    return {lv: float(np.mean(r <= np.quantile(s, lv))) for lv in levels}


def kendall_ci(x, y, B=2000, seed=0):
    """Kendall tau with a bootstrap 95% interval (H4: width vs distance to the bound)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    tau = stats.kendalltau(x, y).statistic
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(B):
        i = rng.integers(0, x.size, x.size)
        bs.append(stats.kendalltau(x[i], y[i]).statistic)
    bs = np.array(bs)[np.isfinite(bs)]
    return float(tau), (float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975)))


def spearman(x, y):
    return float(stats.spearmanr(x, y).statistic)
