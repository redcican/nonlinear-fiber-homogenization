"""Load paths L1-L4 and the nonlinear quantities of interest of the plan (M3).

  L1 tension along x, lateral stresses zero (Fbar_yy, Fbar_zz free)      nu(e), r(e), stiffening
  L2 compression along x, same control                                  nu(e), r(e), eps_cr
  L3 simple shear Fbar = I + g e_x (x) e_y                               G_t(g) / G
  L4 volumetric Fbar = (1 + e) I                                         P(e)

Each step is warm-started from the previous state plus a random kick (default 1e-3 of max|w|), so the path
can leave a symmetric branch; a step that fails is split in two (up to 3 levels). QoIs:
  nu(e)      secant Poisson's ratio -(Fbar_yy - 1) / e
  r(e)       secant ratio Pbar_xx(e) / (E0 e), E0 the small-strain Young's modulus along x
  E_t(e)     tangent modulus d Pbar_xx / d e (central differences along the path)
  eps_cr     first load maximum (compression), the kink strain, the first loss of cell stability
             (lambda_min <= 0), and of strong ellipticity of the effective tangent
"""
import time

import numpy as np
import jax.numpy as jnp

from .mech import Mixed, effective_tangent, rank_one_min

LATERAL = {0: [(1, 1), (2, 2)], 1: [(0, 0), (2, 2)], 2: [(0, 0), (1, 1)]}


def _solve_step(mix, Fb, z, depth=0, F_from=None):
    w, F, log = mix.solve(Fb, z0=z)
    tot = dict(newton=log["newton"], cg=log["cg"], restarts=0, splits=0)
    while not log["converged"] and not log.get("linesearch_failed") and tot["restarts"] < 3:
        w, F, log = mix.solve(Fb, z0=log["z"], g_ref=log["g0"])
        tot["newton"] += log["newton"]
        tot["cg"] += log["cg"]
        tot["restarts"] += 1
    if log["converged"] or depth >= 3 or F_from is None:
        return w, F, log, tot
    Fm = 0.5 * (np.asarray(F_from) + np.asarray(Fb))
    _, _, l1, t1 = _solve_step(mix, Fm, z, depth + 1, F_from)
    for k in tot:
        tot[k] += t1[k]
    if not l1["converged"]:
        return w, F, log, tot
    w, F, log, t2 = _solve_step(mix, Fb, l1["z"], depth + 1, Fm)
    for k in tot:
        tot[k] += t2[k]
    tot["splits"] += 1
    return w, F, log, tot


def run_path(cell, path="tension", emax=0.2, de=0.01, axis=0, kick=1e-3, seed=0, stability=False, k_eigs=2,
             rank_one=False, contact_check=True, verbose=False):
    """Run one load path. Returns a list of per-step dicts (the state z of the last converged step under
    key 'z' of the final row is kept for warm restarts)."""
    if path in ("tension", "compression"):
        mix = Mixed(cell, LATERAL[axis])
        sign = 1.0 if path == "tension" else -1.0

        def Fbar(e):
            F = np.eye(3)
            F[axis, axis] = 1 + e
            return F
    elif path == "shear":
        mix, sign = Mixed(cell, []), 1.0

        def Fbar(g):
            F = np.eye(3)
            F[0, 1] = g
            return F
    elif path in ("vol+", "vol-"):
        mix, sign = Mixed(cell, []), (1.0 if path == "vol+" else -1.0)

        def Fbar(e):
            return (1 + e) * np.eye(3)
    else:
        raise ValueError(path)
    rng = np.random.default_rng(seed)
    rows, z, Fprev, X = [], None, np.eye(3), None
    nsteps = int(round(emax / de))
    void = np.asarray(cell.rho) < 0.5
    for k in range(1, nsteps + 1):
        e = sign * de * k
        t0 = time.time()
        Fb = Fbar(e)
        if z is not None and mix.k:
            Fb = np.array(mix.unpack(z, jnp.asarray(Fb))[1])  # keep the free components of the last state
            Fb[axis, axis] = 1 + e
        w, F, log, tot = _solve_step(mix, Fb, z, F_from=Fprev)
        ok = bool(log["converged"])
        F = np.asarray(F)
        P = np.asarray(cell.mean_stress(w, F))
        row = dict(path=path, e=e, P=P, F=F, converged=ok, newton=tot["newton"], cg=tot["cg"],
                   splits=tot["splits"], negative_curvature=bool(log["negative_curvature"]))
        if path in ("tension", "compression"):
            lat = [i for i in range(3) if i != axis]
            row["nu"] = [-(F[i, i] - 1) / e for i in lat]
        if contact_check and void.any():
            Hg = np.asarray(cell.grad_w(w))
            J = np.linalg.det(np.eye(3) + Hg + (F - np.eye(3)))
            row["void_inverted"] = float(np.mean((J < 0.1)[:, void]))
        if stability and ok:
            vals, vecs, its = mix.lowest_eigs(log["z"], F, k=k_eigs, X=X)
            X = np.asarray(vecs) / np.asarray(mix._diag_scale())[:, None]
            row["lam"] = vals
        if rank_one and ok:
            A = effective_tangent(cell, w, F)
            row["rank_one"], _ = rank_one_min(A, plane=cell.shape[2] == 1)
        row["seconds"] = time.time() - t0
        rows.append(row)
        if verbose:
            print(f"  {path} e {e:+.3f} P {P[axis, axis] if mix.k else P[0, 1]:+.5f} newton {tot['newton']} "
                  f"{'ok' if ok else 'FAILED'} {row['seconds']:.1f}s", flush=True)
        if not ok:
            break
        if contact_check and row.get("void_inverted", 0) > 0.01:
            row["contact"] = True
            break
        z = log["z"]
        if kick > 0:
            nw = mix.nw
            wscale = float(jnp.abs(z[:nw]).max()) + 1e-12
            z = z.at[:nw].add(jnp.asarray(rng.standard_normal(nw)) * kick * wscale)
        Fprev = F
    return rows


def young_x(C):
    """Small-strain Young's modulus along x from a 6x6 C."""
    S = np.linalg.inv(np.asarray(C))
    return 1.0 / S[0, 0]


def nu0(C):
    S = np.linalg.inv(np.asarray(C))
    return -S[1, 0] / S[0, 0]


def path_qois(rows, E0, axis=0, at=(0.1, 0.2)):
    """QoIs of a tension or compression path. Values at strains not reached are nan."""
    rows = [r for r in rows if r["converged"]]
    if not rows:
        return dict(reached=0.0)
    e = np.array([r["e"] for r in rows])
    P = np.array([r["P"][axis, axis] for r in rows])
    nu = np.array([np.mean(r["nu"]) for r in rows]) if "nu" in rows[0] else None
    out = dict(reached=float(np.abs(e).max()))

    def at_e(x, val):
        i = np.where(np.isclose(np.abs(e), x, atol=1e-9))[0]
        return float(val[i[0]]) if len(i) else float("nan")

    for x in at:
        tag = f"{int(round(100 * x))}"
        if nu is not None:
            out[f"nu{tag}"] = at_e(x, nu)
        out[f"r{tag}"] = at_e(x, P / (E0 * e))
    # tangent modulus along the path (with the origin as the first point)
    e0 = np.concatenate([[0.0], e])
    P0 = np.concatenate([[0.0], P])
    Et = np.gradient(P0, e0)
    Et[0] = E0
    for x in at:
        i = np.where(np.isclose(np.abs(e0), x, atol=1e-9))[0]
        out[f"S{int(round(100 * x))}"] = float(Et[i[0]] / E0) if len(i) else float("nan")
    # first load maximum: |P| stops increasing
    absP = np.abs(P0)
    imax = np.where(np.diff(absP) <= 0)[0]
    out["eps_max"] = float(abs(e0[imax[0]])) if len(imax) else float("nan")
    # kink: end of the first step whose secant tangent is below half the previous one
    dP = np.diff(absP) / np.diff(np.abs(e0))
    ik = np.where(dP[1:] < 0.5 * dP[:-1])[0]
    out["eps_kink"] = float(abs(e0[ik[0] + 2])) if len(ik) else float("nan")
    if "lam" in rows[0]:
        lam = np.array([r["lam"][0] for r in rows])
        neg = np.where(lam <= 0)[0]
        if len(neg):
            i = neg[0]
            if i == 0:
                out["eps_cell"] = float(abs(e[0]))
            else:  # linear interpolation of the sign change
                t = lam[i - 1] / (lam[i - 1] - lam[i])
                out["eps_cell"] = float(abs(e[i - 1] + t * (e[i] - e[i - 1])))
        else:
            out["eps_cell"] = float("nan")
    if "rank_one" in rows[0]:
        ro = np.array([r["rank_one"] for r in rows])
        neg = np.where(ro <= 0)[0]
        out["eps_rank1"] = float(abs(e[neg[0]])) if len(neg) else float("nan")
    return out


def censored(x, cap):
    """Censored value: nan (no event before the end of the path) -> cap."""
    return cap if not np.isfinite(x) else x
