"""Noise floor of the stable-branch compression QoIs nu(20 %) and r(20 %) (fs3d -branch), as frozen on 6 Nov.

    python -I scripts/branch_floor.py runs/g1/t0 runs/g1/t2 --gpu 0
Base cells: the first four accepted cells (by seed) whose stable branch reaches 20 % contact-free in
<run>/qoi_compression_branch.json (scripts/eval_cells.py --branch). Components:
  (a) path: re-run with another branch kick (0.03 instead of 0.01) and with half the load step;
  (b) resolution: the same designs at 2x (128^3, re-thresholded at the same volume), change of pairwise
      differences; on even targets only, the others use the largest (b) of those (scripts/decide_g1.py);
  (c) stiffness slack: two base cells polished to 0.05 % binary C^H error and re-run.
Writes <run>/floor_branch.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax.numpy as jnp  # noqa: E402

from nlfiber import fiber as Fb, project as P, qoi as Q, widths as W  # noqa: E402
from nlfiber.backend_fs3d import FS3D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from width_pilot import resample_binary  # noqa: E402

KEYS = ("nu20", "r20")


def qois(be, rho, E0, tag, de=0.01, kick=None):
    rows = be.run_path(rho, "compression", emax=0.2, de=de, crit="1", tag=tag, branch=True, branch_kick=kick)
    return Q.path_qois(rows, E0, at=(0.1, 0.2))


def main(a):
    for run in a.runs:
        run = Path(run)
        R = json.loads((run / "results.json").read_text())
        res = R["results"][0]
        t_id = int(res["target"]["id"])
        c_star = np.asarray(res["target"]["c"], np.float64)
        qb = json.loads((run / "qoi_compression_branch.json").read_text())["cells"]
        cells = np.load(next(run.glob("target*_cells.npz")), allow_pickle=False)
        seeds = sorted(int(k.split("/cell")[1]) for k, q in qb.items() if np.isfinite(q.get("r20", np.nan)))
        base = seeds[: a.n_base]
        be = FS3D(workdir=run / "fs3d_floor", gpu=a.gpu)
        fib = Fb.Fiber(R["args"]["n_o"], v=R["args"]["v"], r=R["args"]["r"])
        floor = {k: dict(a=[], b=[[], []], c=[]) for k in KEYS}
        log = dict(target=t_id, base=base, runs=[])

        def ref(s):
            return next(q for k, q in qb.items() if k.endswith(f"/cell{s}"))

        for s in base:
            rho = cells[f"cell{s}"].astype(np.float64)
            q0 = ref(s)
            E0 = q0["E0"] if "E0" in q0 else None
            if E0 is None:
                C, _ = be.moduli(rho, tag=f"mod{s}")
                E0 = Q.young_x(C)
            for tag, kw in (("kick", dict(kick=0.03)), ("half", dict(de=0.005))):
                q2 = qois(be, rho, E0, f"{tag}{s}", **kw)
                for k in KEYS:
                    floor[k]["a"].append((q0.get(k, np.nan), q2.get(k, np.nan)))
                log["runs"].append(dict(seed=s, kind=tag, **{k: q2.get(k, np.nan) for k in KEYS}, reached=q2["reached"]))
                print(run.name, s, tag, {k: round(q2.get(k, np.nan), 4) for k in KEYS}, flush=True)
            if t_id % 2 == 0:
                fine = resample_binary(fib, cells[f"phi{s}"], 2.0).astype(np.float64)
                Cf, _ = be.moduli(fine, tag=f"modf{s}")
                q3 = qois(be, fine, Q.young_x(Cf), f"fine{s}")
                for k in KEYS:
                    floor[k]["b"][0].append(q0.get(k, np.nan))
                    floor[k]["b"][1].append(q3.get(k, np.nan))
                log["runs"].append(dict(seed=s, kind="2x", **{k: q3.get(k, np.nan) for k in KEYS}, reached=q3["reached"]))
                print(run.name, s, "2x", {k: round(q3.get(k, np.nan), 4) for k in KEYS}, flush=True)
        for s in base[:2]:
            q0 = ref(s)
            _, pol = P.polish_binary(fib, jnp.asarray(cells[f"phi{s}"]), c_star, 16.0, up=1, steps=8, tol_bin=5e-4)
            d2 = np.asarray(pol["binary"]["cell"], np.float64)
            C2, _ = be.moduli(d2, tag=f"modc{s}")
            q4 = qois(be, d2, Q.young_x(C2), f"slack{s}")
            for k in KEYS:
                floor[k]["c"].append((q0.get(k, np.nan), q4.get(k, np.nan)))
            log["runs"].append(dict(seed=s, kind="slack", slack=float(pol["err_bin"]), **{k: q4.get(k, np.nan) for k in KEYS},
                                    reached=q4["reached"]))
            print(run.name, s, f"slack {pol['err_bin']:.1e}", {k: round(q4.get(k, np.nan), 4) for k in KEYS}, flush=True)

        def comp(k):
            f = floor[k]
            fa = W.path_floor([x for x, _ in f["a"]], [[y for _, y in f["a"]]]) if f["a"] else np.nan
            fb = W.difference_floor(*f["b"]) if f["b"][0] else np.nan
            fc = W.path_floor([x for x, _ in f["c"]], [[y for _, y in f["c"]]]) if f["c"] else np.nan
            return dict(a=fa, b=fb, c=fc)

        log["floor"] = {k: comp(k) for k in KEYS}
        (run / "floor_branch.json").write_text(json.dumps(log, default=float, indent=1))
        print(run.name, json.dumps(log["floor"], default=float), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--n-base", type=int, default=4)
    main(ap.parse_args())
