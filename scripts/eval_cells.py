"""Re-evaluate the accepted cells of width-pilot runs along a chosen path and write their QoIs.

    python -I scripts/eval_cells.py runs/g1/t0 runs/g1/t1 --path tension --backend fs3d [--branch]
Reads <run>/target*_cells.npz (cell<seed> arrays), writes <run>/qoi_<path>[_branch].json with per-cell QoIs
and the q05–q95 spreads. With --branch the fs3d path switches to the stable branch at a loss of stability
(fs3d -branch), so the 20 % quantities under compression are read on the buckled branch.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nlfiber import qoi as Q, widths as W, linear as L, mech as Me  # noqa: E402


def main(a):
    be = None
    if a.backend == "fs3d":
        from nlfiber.backend_fs3d import FS3D
    for run in a.runs:
        run = Path(run)
        if a.backend == "fs3d":
            be = FS3D(workdir=run / "fs3d_eval")
        out = {}
        for f in sorted(run.glob("target*_cells.npz")):
            d = np.load(f, allow_pickle=False)
            for key in d.files:
                if not key.startswith("cell"):
                    continue
                rho = d[key].astype(np.float64)
                if be is not None:
                    C, _ = be.moduli(rho, tag=f"mod_{key}")
                    rows = be.run_path(rho, a.path, emax=a.emax, de=a.de, crit="1", tag=f"{a.path}_{key}",
                                       truncate_unstable=not a.branch, branch=a.branch)
                else:
                    C, _, _ = L.LinearHomogenization(rho.shape).C_full(rho)
                    C = np.asarray(C)
                    rows = Q.run_path(Me.Cell(rho), a.path, emax=a.emax, de=a.de, stability=True)
                q = Q.path_qois(rows, Q.young_x(C), at=(0.1, 0.2))
                q["nu0"] = Q.nu0(C)
                out[f"{f.stem}/{key}"] = q
                print(run.name, f.stem, key, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in q.items()
                                            if k in ("nu20", "r20", "eps_max", "eps_cell", "reached")}, flush=True)
        spreads = {}
        for k in ("nu20", "r20", "S20"):
            vals = [q.get(k, np.nan) for q in out.values()]
            spreads[k] = dict(spread=W.spread(vals), n=int(np.isfinite(vals).sum()))
        for k in ("eps_max", "eps_cell"):
            vals = np.abs([q.get(k, np.nan) for q in out.values()])
            spreads[k] = dict(ratio=W.ratio(vals), n=int(np.isfinite(vals).sum()))
        tag = f"qoi_{a.path}{'_branch' if a.branch else ''}.json"
        (run / tag).write_text(json.dumps(dict(path=a.path, backend=a.backend, branch=a.branch, cells=out, spreads=spreads),
                                          default=float, indent=1))
        print(run.name, json.dumps(spreads), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--path", choices=("tension", "compression"), default="tension")
    ap.add_argument("--backend", choices=("jax", "fs3d"), default="fs3d")
    ap.add_argument("--branch", action="store_true", help="fs3d branch switching at a loss of stability")
    ap.add_argument("--emax", type=float, default=0.2)
    ap.add_argument("--de", type=float, default=0.01)
    main(ap.parse_args())
