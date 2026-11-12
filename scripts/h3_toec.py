"""H3 pilot: does the spread of the third-order elastic constants (TOECs) predict the measured spread at 5 %?

    python -I scripts/h3_toec.py eval runs/g1/t0 runs/g1/t1 --gpu 0          # fs3d: C, T and 5 % paths per cell
    python -I scripts/h3_toec.py analyze --sets g1=runs/g1/t{0..7} valid=runs/g1r25m/t{0,4,6}

eval writes <run>/h3_cells.json (resumable): per cell C, T, the TOEC prediction of nu and r at 1, 5 and 20 %
under uniaxial stress (qoi.toec_uniaxial), and the fs3d measurement on the fundamental branch to 5 % in tension
(L1) and compression (L2). analyze applies the frozen protocol (plan, amendment of 12 Nov 2026): per (target,
path) the spread over cells (q05-q95, or max - min below 10 cells) of predicted and measured r(5 %) and nu(5 %);
Spearman rho over all pairs with >= 5 cells; H3 pilot passes if rho(r) >= 0.6. The 20 % comparison uses the
stable-branch and tension measurements of qoi_*.json and is reported only.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nlfiber import qoi as Q, widths as W  # noqa: E402

STRAINS = (0.01, 0.05, 0.2)
SIGN = {"tension": 1.0, "compression": -1.0}


def predict(C, T, E0):
    out = {}
    for path, sg in SIGN.items():
        for e in STRAINS:
            p = Q.toec_uniaxial(C, T, sg * e)
            tag = f"{int(round(100 * e))}"
            out[f"{path}_r{tag}"] = p["P"] / (E0 * sg * e)
            out[f"{path}_nu{tag}"] = p["nu"]
    return out


def evaluate(a):
    from nlfiber.backend_fs3d import FS3D
    for run in a.runs:
        run = Path(run)
        outf = run / "h3_cells.json"
        done = json.loads(outf.read_text()) if outf.exists() else {}
        be = FS3D(workdir=run / "fs3d_h3", gpu=a.gpu)
        for f in sorted(run.glob("target*_cells.npz")):
            d = np.load(f, allow_pickle=False)
            for key in d.files:
                if not key.startswith("cell"):
                    continue
                name = f"{f.stem}/{key}"
                if name in done:
                    continue
                rho = d[key].astype(np.float64)
                C, T = be.moduli(rho, tag=f"mod_{key}")
                E0 = Q.young_x(C)
                rec = dict(C=C.tolist(), T=T.tolist(), E0=E0, vol=float(rho.mean()), pred=predict(C, T, E0), meas={})
                for path in SIGN:
                    rows = be.run_path(rho, path, emax=0.05, de=0.01, crit="1", tag=f"h3_{path}_{key}",
                                       truncate_unstable=True)
                    q = Q.path_qois(rows, E0, at=(0.01, 0.05))
                    for k in ("r1", "r5", "nu1", "nu5"):
                        rec["meas"][f"{path}_{k}"] = q.get(k, np.nan)
                    rec["meas"][f"{path}_onset"] = q.get("eps_cell", np.nan)
                done[name] = rec
                outf.write_text(json.dumps(done, default=float))
                print(run.name, name, {k: round(v, 4) for k, v in rec["meas"].items() if k.endswith(("r5", "nu5"))},
                      {k: round(rec["pred"][k], 4) for k in ("compression_r5", "tension_r5")}, flush=True)


def spearman(x, y):
    from scipy.stats import spearmanr
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return dict(rho=np.nan, p=np.nan, n=int(ok.sum()))
    r = spearmanr(x[ok], y[ok])
    return dict(rho=float(r.statistic), p=float(r.pvalue), n=int(ok.sum()))


def analyze(a):
    sets = {}
    for s in a.sets:
        name, runs = s.split("=", 1)
        sets[name] = runs.split(",")
    pairs, cellwise, at20 = [], [], []
    for sname, runs in sets.items():
        for run in runs:
            run = Path(run)
            H = json.loads((run / "h3_cells.json").read_text())
            m20 = {p: json.loads((run / f).read_text())["cells"] for p, f in
                   (("compression", "qoi_compression_branch.json"), ("tension", "qoi_tension.json")) if (run / f).exists()}
            for path in SIGN:
                for qty in ("r", "nu"):
                    pv = np.array([h["pred"][f"{path}_{qty}5"] for h in H.values()])
                    mv = np.array([h["meas"][f"{path}_{qty}5"] for h in H.values()], float)
                    ok = np.isfinite(pv) & np.isfinite(mv)
                    rec = dict(set=sname, run=run.name, path=path, qty=qty, n=int(ok.sum()),
                               pred=W.spread(pv[ok]) if ok.sum() >= 2 else np.nan,
                               meas=W.spread(mv[ok]) if ok.sum() >= 2 else np.nan,
                               err_med=float(np.median(np.abs(pv[ok] - mv[ok]))) if ok.any() else np.nan)
                    pairs.append(rec)
                    if qty == "r" and ok.sum() >= 5:
                        cellwise.append(dict(set=sname, run=run.name, path=path, **spearman(pv[ok], mv[ok])))
                    if path in m20:
                        p20 = np.array([h["pred"][f"{path}_{qty}20"] for h in H.values()])
                        q20 = np.array([m20[path].get(k, {}).get(f"{qty}20", np.nan) for k in H], float)
                        ok2 = np.isfinite(p20) & np.isfinite(q20)
                        at20.append(dict(set=sname, run=run.name, path=path, qty=qty, n=int(ok2.sum()),
                                         pred=W.spread(p20[ok2]) if ok2.sum() >= 2 else np.nan,
                                         meas=W.spread(q20[ok2]) if ok2.sum() >= 2 else np.nan))
    counted = [r for r in pairs if r["n"] >= 5]

    def rho_of(rows, qty, **flt):
        rs = [r for r in rows if r["qty"] == qty and all(r[k] == v for k, v in flt.items())]
        return spearman([r["pred"] for r in rs], [r["meas"] for r in rs])

    print("| set | target | path | n | pred r(5%) spread | meas r(5%) spread | median |Δr| | pred ν(5%) spread | meas ν(5%) spread |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in [x for x in pairs if x["qty"] == "r"]:
        nu = next(x for x in pairs if x["qty"] == "nu" and x["run"] == r["run"] and x["set"] == r["set"] and x["path"] == r["path"])
        q = lambda v: f"{v:.4f}" if v == v else "–"  # noqa: E731
        print(f"| {r['set']} | {r['run']} | {r['path']} | {r['n']} | {q(r['pred'])} | {q(r['meas'])} | {q(r['err_med'])} | "
              f"{q(nu['pred'])} | {q(nu['meas'])} |")
    primary = rho_of(counted, "r")
    print("\nPRIMARY rho(r 5%) over counted (target, path) pairs:", primary, "-> H3 pilot", "PASS" if primary["rho"] >= 0.6 else "FAIL")
    print("rho(nu 5%):", rho_of(counted, "nu"))
    for sname in sets:
        print(f"set {sname}: rho(r) {rho_of(counted, 'r', set=sname)}")
    for path in SIGN:
        print(f"path {path}: rho(r) {rho_of(counted, 'r', path=path)}")
    cw = [c["rho"] for c in cellwise if np.isfinite(c["rho"])]
    print("per-cell Spearman(pred, meas r5) within targets: median", float(np.median(cw)) if cw else np.nan,
          "range", (min(cw), max(cw)) if cw else None, "n", len(cw))
    c20 = [r for r in at20 if r["n"] >= 5]
    print("20 % (reported): rho(r)", rho_of(c20, "r"), " rho(nu)", rho_of(c20, "nu"))
    if a.out:
        Path(a.out).write_text(json.dumps(dict(pairs=pairs, cellwise=cellwise, at20=at20, primary=primary), default=float, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval")
    e.add_argument("runs", nargs="+")
    e.add_argument("--gpu", type=int, default=None)
    z = sub.add_parser("analyze")
    z.add_argument("--sets", nargs="+", required=True, help="name=run1,run2,...")
    z.add_argument("--out", default=None)
    a = ap.parse_args()
    evaluate(a) if a.cmd == "eval" else analyze(a)
