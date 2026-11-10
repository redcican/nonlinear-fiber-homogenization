"""G1 decisions under the compression protocol of 9 Nov 2026, from the width-pilot results and the cell
re-evaluations: eps_cr = first loss of cell stability (fs3d, fundamental branch); nu(20%) and r(20%) on the
stable branch (qoi_compression_branch.json) and in tension (qoi_tension.json) when present. The floors of the
stable-branch nu(20%) and r(20%) come from floor_branch.json (scripts/branch_floor.py); targets without the 2x
check use the largest (b) of the targets with it, as frozen on 6 Nov.

    python -I scripts/decide_g1.py runs/g1/t*/
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nlfiber import widths as W  # noqa: E402


def branch_floors(runs):
    fl = {}
    for run in runs:
        g = Path(run) / "floor_branch.json"
        if g.exists():
            J = json.load(open(g))
            fl[J["target"]] = J["floor"]
            if len(J["base"]) < 2:  # (b) compares pairs: not measured with one base cell
                for k in ("nu20", "r20"):
                    fl[J["target"]][k]["b"] = np.nan
    bmax = {k: max([v[k]["b"] for v in fl.values() if np.isfinite(v[k].get("b", np.nan))], default=np.nan)
            for k in ("nu20", "r20")}
    out = {}
    for t, v in fl.items():
        out[t] = {k: W.target_floor(v[k]["a"], v[k]["b"] if np.isfinite(v[k]["b"]) else bmax[k], v[k]["c"])
                  for k in ("nu20", "r20")}
    return out


def main(runs):
    decisions = []
    floors = branch_floors(runs)
    # targets without the 2x check take the largest (b) of the targets with it (frozen 6 Nov)
    def eps_floor(r):
        r = Path(r)
        if (r / "floor_onset.json").exists():
            return json.load(open(r / "floor_onset.json"))["floor"]["eps_cell"]
        return json.load(open(r / "results.json"))["results"][0]["floor"]["eps_cell"]
    b_eps = [eps_floor(r).get("b") for r in runs if (Path(r) / "results.json").exists()]
    b_eps_max = max([b for b in b_eps if b is not None and np.isfinite(b)], default=np.nan)
    print("| target | n | onset ε_cell q10–q90 | ratio | ln floor | pass (≥1.5, ≥3 floor) | stable ν(20%) spread | r(20%) spread | "
          "floor ν | floor r | stable ν(10%) | r(10%) | median contact-free range | tension ν(20%) spread | r(20%) spread |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for run in runs:
        run = Path(run)
        f = run / "results.json"
        if not f.exists():
            continue
        r = json.load(open(f))["results"][0]
        eps = np.array([s["qoi"].get("eps_cell", np.nan) for s in r["samples"]], float)
        n = len(eps)
        ok = eps[np.isfinite(eps)]
        ratio = W.ratio(ok) if ok.size >= 5 else np.nan
        fl = dict(r["floor"]["eps_cell"])
        if (run / "floor_onset.json").exists():  # first four accepted cells had < 2 onsets: floor on cells with one
            fl = dict(json.load(open(run / "floor_onset.json"))["floor"]["eps_cell"])
        if fl.get("b") is None or not np.isfinite(fl["b"]):
            fl["b"] = b_eps_max
        comps = [v for v in (fl.get("a"), fl.get("b"), fl.get("c")) if v is not None and np.isfinite(v)]
        base = np.nanmedian(ok) if ok.size else np.nan
        lnfloor = max(comps) / base if comps and base else np.nan  # relative change of the onset
        passed = bool(n >= 10 and ok.size >= 5 and ratio >= 1.5 and np.log(ratio) >= 3 * lnfloor)
        row = dict(target=r["target"]["id"], n=n, ratio=ratio, lnfloor=lnfloor, eps_pass=passed, evaluable=n >= 10)
        extra = {}
        for tag, key in (("stable", "qoi_compression_branch.json"), ("tension", "qoi_tension.json")):
            g = run / key
            if g.exists():
                J = json.load(open(g))
                sp = J["spreads"]
                cells = J["cells"].values()
                nu10 = W.spread([c.get("nu10", np.nan) for c in cells])
                r10 = W.spread([c.get("r10", np.nan) for c in cells])
                reached = [c.get("reached", np.nan) for c in cells]
                extra[tag] = (sp["nu20"]["spread"], sp["r20"]["spread"], sp["nu20"]["n"], nu10, r10,
                              float(np.nanmedian(reached)))
        q = lambda x: f"{x:.3f}" if x == x else "–"  # noqa: E731
        st = extra.get("stable", (np.nan, np.nan, 0, np.nan, np.nan, np.nan))
        te = extra.get("tension", (np.nan, np.nan, 0, np.nan, np.nan, np.nan))
        row["nu_stable"], row["r_stable"] = st[0], st[1]
        fb = floors.get(row["target"], dict(nu20=np.nan, r20=np.nan))
        row["floor_nu"], row["floor_r"] = fb["nu20"], fb["r20"]
        # a spread counts only with its floor measured: >= 0.1 (nu) or 0.05 (r) and >= 3x the floor
        nu_pass = bool(st[0] == st[0] and fb["nu20"] == fb["nu20"] and st[0] >= 0.1 and st[0] >= 3 * fb["nu20"])
        r_pass = bool(st[1] == st[1] and fb["r20"] == fb["r20"] and st[1] >= 0.05 and st[1] >= 3 * fb["r20"])
        row["H1"] = bool(row["evaluable"] and (passed or nu_pass))
        row["H1s"] = bool(row["evaluable"] and r_pass)
        decisions.append(row)
        print(f"| {row['target']} | {n} | {q(np.quantile(ok, 0.1) if ok.size else np.nan)}–{q(np.quantile(ok, 0.9) if ok.size else np.nan)} | "
              f"{q(ratio)} | {q(lnfloor)} | {'pass' if passed else 'fail'} | {q(st[0])} (n {st[2]}) | {q(st[1])}"
              f"{' ✓' if r_pass else ''} | "
              f"{q(fb['nu20'])} | {q(fb['r20'])} | {q(st[3])} | {q(st[4])} | {q(st[5])} | {q(te[0])} (n {te[2]}) | {q(te[1])} |")
    print()
    pooled = {k: W.target_floor(*[f[k] for f in floors.values()]) for k in ("nu20", "r20")} if floors else {}
    print("pooled stable-branch floors:", pooled)
    print("H1 quantifier:", W.quantifier(decisions, "H1"), " H1s:", W.quantifier(decisions, "H1s"))


if __name__ == "__main__":
    main(sys.argv[1:])
