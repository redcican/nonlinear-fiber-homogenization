"""Markdown summary of one or more width-pilot result files.

    python -I scripts/summarize_pilot.py runs/pilot32/results.json runs/pilot32_fs3d/results.json
"""
import json
import sys

import numpy as np


def fmt(x, d=3):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "–"
    return "–" if not np.isfinite(x) else f"{x:.{d}f}"


def main(paths):
    for p in paths:
        R = json.load(open(p))
        a = R["args"]
        print(f"## {p}\n")
        print(f"cell {2 * a['n_o']}^3, v = {a['v']}, evaluator {a.get('backend', 'jax')}, {a['eval']} designs, "
              f"compression to {a['emax']} in steps of {a['de']}\n")
        print("| target | c* (C11, C12, C44)/E | n | IoU | ν(20%) q05–q95 | floor | r(20%) spread | floor | "
              "ε_max ratio | ε_cell ratio | H1 | H1s |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in R["results"]:
            t, d, f = r["target"], r["decision"], r["floor"]
            qs = [s["qoi"] for s in r["samples"]]
            eps_cell = [q.get("eps_cell", np.nan) for q in qs]
            ec = [e for e in eps_cell if e is not None and np.isfinite(e)]
            ratio_cell = (np.quantile(ec, 0.9) / np.quantile(ec, 0.1)) if len(ec) >= 2 else np.nan
            print(f"| {t['id']} | ({', '.join(fmt(x, 4) for x in t['c'])}) | {r['n_accepted']} | {fmt(r['mean_iou'], 2)} | "
                  f"{fmt(d['nu_spread'])} | {fmt(f['nu20']['floor'])} | {fmt(d['r_spread'])} | {fmt(f['r20']['floor'])} | "
                  f"{fmt(d['eps_ratio'], 2)} | {fmt(ratio_cell, 2)} | {'pass' if d['H1'] else 'fail'} | {'pass' if d['H1s'] else 'fail'} |")
        if "summary" in R:
            s = R["summary"]
            print(f"\nQuantifier: H1 {s['H1']['k']}/{s['H1']['n']} targets, H1s {s['H1s']['k']}/{s['H1s']['n']} targets.\n")
        print("\nPer-sample QoIs:\n")
        print("| target | seed | C^H err | ν0 | ν(20%) | r(20%) | S(20%) | ε_max | ε_kink | ε_cell | Newton/step | s |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in R["results"]:
            for s_ in r["samples"]:
                q = s_["qoi"]
                print(f"| {r['target']['id']} | {s_['seed']} | {fmt(s_['err_bin'], 4)} | {fmt(q.get('nu0'))} | {fmt(q.get('nu20'))} | "
                      f"{fmt(q.get('r20'))} | {fmt(q.get('S20'))} | {fmt(q.get('eps_max'))} | {fmt(q.get('eps_kink'))} | "
                      f"{fmt(q.get('eps_cell'))} | {fmt(q.get('newton_per_step'), 1)} | {fmt(q.get('seconds'), 0)} |")
        print()


if __name__ == "__main__":
    main(sys.argv[1:])
