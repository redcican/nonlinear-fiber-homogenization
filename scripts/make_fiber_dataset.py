"""Training cells on fibers: random symmetric starts projected onto the stiffness of other random cells.

For the prior-coverage question (H7): procedural cells alone leave the prior far from stiff, low-C12 fibers.
Each item: a random target cell (random start, binary at the design grid, its invariants c*), and `per_target`
random starts projected onto c* (grey, continuation to beta 16). Saved as pre-densities phi (octants) with c*,
the achieved c and the grey error, in <out>/fiber_<n_o>.npz (appended in chunks so a crash loses little).

    python -I scripts/make_fiber_dataset.py --n-o 16 --targets 60 --per-target 4 --out data
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nlfiber import fiber as Fb, project as P, topology as T  # noqa: E402


def main(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"fiber_{a.n_o}.npz"
    phis, cs, ts, errs, tid = [], [], [], [], []
    if f.exists():
        d = np.load(f, allow_pickle=False)
        phis, cs, ts, errs, tid = (list(d[k]) for k in ("phi", "c", "target", "err", "target_id"))
    done = set(tid)
    rng = np.random.default_rng(a.seed)
    fib = Fb.Fiber(a.n_o, v=a.v, r=1.5)
    k, seed_t = 0, 5000 + a.seed * 10000
    t_start = time.time()
    while k < a.targets:
        phi_t = Fb.random_start(a.n_o, seed=seed_t, kmax=2)
        seed_t += 1
        chk = fib.binary_check(phi_t, up=1)
        val = T.validity(chk["cell"])
        if abs(chk["vol"] - a.v) > 0.02 or val["components"] != 1 or val["spanning"] != 1:
            continue
        if k in done:
            k += 1
            continue
        c_star = chk["c"]
        for j in range(a.per_target):
            phi0 = Fb.random_start(a.n_o, seed=int(rng.integers(1 << 30)), kmax=2)
            phi, info = P.project(fib, phi0, c_star, betas=(2, 4, 8, 16), tol=1e-4)
            if info["err"] > 1e-2:
                continue
            phis.append(np.asarray(phi, np.float32))
            cs.append(np.asarray(info["c"], np.float32))
            ts.append(np.asarray(c_star, np.float32))
            errs.append(np.float32(info["err"]))
            tid.append(k)
        np.savez(f, phi=np.stack(phis), c=np.stack(cs), target=np.stack(ts), err=np.array(errs), target_id=np.array(tid))
        print(f"target {k}: c* {np.round(c_star, 4)}  {len(phis)} cells so far  ({(time.time() - t_start) / 60:.1f} min)", flush=True)
        k += 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-o", type=int, default=16)
    ap.add_argument("--v", type=float, default=0.25)
    ap.add_argument("--targets", type=int, default=60)
    ap.add_argument("--per-target", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data")
    main(ap.parse_args())
