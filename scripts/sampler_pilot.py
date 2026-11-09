"""Small pilot of the exact samplers: a procedural prior at a 16^3 cell, S1 in the Gaussian-Tweedie metric
against the Euclidean projection B5 and the random baseline B8, then the latent manifold MCMC S2.

    python -I scripts/sampler_pilot.py --n-o 8 --cells 400 --steps 3000 --samples 8 --out runs/sampler16
Reports exactness (grey and binary C^H error), validity, diversity (mean pairwise IoU), solver calls and time per
accepted design, and for S2 the acceptance rate, reverse-check failures and the on-fiber residual.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from nlfiber import fiber as Fb, metric as M, topology as T, symmetry as S  # noqa: E402
from nlfiber.prior import data as Dd, flow as Fl  # noqa: E402
from nlfiber.sampler import s1, latent, zhg, abc  # noqa: E402
from nlfiber.fiber import rel_error  # noqa: E402


def summarize(rows, fib):
    cells = [r["binary"]["cell"] for r in rows]
    return dict(n=len(rows), err_grey=float(np.median([r["err_grey_before_polish"] for r in rows])),
                err_bin_median=float(np.median([r["err_bin"] for r in rows])),
                success_05=float(np.mean([r["err_bin"] <= 0.005 for r in rows])),
                valid=float(np.mean([T.validity(c)["valid"] for c in cells])),
                iou=T.mean_pairwise_iou(cells), solves=float(np.mean([r["solves"] for r in rows])),
                seconds=float(np.mean([r["seconds"] for r in rows])))


def run(args):
    out = Path(args["out"])
    out.mkdir(parents=True, exist_ok=True)
    n_o = args["n_o"]
    log = open(out / "log.txt", "a")

    def say(*a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    t0 = time.time()
    X, fam, vol = Dd.make_dataset(n_o, args["cells"], v_range=(0.15, 0.35), seed=0)
    say(f"data: {X.shape} families {np.bincount(fam).tolist()} ({time.time() - t0:.0f}s)")
    held = None
    if args.get("fiber_data"):
        # cells on fibers (scripts/make_fiber_dataset.py); the last target of each file is held out as the test target
        parts, held = [], []
        for f in args["fiber_data"]:
            d = np.load(f, allow_pickle=False)
            last = d["target_id"].max()
            keep = d["target_id"] != last
            parts.append(np.asarray(d["phi"][keep], np.float32))
            held.append(d["target"][~keep][0])
        Xf = np.concatenate(parts)
        X = np.concatenate([X, Xf])
        say(f"fiber cells: {Xf.shape[0]} added ({len(held)} targets held out); data now {X.shape}")
    prior = Fl.FlowPrior(n_o, widths=(16, 32, 64), seed=0)
    t0 = time.time()
    hist = prior.fit(X, steps=args["steps"], batch=16, lr=3e-4, log_every=500)
    say(f"prior: {prior.nparams()} parameters, loss {hist[:100].mean():.3f} -> {hist[-200:].mean():.3f} ({time.time() - t0:.0f}s)")
    prior.save(out / "prior.npz")
    spec = M.spectrum_from_data(prior.to_model(jnp.asarray(X)))

    fib = Fb.Fiber(n_o, v=args["v"], r=1.5)
    if held:
        target = np.asarray(held[0], np.float64)
        say(f"target c = {target} (held-out fiber target)")
    else:
        chk = fib.binary_check(Fb.random_start(n_o, seed=1000, kmax=2), up=1)
        target = chk["c"]
        say(f"target c = {target} vol {chk['vol']:.3f}")

    results = {}
    for name, metric in (("S1", "gaussian"), ("B5", "euclid")):
        rows = []
        for i in range(args["samples"]):
            z = prior.latent(jax.random.PRNGKey(i))
            phi, info = s1.s1_sample(prior, fib, target, z, K=16, t0=0.4, metric=metric, spectrum=spec,
                                     betas=(2, 4, 8, 16), up=1, polish_steps=6)
            rows.append(info)
            say(f"  {name} {i}: grey {info['err_grey_before_polish']:.1e} bin {info['err_bin']:.2e} solves {info['solves']} {info['seconds']:.0f}s")
        results[name] = summarize(rows, fib)
        np.savez(out / f"{name}_cells.npz", **{f"cell{i}": r["binary"]["cell"] for i, r in enumerate(rows)})
        say(name, json.dumps(results[name]))

    # B8: prior samples accepted by stiffness (ABC)
    t0 = time.time()
    zs = prior.latent(jax.random.PRNGKey(99), args["abc"])
    phis = prior.to_phi(jax.jit(lambda z: prior.generate(z, 16))(zs))
    cs = np.array([fib.value(p.astype(jnp.float64), 16.0) for p in phis])
    acc = abc.abc_accept(cs, target, eps_list=(0.05, 0.02, 0.01))
    results["B8"] = {str(e): int(len(i)) for e, i in acc.items()}
    results["B8"]["seconds"] = time.time() - t0
    say("B8 accepted of", args["abc"], json.dumps(results["B8"]))

    # S2: latent chains from projected prior samples
    lat = latent.LatentFiber(prior, fib, target, beta=16.0, K=8)
    chains, stats = [], []
    for i in range(args["chains"]):
        z, info = lat.project(prior.latent(jax.random.PRNGKey(200 + i)), tol=1e-7)
        if not info["converged"]:
            say(f"  S2 chain {i}: latent projection failed ({info['res_inf']:.1e})")
            continue
        sampler = zhg.ZHG(lat.cj, sym=S.sym_perm, newton_tol=1e-8)
        t0 = time.time()
        chain, step = sampler.run(z, args["mcmc"], 0.1, seed=i, adapt=args["mcmc"] // 2)
        h, _ = lat.cj(chain[-1])
        cells = [fib.dm.binary(np.asarray(lat.phi_of(c)), up=1) for c in chain[len(chain) // 2::max(1, len(chain) // 10)]]
        st = dict(acceptance=sampler.stats["accepted"] / sampler.stats["steps"], fwd_fail=sampler.stats["fwd_fail"],
                  rev_fail=sampler.stats["rev_fail"], res=float(np.abs(h).max()), step=float(step),
                  iou_along_chain=T.mean_pairwise_iou(cells), iou_start_end=T.iou(cells[0], cells[-1]),
                  seconds=time.time() - t0, calls=lat.calls)
        stats.append(st)
        chains.append(chain)
        say(f"  S2 chain {i}: {json.dumps(st)}")
    results["S2"] = stats
    (out / "results.json").write_text(json.dumps(dict(args=args, target=[float(x) for x in target], results=results),
                                                 default=float, indent=1))
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-o", type=int, default=8)
    ap.add_argument("--v", type=float, default=0.25)
    ap.add_argument("--cells", type=int, default=400)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--abc", type=int, default=200)
    ap.add_argument("--chains", type=int, default=3)
    ap.add_argument("--mcmc", type=int, default=40)
    ap.add_argument("--fiber-data", nargs="*", default=None, help="fiber_<n_o>.npz files of make_fiber_dataset.py")
    ap.add_argument("--out", default="runs/sampler16")
    a = ap.parse_args()
    run(vars(a))
