"""End-to-end pilot of the fiber-width measurement (the G1 protocol of the plan at a small size).

For each target: random symmetric starts are projected onto the stiffness target, binarized at the design
resolution and accepted at C^H error <= 0.5 %; each accepted cell follows the uniaxial compression path L2
with the cell stability criterion; the QoIs nu(20%), r(20%), the first load maximum, the kink strain and the
first loss of cell stability are collected. The noise floor of the G1 amendment is measured on the first four
accepted cells: (a) path re-runs with another kick and with half the load step, (b) re-evaluation at 1.5x
resolution (change of pairwise differences), (c) a tighter stiffness slack. H1 and H1s are decided per target
and with the 50 % quantifier.

    python -I scripts/width_pilot.py --n-o 16 --targets 2 --starts 16 --accept 10 --out runs/pilot32
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax.numpy as jnp  # noqa: E402

from nlfiber import fiber as Fb, project as P, topology as T, mech as Me, qoi as Q, widths as W  # noqa: E402
from nlfiber import linear as L  # noqa: E402
from nlfiber.density import upsample_periodic  # noqa: E402

QOIS = ("nu20", "r20", "eps_max", "eps_kink", "eps_cell", "S20", "reached")


BACKEND = {"name": "jax", "fs3d": None, "path": "compression"}


def evaluate(cell_bin, emax, de, kick_seed=0, stability=True, verbose=False):
    """Compression path of a cell; returns the QoI dict and the per-step rows. With the fs3d back end the
    kick is the solver's own (deterministic), so kick re-runs change only the load step."""
    rho = jnp.asarray(cell_bin, jnp.float64)
    if BACKEND["fs3d"] is not None:  # small-strain moduli from the GPU solver (the JAX solve of a 128^3 cell exceeds the card)
        C, _ = BACKEND["fs3d"].moduli(np.asarray(rho), tag=f"moduli_{kick_seed}_{rho.shape[0]}")
    else:
        C, _, _ = L.LinearHomogenization(rho.shape).C_full(rho)
        C = np.asarray(C)
    E0 = Q.young_x(C)
    if BACKEND["fs3d"] is not None:
        rows = BACKEND["fs3d"].run_path(np.asarray(rho), BACKEND["path"], emax=emax, de=de, crit="1" if stability else "",
                                        tag=f"path_{kick_seed}_{de:g}")
    else:
        cell = Me.Cell(rho)
        rows = Q.run_path(cell, BACKEND["path"], emax=emax, de=de, seed=kick_seed, stability=stability, verbose=verbose)
    q = Q.path_qois(rows, E0, at=(0.1, 0.2))
    q["nu0"] = Q.nu0(C)
    q["E0"] = float(E0)
    q["newton_per_step"] = float(np.mean([r["newton"] for r in rows]))
    q["seconds"] = float(sum(r["seconds"] for r in rows))
    return q, rows


def resample_binary(fib, phi, factor):
    """Binary cell of the same design at `factor` times the resolution (threshold at the same volume)."""
    u = np.asarray(fib.dm.smooth(jnp.asarray(phi)))
    u = upsample_periodic(u, factor)
    from nlfiber import symmetry as S
    u = np.asarray(S.exact_symmetrize(jnp.asarray(u)))
    b = u > np.quantile(u, 1 - fib.v)
    return T.remove_floating(b)


def resample_grey(fib, phi, factor, beta):
    """Grey design of the same pre-density at `factor` times the resolution (same threshold eta)."""
    u = np.asarray(fib.dm.smooth(jnp.asarray(phi)))
    eta = fib.dm.eta(jnp.asarray(phi), beta)
    u = upsample_periodic(u, factor)
    return 1.0 / (1.0 + np.exp(-beta * (u - eta)))


def run(args):
    out = Path(args["out"])
    out.mkdir(parents=True, exist_ok=True)
    n_o, v, r = args["n_o"], args["v"], args["r"]
    emax, de = args["emax"], args["de"]
    fib = Fb.Fiber(n_o, v=v, r=r)
    betas = (2, 4, 8, 16)
    BACKEND["path"] = args.get("path", "compression")
    if args.get("backend") == "fs3d":
        from nlfiber.backend_fs3d import FS3D
        BACKEND["fs3d"] = FS3D(workdir=out / "fs3d")
        BACKEND["name"] = "fs3d"
    log = open(out / "log.txt", "a")

    def say(*a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    # targets: the stiffness of reference designs (random starts whose binary cell spans the period)
    targets, seed_t = [], 1000 + 100 * args.get("target_offset", 0)
    if args.get("target_file"):
        # reuse the targets of earlier runs (e.g. G1), so that a re-run with another filter radius keeps c*
        for f in args["target_file"]:
            tg = json.loads(Path(f).read_text())["results"][0]["target"]
            targets.append(tg)
            say(f"target {tg['id']}: c = {np.asarray(tg['c'])} (from {f})")
    while len(targets) < args["targets"]:
        phi_t = Fb.random_start(n_o, seed=seed_t, kmax=2)
        seed_t += 1
        chk = fib.binary_check(phi_t, up=1)
        val = T.validity(chk["cell"])
        if abs(chk["vol"] - v) > 0.02 or val["components"] != 1 or val["spanning"] != 1:
            continue
        if args["eval"] == "grey":
            c_t = fib.value(phi_t, betas[-1])
        else:
            c_t = chk["c"]
        k = len(targets) + args.get("target_offset", 0)
        targets.append(dict(id=k, c=[float(x) for x in c_t], vol=chk["vol"], seed=seed_t - 1))
        say(f"target {k}: c = {np.asarray(c_t)} ({args['eval']} reference, vol {chk['vol']:.3f})")

    results = []
    for tg in targets:
        t_id, c_star = tg["id"], np.array(tg["c"])
        accepted = []
        for seed in range(args["starts"]):
            if len(accepted) >= args["accept"]:
                break
            t0 = time.time()
            phi0 = Fb.random_start(n_o, seed=seed, kmax=2)
            try:
                phi, info = P.project_to_fiber(fib, phi0, c_star, betas=betas, up=1, polish_steps=6)
            except Exception as e:  # noqa: BLE001
                say(f"  t{t_id} s{seed}: projection failed: {e}")
                continue
            b = info["binary"]["cell"]
            val = T.validity(b)
            connected = val["components"] == 1 and val["spanning"] == 1
            if args["eval"] == "grey":
                err = info["err_grey"]
                ok = err <= 1e-3 and connected
                design = np.asarray(fib.rho(phi, betas[-1]))
            else:
                err = info["err_bin"]
                ok = err <= 0.005 and connected
                design = b
            if args.get("require_valid"):
                ok = ok and val["valid"]  # all validity rules of plan Sec. 5.4, minimum feature size included
            say(f"  t{t_id} s{seed}: grey {info['err_grey']:.1e} bin {info['err_bin']:.2e} "
                f"vol {info['binary']['vol']:.3f} valid {val['valid']} {'ACCEPT' if ok else 'reject'} "
                f"({info['solves']} solves, {time.time() - t0:.0f}s)")
            if not ok:
                continue
            q, rows = evaluate(design, emax, de, kick_seed=seed)
            say(f"     path: reached {q['reached']:.2f} nu20 {q.get('nu20', np.nan):+.3f} r20 {q.get('r20', np.nan):.3f} "
                f"eps_max {q['eps_max']:.3f} eps_cell {q.get('eps_cell', np.nan):.3f} newton/step {q['newton_per_step']:.1f} "
                f"{q['seconds']:.0f}s")
            accepted.append(dict(seed=seed, phi=np.asarray(phi), cell=b, design=design, err_bin=info["err_bin"],
                                 err_grey=info["err_grey"], qoi=q,
                                 path=[dict(e=r["e"], P11=float(r["P"][0, 0]), nu=float(np.mean(r["nu"])),
                                            lam=float(r["lam"][0]) if "lam" in r else np.nan) for r in rows]))
        cells = [a["cell"] for a in accepted]
        res = dict(target=tg, n_accepted=len(accepted), mean_iou=T.mean_pairwise_iou(cells) if len(cells) > 1 else np.nan,
                   samples=[dict(seed=a["seed"], err_bin=a["err_bin"], err_grey=a["err_grey"], qoi=a["qoi"], path=a["path"])
                            for a in accepted])
        np.savez(out / f"target{t_id}_cells.npz", **{f"cell{a['seed']}": a["cell"] for a in accepted},
                 **{f"phi{a['seed']}": a["phi"] for a in accepted})

        # noise floor on the first four accepted cells
        floor = {k: dict(a=[], b=[], c=[]) for k in ("nu20", "r20", "eps_max", "eps_cell")}
        base = accepted[:4]
        if base:
            # (a) path re-runs: another kick, half step
            for a in base:
                for tag, kw in (("kick", dict(kick_seed=a["seed"] + 777, de=de)), ("half", dict(kick_seed=a["seed"], de=de / 2))):
                    q2, _ = evaluate(a["design"], emax, kw["de"], kick_seed=kw["kick_seed"])
                    for k in floor:
                        floor[k]["a"].append((a["qoi"].get(k, np.nan), q2.get(k, np.nan)))
                    say(f"  floor(a) t{t_id} s{a['seed']} {tag}: nu20 {q2.get('nu20', np.nan):+.3f} r20 {q2.get('r20', np.nan):.3f} eps_max {q2['eps_max']:.3f}")
            # (b) finer resolution on even targets: 1.5x (the G1 protocol); 2x with fs3d, which needs powers of two
            if t_id % 2 == 0:
                fine = []
                fac = 2.0 if BACKEND["name"] == "fs3d" else 1.5
                for a in base:
                    b15 = resample_binary(fib, a["phi"], fac) if args["eval"] == "binary" else resample_grey(fib, a["phi"], fac, betas[-1])
                    q3, _ = evaluate(b15, emax, de, kick_seed=a["seed"])
                    fine.append(q3)
                    say(f"  floor(b) t{t_id} s{a['seed']} {fac}x ({b15.shape[0]}^3): nu20 {q3.get('nu20', np.nan):+.3f} r20 {q3.get('r20', np.nan):.3f} eps_max {q3['eps_max']:.3f}")
                for k in floor:
                    floor[k]["b"] = [[a["qoi"].get(k, np.nan) for a in base], [q.get(k, np.nan) for q in fine]]
            # (c) tighter stiffness slack on two cells
            for a in base[:2]:
                if args["eval"] == "binary":
                    phi2, pol = P.polish_binary(fib, jnp.asarray(a["phi"]), c_star, betas[-1], up=1, steps=8, tol_bin=5e-4)
                    d2, slack = pol["binary"]["cell"], pol["err_bin"]
                else:
                    phi2, pinfo = P.gn_project(fib, jnp.asarray(a["phi"]), c_star, betas[-1], tol=1e-6)
                    d2, slack = np.asarray(fib.rho(phi2, betas[-1])), pinfo["err"]
                q4, _ = evaluate(d2, emax, de, kick_seed=a["seed"])
                for k in floor:
                    floor[k]["c"].append((a["qoi"].get(k, np.nan), q4.get(k, np.nan)))
                say(f"  floor(c) t{t_id} s{a['seed']} slack {slack:.1e}: nu20 {q4.get('nu20', np.nan):+.3f} r20 {q4.get('r20', np.nan):.3f} eps_max {q4['eps_max']:.3f}")

        def comp(k):
            fa = W.path_floor([x for x, _ in floor[k]["a"]], [[y for _, y in floor[k]["a"]]]) if floor[k]["a"] else np.nan
            fb = W.difference_floor(*floor[k]["b"]) if floor[k]["b"] else np.nan
            fc = W.path_floor([x for x, _ in floor[k]["c"]], [[y for _, y in floor[k]["c"]]]) if floor[k]["c"] else np.nan
            return dict(a=fa, b=fb, c=fc, floor=W.target_floor(fa, fb, fc))

        res["floor"] = {k: comp(k) for k in floor}
        le = [(x, y) for x, y in floor["eps_max"]["a"] + floor["eps_max"]["c"] if np.isfinite(x) and np.isfinite(y)]
        res["floor"]["ln_eps"] = W.log_floor([x for x, _ in le], [[y for _, y in le]]) if le else np.nan
        nu20 = [a["qoi"].get("nu20", np.nan) for a in accepted]
        r20 = [a["qoi"].get("r20", np.nan) for a in accepted]
        eps = [a["qoi"]["eps_max"] for a in accepted]
        res["decision"] = W.h1_target(nu20, eps, r20, res["floor"]["nu20"]["floor"],
                                      res["floor"]["ln_eps"] if np.isfinite(res["floor"]["ln_eps"]) else 0.0,
                                      res["floor"]["r20"]["floor"], n_min=args["n_min"])
        say(f"target {t_id}: n {len(accepted)} IoU {res['mean_iou']:.2f} nu20 spread {res['decision']['nu_spread']:.3f} "
            f"(floor {res['floor']['nu20']['floor']:.3f}) r20 spread {res['decision']['r_spread']:.3f} "
            f"(floor {res['floor']['r20']['floor']:.3f}) eps ratio {res['decision']['eps_ratio']:.2f} "
            f"H1 {res['decision']['H1']} H1s {res['decision']['H1s']}")
        results.append(res)
        (out / "results.json").write_text(json.dumps(dict(args=args, targets=targets, results=results), default=float, indent=1))

    summary = dict(H1=W.quantifier([r["decision"] for r in results], "H1"),
                   H1s=W.quantifier([r["decision"] for r in results], "H1s"))
    say("summary:", json.dumps(summary))
    (out / "results.json").write_text(json.dumps(dict(args=args, targets=targets, results=results, summary=summary),
                                                 default=float, indent=1))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-o", type=int, default=8)
    ap.add_argument("--v", type=float, default=0.25)
    ap.add_argument("--r", type=float, default=1.5)
    ap.add_argument("--targets", type=int, default=2)
    ap.add_argument("--target-offset", type=int, default=0, help="first target id (shards the targets over GPUs)")
    ap.add_argument("--starts", type=int, default=12)
    ap.add_argument("--accept", type=int, default=10)
    ap.add_argument("--n-min", type=int, default=10)
    ap.add_argument("--emax", type=float, default=0.2)
    ap.add_argument("--de", type=float, default=0.01)
    ap.add_argument("--eval", choices=("binary", "grey"), default="binary",
                    help="evaluate the binary cell (accepted at 0.5 %% C^H error) or the grey design (1e-3)")
    ap.add_argument("--path", choices=("compression", "tension"), default="compression")
    ap.add_argument("--backend", choices=("jax", "fs3d"), default="jax",
                    help="finite-strain evaluator: the JAX FP64 solver or fs3d (NLFIBER_FS3D)")
    ap.add_argument("--target-file", nargs="*", default=None, help="results.json files whose targets are reused")
    ap.add_argument("--require-valid", action="store_true", help="accept only cells that pass every validity rule")
    ap.add_argument("--out", default="runs/pilot32")
    a = ap.parse_args()
    run(dict(n_o=a.n_o, v=a.v, r=a.r, targets=a.targets, starts=a.starts, accept=a.accept, n_min=a.n_min,
             emax=a.emax, de=a.de, out=a.out, eval=a.eval, backend=a.backend, target_offset=a.target_offset, path=a.path))
