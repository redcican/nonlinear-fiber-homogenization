"""The finite-strain GPU solver fs3d of the companion repository as an evaluator back end.

fs3d reads a density r on an n^3 grid (n = 16, 32, 64, ...) and uses the stiffness s = 1e-9 + r^3 and the
void interpolation gamma(r^3). To evaluate the same solid as `mech.Cell` (s = emin + (1 - emin) rho^p) we pass
r = s^(1/3); the two codes then share s to 1e-9 and gamma to its value at the void floor.

The results are returned in the formats of `qoi.run_path` (rows) and `mech.toec` (C, T), so the width
statistics and the steering losses can use either evaluator. Paths use the fs3d CSV columns documented in the
companion README. Set the binary with NLFIBER_FS3D=<path to build/fs3d>.
"""
import csv
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .mech import CUBIC_TOEC

PATHS = {"tension": "L1", "compression": "L2", "shear": "L3", "vol+": "L4"}


class FS3D:
    def __init__(self, binary=None, E=1.0, nu=0.3, emin=1e-3, p=3.0, workdir=None, gpu=None, prec="adaptive",
                 newton_tol=1e-6):
        self.bin = binary or os.environ.get("NLFIBER_FS3D")
        if not self.bin or not Path(self.bin).exists():
            raise FileNotFoundError("fs3d binary not found; set NLFIBER_FS3D")
        self.E, self.nu, self.emin, self.p = E, nu, emin, p
        self.work = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="fs3d_", dir=os.getcwd()))
        self.work.mkdir(parents=True, exist_ok=True)
        self.env = dict(os.environ)
        if gpu is not None:
            self.env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        self.prec, self.newton_tol = prec, newton_tol
        self.calls = 0

    def _density_file(self, rho, tag):
        rho = np.clip(np.asarray(rho, np.float64), 0, 1)
        if rho.ndim == 2:
            raise ValueError("fs3d needs a 3D cell")
        n = rho.shape[0]
        if rho.shape != (n, n, n) or n & (n - 1) or n < 16:
            raise ValueError("fs3d needs an n^3 cell with n = 16, 32, 64, ...")
        s = self.emin + (1 - self.emin) * rho**self.p
        r = np.cbrt(s)
        path = self.work / f"{tag}.npy"
        np.save(path, r.astype(np.float64))
        return path, n

    def _run(self, args, tag):
        out = self.work / tag
        out.mkdir(exist_ok=True)
        cmd = [self.bin, *args, "-E", str(self.E), "-nu", str(self.nu), "-prefix", str(out) + "/", "-verbose", "0"]
        self.calls += 1
        res = subprocess.run(cmd, env=self.env, capture_output=True, text=True)
        (out / "stdout.txt").write_text(res.stdout + "\n" + res.stderr)
        return out, res.returncode

    # --- small-strain moduli and third-order constants ---------------------------------------
    def moduli(self, rho, tag="toec"):
        """(C 6x6, T 6x6x6) of the cell; T in the Lagrangian strain, Voigt order xx yy zz yz xz xy."""
        f, n = self._density_file(rho, tag)
        out, rc = self._run(["-mode", "toec", "-in", str(f), "-prec", self.prec], tag)
        lines = (out / "moduli.txt").read_text().splitlines()
        C = np.array([[float(x) for x in lines[i].split()] for i in range(1, 7)])
        T = np.zeros((6, 6, 6))
        for ln in lines[8:]:
            if not ln.strip():
                continue
            name, val = ln.split()
            I, J, K = (int(ch) - 1 for ch in name[1:4])
            for pidx in {(I, J, K), (I, K, J), (J, I, K), (J, K, I), (K, I, J), (K, J, I)}:
                T[pidx] = float(val)
        return C, T

    def cubic_toec(self, T):
        return {k: float(np.mean([T[i] for i in v])) for k, v in CUBIC_TOEC.items()}

    # --- load paths ---------------------------------------------------------------------------
    def run_path(self, rho, path="compression", emax=0.2, de=0.01, crit="1", stop_distortion=True, tag="path",
                 truncate_unstable=True, branch=False, branch_kick=None):
        """Rows in the format of qoi.run_path: e, P (3x3), F, nu (lateral), converged, newton, lam, seconds.
        truncate_unstable: end the path at the first load point whose cell criterion is non-positive.
        branch: let fs3d switch to the stable branch at a loss of stability (fs3d -branch); the rows then carry
        'branch' (switches so far) and are not truncated. branch_kick: first kick amplitude (fs3d default 0.01)."""
        f, n = self._density_file(rho, tag)
        steps = int(round(emax / de))
        args = ["-mode", "path", "-in", str(f), "-path", PATHS[path], "-steps", str(steps), "-inc", str(de),
                "-crit", crit, "-prec", self.prec, "-newton_tol", str(self.newton_tol)]
        if not stop_distortion:
            args.append("-nostop_distortion")
        if branch:
            args.append("-branch")
            if branch_kick is not None:
                args += ["-branch_kick", str(branch_kick)]
            truncate_unstable = False
        out, rc = self._run(args, tag)
        rows = []
        csvf = out / f"path_{PATHS[path]}.csv"
        if not csvf.exists():
            return rows
        with open(csvf) as fh:
            for r in csv.DictReader(fh):
                g = lambda k: float(r[k]) if r.get(k, "") not in ("", "nan") else np.nan  # noqa: E731
                F = np.eye(3)
                F[0, 0], F[1, 1], F[2, 2], F[0, 1] = g("F11"), g("F22"), g("F33"), g("F12")
                P = np.array([[g("P11"), g("P12"), g("P13")], [g("P21"), g("P22"), g("P23")], [g("P31"), g("P32"), g("P33")]])
                e = g("eps")
                row = dict(path=path, e=e, P=P, F=F, converged=True, newton=g("newton"), cg=g("krylov"),
                           splits=g("substeps"), seconds=g("seconds"), negative_curvature=bool(g("minres") > 0))
                if path in ("tension", "compression"):
                    row["nu"] = [-(F[1, 1] - 1) / e, -(F[2, 2] - 1) / e]
                lam = g("lambda_cell")
                if np.isfinite(lam):
                    row["lam"] = np.array([lam])
                if np.isfinite(g("lambda_supercell")):
                    row["lam_supercell"] = g("lambda_supercell")
                if np.isfinite(g("ellipticity")):
                    row["rank_one"] = g("ellipticity")
                row["void_inverted"] = g("void_distorted")
                if "branch" in r:
                    row["branch"] = g("branch")
                rows.append(row)
        if rc == 2 and rows:  # the path stopped early: mark the end
            rows[-1]["failed_after"] = True
        if truncate_unstable:
            # fs3d follows the fundamental branch past an instability (MINRES on the indefinite tangent); the
            # stable response there is the buckled branch, which the FP64 evaluator reaches with a kick. We keep
            # the first load point with lambda_cell <= 0 (it fixes the onset by interpolation) and drop the rest.
            for i, r in enumerate(rows):
                if "lam" in r and r["lam"][0] <= 0:
                    rows = rows[: i + 1]
                    rows[-1]["unstable"] = True
                    break
        return rows

    # --- adjoint gradient ---------------------------------------------------------------------
    def grad(self, rho, obj="P11", path="tension", e=0.1, steps=5, tag="grad"):
        """dJ/drho (J = P11 or nu12 at strain e along the path) as an array of the cell's shape.
        fs3d differentiates with respect to its own density r; the chain rule r = s^(1/3) converts to rho."""
        f, n = self._density_file(rho, tag)
        inc = e / steps
        gfile = self.work / f"{tag}_grad.npy"
        out, rc = self._run(["-mode", "fdcheck", "-in", str(f), "-path", PATHS[path], "-steps", str(steps), "-inc",
                             str(inc), "-obj", obj, "-prec", "fp32", "-fd_n", "0", "-fd_out", str(gfile)], tag)
        g_r = np.load(gfile, allow_pickle=False)
        rho = np.clip(np.asarray(rho, np.float64), 0, 1)
        s = self.emin + (1 - self.emin) * rho**self.p
        dr_drho = (1.0 / 3.0) * s ** (-2.0 / 3.0) * (1 - self.emin) * self.p * rho ** (self.p - 1)
        return g_r * dr_drho
