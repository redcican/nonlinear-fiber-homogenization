## Nonlinear Fiber of Homogenization

This repository contains the code for sampling and programming the set of periodic cells that share one linear stiffness tensor (the *fiber* of linear homogenization), and for measuring how much large-strain behaviour that set hides:

* the fiber as a constraint map: symmetric pre-density on the fundamental domain, exact volume by thresholding, exact cubic (3D) or square (2D) symmetry by construction, and the normalized stiffness invariants with their Jacobian rows (no adjoint solve);
* small-strain periodic homogenization on voxel grids (JAX), with exact sensitivities and an adjoint-differentiable corrector solve;
* Gauss-Newton projection onto the fiber in a chosen metric (Euclidean, Gaussian-Tweedie in the DCT basis, learned Tweedie), sharpness continuation and binary polish against a re-solve at a finer grid;
* a finite-strain periodic evaluator (JAX, FP64): neo-Hookean solid with void interpolation, mixed macroscopic control, load paths, adjoint gradients, stability criteria and third-order elastic constants;
* fiber-width statistics, noise floors and the hypothesis decisions of the study;
* a geometry prior (procedural symmetric cells, 3D U-Net, flow matching) and its deterministic generator;
* exact samplers of the fiber: the projected flow S1 (Gauss-Newton in the Gaussian-Tweedie or learned Tweedie metric) and the constrained-manifold MCMC S2 in the generator's latent space (Zappa, Holmes-Cerfon & Goodman, with the co-area Gram factor);
* programming on the fiber: null-space gradient flow with finite-strain adjoint losses and tilted sequential Monte Carlo; an amortized conditional generator;
* the baselines B1–B8 and a file-lock work queue sharded over GPUs.

The finite-strain GPU solver `fs3d` and the GPU solver for inverse homogenization of the companion repository [periodic-finite-strain-homogenization](https://github.com/redcican/periodic-finite-strain-homogenization) can serve as faster back ends.

### Repository layout

| Path | Content |
|------|---------|
| `nlfiber/symmetry.py` | fundamental-domain representation, mirror and fold maps, exact symmetrization |
| `nlfiber/density.py` | periodic filter, smoothed Heaviside, exact-volume threshold, binarization |
| `nlfiber/linear.py` | small-strain homogenization (matrix-free PCG with an FFT preconditioner) |
| `nlfiber/fiber.py` | the constraint map $c(\phi)$ and its Jacobian rows, binary re-solve |
| `nlfiber/metric.py` | projection metrics |
| `nlfiber/project.py` | Gauss-Newton projection, continuation, binary polish |
| `nlfiber/topology.py` | periodic connectivity, validity checks, topological signature, IoU |
| `nlfiber/mech.py` | finite-strain evaluator |
| `nlfiber/qoi.py` | load paths L1–L4 and nonlinear quantities of interest |
| `nlfiber/widths.py` | spreads, bootstrap intervals, noise floors, hypothesis decisions, distribution distances |
| `nlfiber/prior/` | procedural data (`data.py`), U-Net (`unet.py`), flow-matching prior and generator (`flow.py`) |
| `nlfiber/sampler/` | S1 projected flow (`s1.py`), latent fiber (`latent.py`), manifold MCMC (`zhg.py`), ABC arbiter (`abc.py`) |
| `nlfiber/steer/` | null-space flow (`nullspace.py`), tilted SMC (`smc.py`), finite-strain losses with adjoints (`losses.py`) |
| `nlfiber/amortize.py` | conditional generator and the distillation gap |
| `nlfiber/baselines/` | B1 multi-start, B2 deflation, B4 DPS guidance, B6 values-only SMC, B7 parametric, B8 random |
| `nlfiber/runner.py` | work queue sharded over GPUs |
| `tests/` | unit tests |

### Dependency

* JAX with FP64 and CUDA 12
* NumPy, SciPy

Create the conda environment by:

```bash
conda env create -f environment.yml
conda activate nlfiber
```

### Usage

Project random starting cells onto the stiffness of a Schwarz P cell at volume fraction 0.3 (a $32^3$ cell, binary check at $64^3$):

```python
import numpy as np
from nlfiber import fiber, project

fib = fiber.Fiber(n_o=16, v=0.3, r=1.5)                 # 16^3 octant -> 32^3 cell
x = 2 * np.pi * (np.arange(16) + 0.5) / 32
X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
target = fib.value(-(np.cos(X) + np.cos(Y) + np.cos(Z)), beta=16.0)  # (C11, C12, C44) / E

phi0 = fiber.random_start(16, seed=0, kmax=2)
phi, info = project.project_to_fiber(fib, phi0, target, betas=(2, 4, 8, 16), up=2)
print(info["err_grey_before_polish"], info["err_bin"], info["binary"]["vol"])
```

`beta` multiplies the filtered pre-density, whose standard deviation is about one; the default continuation is 2, 4, 8, 16.

Finite-strain path of a cell under uniaxial compression with the cell stability criterion:

```python
from nlfiber import mech, qoi
cell = mech.Cell(rho)                        # rho: (n, n, n) density in [0, 1]
rows = qoi.run_path(cell, "compression", emax=0.2, de=0.01, stability=True)
print(qoi.path_qois(rows, E0=qoi.young_x(C)))
```

Sample the fiber with a learned prior (S1), then correct the measure with the latent manifold MCMC (S2):

```python
import jax
from nlfiber.prior import data, flow
from nlfiber.sampler import s1, latent, zhg
from nlfiber import metric, symmetry

X, fam, vol = data.make_dataset(n_o=16, count=2000)        # procedural symmetric cells -> pre-densities
prior = flow.FlowPrior(16, widths=(16, 32, 64))
prior.fit(X, steps=20000, batch=16)
spec = metric.spectrum_from_data(prior.to_model(X))        # Gaussian-Tweedie metric of the data

phi, info = s1.s1_sample(prior, fib, target, prior.latent(jax.random.PRNGKey(0)), K=16, spectrum=spec)

lat = latent.LatentFiber(prior, fib, target, beta=16.0, K=8)
z, _ = lat.project(prior.latent(jax.random.PRNGKey(1)))
chain, step = zhg.ZHG(lat.cj, sym=symmetry.sym_perm).run(z, 200, 0.1, adapt=100)
```

Run a work queue on both cards:

```bash
python -m nlfiber.runner runs/queue --shard 0 --n-shards 2 --gpu 0 &
python -m nlfiber.runner runs/queue --shard 1 --n-shards 2 --gpu 1 &
```

### Tests

```bash
python -m unittest discover -s tests -t .
```

`NLFIBER_REFERENCE=<dir of oracle.py>` adds the comparisons with the FP64 reference solver of the companion repository (`reference/jax`).
