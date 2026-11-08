"""Sampling and programming the set of periodic cells that share one linear stiffness tensor.

Modules:
  symmetry   fundamental-domain representation of cubic (3D) and square (2D) symmetric cells
  density    filter, smoothed Heaviside, exact-volume threshold, binarization
  linear     small-strain homogenization with exact sensitivities (JAX)
  topology   periodic connectivity, validity checks, diversity measures
  fiber      the constraint map phi -> normalized stiffness invariants, with Jacobian rows
  project    Gauss-Newton projection onto the fiber in a chosen metric, binary polish
  metric     projection metrics: Euclidean, Gaussian-Tweedie (DCT), learned Tweedie
  mech       finite-strain periodic homogenization (JAX, FP64): mixed control, adjoints, stability, TOECs
  qoi        load paths and nonlinear quantities of interest
  widths     fiber-width statistics, noise floor, hypothesis decisions, distribution distances
  prior      procedural data, 3D U-Net, flow matching, deterministic generator
  sampler    S1 projected flow, S2 constrained-manifold MCMC (ZHG), ABC arbiter
  steer      null-space gradient flow and tilted SMC on the fiber
  amortize   conditional generator for nonlinear targets
  baselines  B1-B8
  runner     file-lock work queue sharded over GPUs
"""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

__version__ = "0.1.0"
