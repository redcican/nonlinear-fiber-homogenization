"""Baselines of the plan (Sec. 5.3).

  B1  multi-start equality-constrained voxel optimization: a random linear functional (sampling) or a
      nonlinear loss (programming) minimized by the null-space flow from random starts (the voxel analogue of
      Panetta et al. 2017 with an exact C^H constraint)
  B2  deflation multi-start (Papadopoulos, Farrell & Surowiec 2021): B1 with a penalty on found designs
  B4  DPS-style soft guidance through the flow: the endpoint moves along the Euclidean gradient of |h|^2 with a
      strength s, then (optionally) the exact final projection
  B5  PCFM Euclidean projection: sampler.s1.s1_sample(metric='euclid')
  B6  values-only SMC / Feynman-Kac steering with the prior (no Jacobian, no manifold moves): tempered prior
      samples weighted by exp(-T |h|^2), followed by projection
  B7  TPMS parametric sweep with projection in parameter space (mixing and sheet/skeletal parameters)
  B8  random: unconditional prior samples accepted within eps (sampler.abc), random procedural cells
"""
from .multistart import b1_multistart, b2_deflation  # noqa: F401
from .guidance import b4_dps, b6_values_only  # noqa: F401
from .parametric import b7_parametric, b8_random  # noqa: F401
