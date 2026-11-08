"""Shared fixtures for the tests: small symmetric cells."""
import numpy as np
import jax.numpy as jnp

from nlfiber import fiber as Fb, symmetry as S


def schwarz_p_phi(n_o, d=3):
    x = 2 * np.pi * (np.arange(n_o) + 0.5) / (2 * n_o)
    G = np.meshgrid(*[x] * d, indexing="ij")
    return jnp.asarray(-sum(np.cos(g) for g in G))


def binary_cell(n_o=4, d=3, v=0.35, seed=None):
    """A connected binary symmetric cell of side 2 n_o (Schwarz P skeleton, or a random field)."""
    from nlfiber.density import DensityMap
    phi = schwarz_p_phi(n_o, d) if seed is None else Fb.random_start(n_o, d, seed=seed, kmax=1)
    return DensityMap(n_o, d, r=1.0, v=v).binary(np.asarray(phi))


def grey_cell(n_o=4, d=3, v=0.35, beta=3.0, seed=0):
    from nlfiber.density import DensityMap
    phi = Fb.random_start(n_o, d, seed=seed, kmax=1) + 0.5 * schwarz_p_phi(n_o, d)
    return DensityMap(n_o, d, r=1.0, v=v).rho(S.sym_perm(phi), beta)
