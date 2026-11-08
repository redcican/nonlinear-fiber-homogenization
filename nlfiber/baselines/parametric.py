"""B7 parametric sweep with projection in parameter space, and B8 random cells."""
import numpy as np
import jax.numpy as jnp
from scipy import optimize

from .. import symmetry as S
from ..prior import data as Dd
from ..fiber import rel_error


def parametric_field(theta, n, d, kind="tpms"):
    """A two-parameter family: mixing a of two TPMS (P and I-WP) and a sheet/skeletal blend b."""
    a, b = theta
    X = Dd._coords(n, d)
    c = [np.cos(2 * np.pi * x) for x in X]
    P = sum(c)
    if d == 3:
        c2 = [np.cos(4 * np.pi * x) for x in X]
        IWP = 2 * (c[0] * c[1] + c[1] * c[2] + c[2] * c[0]) - sum(c2)
    else:
        IWP = c[0] * c[1]
    f = (1 - a) * P / np.abs(P).max() + a * IWP / np.abs(IWP).max()
    return (1 - b) * f - b * np.abs(f)


def b7_parametric(fiber, target, n_eval=60, seed=0, up=2):
    """Nelder-Mead on (a, b) in [0, 1]^2 of the projected-stiffness error of the binary cell at volume v;
    the volume is exact by thresholding. Returns dict(theta, err_bin, binary)."""
    n, d = 2 * fiber.n_o, fiber.d
    best = dict(err_bin=np.inf)

    def f(theta):
        theta = np.clip(theta, 0, 1)
        fld = Dd._sym(parametric_field(theta, n * up, d))
        b = Dd.binary_from_field(fld, fiber.v)
        if b is None:
            return 10.0
        lh = fiber.lh_for(b.shape)
        c, _, _ = lh.invariants(jnp.asarray(b, float))
        fiber.solves += 1
        e = rel_error(np.asarray(c), target)
        if e < best["err_bin"]:
            best.update(err_bin=e, theta=theta.copy(), binary=dict(c=np.asarray(c), vol=float(b.mean()), cell=b))
        return e

    rng = np.random.default_rng(seed)
    for x0 in rng.uniform(0, 1, (3, 2)):
        optimize.minimize(f, x0, method="Nelder-Mead", options=dict(maxfev=n_eval // 3, xatol=1e-3, fatol=1e-5))
    return best


def b8_random(fiber, target, count, eps=0.05, seed=0, families=Dd.FAMILIES):
    """Random procedural cells at the fiber's volume accepted within relative error eps."""
    rng = np.random.default_rng(seed)
    n, d = 2 * fiber.n_o, fiber.d
    acc, tried = [], 0
    while tried < count:
        fam = families[tried % len(families)]
        tried += 1
        fld = Dd._sym(Dd.GEN[fam](n, d, rng))
        b = Dd.binary_from_field(fld, fiber.v)
        if b is None:
            continue
        c, _, _ = fiber.lh_for(b.shape).invariants(jnp.asarray(b, float))
        fiber.solves += 1
        e = rel_error(np.asarray(c), target)
        if e < eps:
            acc.append(dict(cell=b, c=np.asarray(c), err=e, family=fam))
    return acc, tried
