"""Stress recovery and biharmonic checks for Airy functions."""

import sympy as sp


def stress_from_airy(U, x, y):
    """Calculate Cartesian stresses from an Airy stress function.

    The convention is ``sigma_x = U_yy``, ``sigma_y = U_xx`` and
    ``tau_xy = -U_xy``.
    """
    sigma_x = sp.diff(U, y, 2)
    sigma_y = sp.diff(U, x, 2)
    tau_xy = -sp.diff(U, x, y)

    return {
        "sigma_x": sp.simplify(sp.expand(sigma_x)),
        "sigma_y": sp.simplify(sp.expand(sigma_y)),
        "tau_xy": sp.simplify(sp.expand(tau_xy)),
    }


def biharmonic(U, x, y):
    """Return ``U_xxxx + 2 U_xxyy + U_yyyy``."""
    result = (
        sp.diff(U, x, 4)
        + 2 * sp.diff(U, x, 2, y, 2)
        + sp.diff(U, y, 4)
    )
    return sp.expand(result)


def is_biharmonic(U, x, y):
    """Return whether ``U`` satisfies the biharmonic equation identically."""
    return sp.simplify(biharmonic(U, x, y)) == 0
