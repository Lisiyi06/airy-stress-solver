"""Catalogues of homogeneous polynomial Airy stress states."""

from dataclasses import dataclass

import sympy as sp

from .polynomial import homogeneous_biharmonic_basis
from .stress import stress_from_airy


@dataclass(frozen=True)
class CatalogEntry:
    """One biharmonic polynomial Airy basis state and its stresses."""

    degree: int
    index: int
    U: object
    sigma_x: object
    sigma_y: object
    tau_xy: object


def build_polynomial_catalog(max_degree, x, y):
    """Build a degree-indexed catalogue of polynomial Airy stress states.

    Constant and linear gauge terms are intentionally omitted. The catalogue
    therefore starts at degree two, where Airy functions first produce stress.
    """
    if (
        not isinstance(max_degree, int)
        or isinstance(max_degree, bool)
        or max_degree < 2
    ):
        raise ValueError("max_degree must be an integer >= 2")

    catalog = {}
    for degree in range(2, max_degree + 1):
        entries = []
        for index, U in enumerate(homogeneous_biharmonic_basis(degree, x, y), start=1):
            stress = stress_from_airy(U, x, y)
            entries.append(
                CatalogEntry(
                    degree=degree,
                    index=index,
                    U=sp.expand(U),
                    sigma_x=stress["sigma_x"],
                    sigma_y=stress["sigma_y"],
                    tau_xy=stress["tau_xy"],
                )
            )
        catalog[degree] = entries
    return catalog


def print_catalog(catalog):
    """Pretty-print catalogue entries to standard output."""
    for degree in sorted(catalog):
        for entry in catalog[degree]:
            print(f"degree = {entry.degree}")
            print(f"basis_index = {entry.index}")
            print(f"U = {sp.pretty(entry.U)}")
            print(f"sigma_x = {sp.pretty(entry.sigma_x)}")
            print(f"sigma_y = {sp.pretty(entry.sigma_y)}")
            print(f"tau_xy = {sp.pretty(entry.tau_xy)}")
            print()
