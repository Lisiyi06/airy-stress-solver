"""Shared symbolic coordinates for the Airy stress-function solver."""

import sympy as sp

x, y = sp.symbols("x y", real=True)


def default_symbols():
    """Return the default real Cartesian coordinates ``(x, y)``."""
    return x, y
