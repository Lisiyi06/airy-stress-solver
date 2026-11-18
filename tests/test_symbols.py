import sympy as sp

from airy_solver.symbols import default_symbols, x, y


def test_default_symbols():
    assert default_symbols() == (x, y)
    assert x.is_real
    assert y.is_real


def test_symbols_are_distinct_real_coordinates():
    assert x != y
    assert sp.Symbol("x", real=True) == x
    assert sp.Symbol("y", real=True) == y
