import sympy as sp

from airy_solver.stress import biharmonic, is_biharmonic, stress_from_airy
from airy_solver.symbols import x, y


def test_constant_stress():
    U = sp.Rational(1, 2) * y**2
    stress = stress_from_airy(U, x, y)

    assert sp.simplify(stress["sigma_x"] - 1) == 0
    assert sp.simplify(stress["sigma_y"]) == 0
    assert sp.simplify(stress["tau_xy"]) == 0


def test_pure_shear():
    U = -x * y
    stress = stress_from_airy(U, x, y)

    assert sp.simplify(stress["sigma_x"]) == 0
    assert sp.simplify(stress["sigma_y"]) == 0
    assert sp.simplify(stress["tau_xy"] - 1) == 0


def test_pure_bending():
    U = y**3
    stress = stress_from_airy(U, x, y)

    assert sp.simplify(stress["sigma_x"] - 6 * y) == 0
    assert sp.simplify(stress["sigma_y"]) == 0
    assert sp.simplify(stress["tau_xy"]) == 0


def test_degree_less_than_four_is_biharmonic():
    for U in [x**2, x * y, y**2, x**3, x**2 * y, x * y**2, y**3]:
        assert is_biharmonic(U, x, y)


def test_biharmonic_operator_is_zero_for_constant_stress_function():
    U = x**2 + 2 * x * y + y**2
    assert sp.simplify(biharmonic(U, x, y)) == 0
