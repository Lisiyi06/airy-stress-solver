import pytest
import sympy as sp

from airy_solver.boundary import (
    NonPolynomialBoundaryError,
    TractionBC,
    edge_traction,
    polynomial_coefficient_equations,
    traction_residuals,
)
from airy_solver.geometry import BOTTOM, LEFT, RIGHT, TOP, Rectangle
from airy_solver.stress import stress_from_airy
from airy_solver.symbols import x, y


RECTANGLE = Rectangle(0, 10, -1, 1)


def test_rectangle_dimensions_preserve_symbolic_values():
    L, h = sp.symbols("L h", positive=True)
    rectangle = Rectangle(0, L, -h / 2, h / 2)

    assert rectangle.length == L
    assert rectangle.height == h


def test_edge_tractions_have_correct_outward_signs():
    stress = {"sigma_x": 2, "sigma_y": 3, "tau_xy": 5}

    assert edge_traction(stress, RECTANGLE, LEFT, x, y) == (-2, -5)
    assert edge_traction(stress, RECTANGLE, RIGHT, x, y) == (2, 5)
    assert edge_traction(stress, RECTANGLE, BOTTOM, x, y) == (-5, -3)
    assert edge_traction(stress, RECTANGLE, TOP, x, y) == (5, 3)


def test_edge_traction_substitutes_fixed_coordinate():
    stress = {"sigma_x": x + y, "sigma_y": x - y, "tau_xy": x * y}

    tx, ty = edge_traction(stress, RECTANGLE, RIGHT, x, y)
    assert tx == y + 10
    assert ty == 10 * y


def test_traction_residuals_substitute_target_boundary_coordinates():
    U = sp.Rational(1, 2) * y**2
    stress = stress_from_airy(U, x, y)
    bc = TractionBC(edge=RIGHT, tx=1, ty=0)

    assert traction_residuals(stress, RECTANGLE, bc, x, y) == (0, 0)


def test_polynomial_coefficient_equations():
    a, b, c, d = sp.symbols("a b c d")
    equations = polynomial_coefficient_equations(
        (2 * a + 3 * b) * x**2 + (c - 1) * x + d,
        x,
    )

    assert equations == [2 * a + 3 * b, c - 1, d]


def test_non_polynomial_boundary_is_rejected():
    with pytest.raises(NonPolynomialBoundaryError):
        polynomial_coefficient_equations(sp.sin(x), x)


def test_non_polynomial_traction_residual_is_rejected():
    stress = {"sigma_x": sp.sin(y), "sigma_y": 0, "tau_xy": 0}
    bc = TractionBC(edge=RIGHT, tx=0, ty=0)

    with pytest.raises(NonPolynomialBoundaryError):
        traction_residuals(stress, RECTANGLE, bc, x, y)
