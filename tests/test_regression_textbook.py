"""Regression tests for the textbook-style sixth-degree Airy polynomial."""

import sympy as sp

from airy_solver.boundary import TractionBC, edge_traction, traction_residuals
from airy_solver.geometry import EDGES, Rectangle
from airy_solver.polynomial import biharmonic_basis
from airy_solver.solver import solve_linear_airy_problem
from airy_solver.stress import biharmonic, stress_from_airy
from airy_solver.symbols import x, y


RECTANGLE = Rectangle(0, 2, -1, 1)


def textbook_airy_with_symbols():
    """Construct the textbook polynomial before imposing its constraint."""
    A, B, C, D, E, F = sp.symbols("A B C D E F")
    U = (
        A * x**3 * y**3
        + B * x * y**5
        + C * x**3 * y
        + D * x * y**3
        + E * x**3
        + F * x * y
    )
    return sp.expand(U), (A, B, C, D, E, F)


def textbook_boundary_conditions(stress):
    return [
        TractionBC(edge, *edge_traction(stress, RECTANGLE, edge, x, y))
        for edge in EDGES
    ]


def test_textbook_sixth_degree_biharmonic_relation_is_derived_symbolically():
    U, (A, B, *_rest) = textbook_airy_with_symbols()
    residual = sp.Poly(biharmonic(U, x, y), x, y)

    assert residual.as_expr() == (72 * A + 120 * B) * x * y
    assert sp.simplify(
        residual.coeff_monomial(x * y) - (72 * A + 120 * B)
    ) == 0


def test_textbook_coefficients_satisfying_relation_are_biharmonic():
    U, coefficients = textbook_airy_with_symbols()
    A, B, C, D, E, F = coefficients
    textbook_U = U.subs({
        A: 1,
        B: -sp.Rational(3, 5),
        C: 2,
        D: -1,
        E: sp.Rational(1, 2),
        F: sp.Rational(7, 3),
    })

    assert 72 * 1 + 120 * (-sp.Rational(3, 5)) == 0
    assert biharmonic(textbook_U, x, y) == 0


def test_textbook_stress_components_are_generated_from_airy_function():
    U, coefficients = textbook_airy_with_symbols()
    A, B, C, D, E, F = coefficients
    textbook_U = U.subs({
        A: 1,
        B: -sp.Rational(3, 5),
        C: 2,
        D: -1,
        E: sp.Rational(1, 2),
        F: sp.Rational(7, 3),
    })
    stress = stress_from_airy(textbook_U, x, y)

    expected = {
        "sigma_x": 6 * x**3 * y - 12 * x * y**3 - 6 * x * y,
        "sigma_y": 6 * x * y**3 + 12 * x * y + 3 * x,
        "tau_xy": -9 * x**2 * y**2 + 3 * y**4 - 6 * x**2 + 3 * y**2 - sp.Rational(7, 3),
    }
    for name in expected:
        assert sp.simplify(stress[name] - expected[name]) == 0


def test_textbook_boundary_problem_regression_has_zero_residuals():
    U, coefficients = textbook_airy_with_symbols()
    A, B, C, D, E, F = coefficients
    textbook_U = U.subs({
        A: 1,
        B: -sp.Rational(3, 5),
        C: 2,
        D: -1,
        E: sp.Rational(1, 2),
        F: sp.Rational(7, 3),
    })
    exact_stress = stress_from_airy(textbook_U, x, y)
    conditions = textbook_boundary_conditions(exact_stress)

    solution = solve_linear_airy_problem(
        biharmonic_basis(6, x, y), conditions, RECTANGLE, x, y
    )

    assert solution.diagnostics.status == "unique"
    assert solution.diagnostics.max_symbolic_residual == 0
    for condition in conditions:
        assert traction_residuals(
            {
                "sigma_x": solution.sigma_x,
                "sigma_y": solution.sigma_y,
                "tau_xy": solution.tau_xy,
            },
            RECTANGLE,
            condition,
            x,
            y,
        ) == (0, 0)
    for name in ("sigma_x", "sigma_y", "tau_xy"):
        assert sp.simplify(getattr(solution, name) - exact_stress[name]) == 0