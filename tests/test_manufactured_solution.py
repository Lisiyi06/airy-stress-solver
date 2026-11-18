import sympy as sp

from airy_solver.boundary import TractionBC, edge_traction, traction_residuals
from airy_solver.geometry import EDGES, Rectangle
from airy_solver.polynomial import biharmonic_basis
from airy_solver.solver import solve_linear_airy_problem
from airy_solver.stress import biharmonic, is_biharmonic, stress_from_airy
from airy_solver.symbols import x, y


RECTANGLE = Rectangle(0, 2, -1, 1)


def manufactured_airy():
    a, b, c, d, e = sp.symbols("a b c d e")
    return sp.expand((
        a * y**3 + b * x * y**3 + c * x**3 * y + d * x * y + e * x**3
    ).subs({a: 2, b: -sp.Rational(3, 2), c: sp.Rational(5, 4),
            d: sp.Rational(7, 3), e: -sp.Rational(1, 2)}))


def boundary_conditions_from_stress(stress):
    return [
        TractionBC(edge, *edge_traction(stress, RECTANGLE, edge, x, y))
        for edge in EDGES
    ]


def test_manufactured_airy_function_is_biharmonic():
    exact_U = manufactured_airy()
    residual = sp.simplify(biharmonic(exact_U, x, y))

    assert residual == 0
    assert is_biharmonic(exact_U, x, y)


def test_manufactured_boundary_conditions_are_exactly_reproduced():
    exact_stress = stress_from_airy(manufactured_airy(), x, y)

    conditions = boundary_conditions_from_stress(exact_stress)
    for condition in conditions:
        assert traction_residuals(
            exact_stress, RECTANGLE, condition, x, y
        ) == (0, 0)


def test_solver_recovers_manufactured_stress_from_generated_tractions():
    exact_stress = stress_from_airy(manufactured_airy(), x, y)
    solution = solve_linear_airy_problem(
        biharmonic_basis(5, x, y),
        boundary_conditions_from_stress(exact_stress),
        RECTANGLE,
        x,
        y,
    )

    assert solution.diagnostics.status == "unique"
    assert solution.diagnostics.max_symbolic_residual == 0
    for name in ("sigma_x", "sigma_y", "tau_xy"):
        assert sp.simplify(getattr(solution, name) - exact_stress[name]) == 0


def test_stress_comparison_ignores_airy_gauge_terms():
    exact_U = manufactured_airy()
    gauged_U = exact_U + 11 - 4 * x + 9 * y
    exact_stress = stress_from_airy(exact_U, x, y)
    gauged_stress = stress_from_airy(gauged_U, x, y)

    assert sp.simplify(gauged_U - exact_U) != 0
    for name in ("sigma_x", "sigma_y", "tau_xy"):
        assert sp.simplify(gauged_stress[name] - exact_stress[name]) == 0