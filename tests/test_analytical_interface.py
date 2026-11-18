"""High-order recovery, mixed conditions and text-only interface regressions."""
import pytest
import sympy as sp

from airy_solver.boundary import TractionBC, edge_traction, traction_residuals
from airy_solver.cli import interactive, parse_expression
from airy_solver.constraints import IntegralConstraint, axial_resultant, shear_resultant, bending_moment
from airy_solver.diagnostics import InconsistentBoundaryError, UnderdeterminedSystemError
from airy_solver.geometry import EDGES, Rectangle
from airy_solver.polynomial import biharmonic_basis
from airy_solver.solver import solve_linear_airy_problem, solve_polynomial_problem
from airy_solver.stress import biharmonic, stress_from_airy
from airy_solver.symbols import x, y

RECTANGLE = Rectangle(0, 2, -1, 1)


@pytest.mark.parametrize("degree", [4, 5, 6, 7, 8])
def test_recover_high_order_four_edge_problem(degree):
    # Independent harmonic construction, not manufactured from the solver basis.
    harmonic = sp.expand_complex((x + sp.I*y)**(degree - 2)).as_real_imag()[0]
    exact = sp.expand((x*x + y*y)*harmonic + x*y + y**3)
    assert biharmonic(exact, x, y) == 0
    stress = stress_from_airy(exact, x, y)
    conditions = [TractionBC(edge, *edge_traction(stress, RECTANGLE, edge, x, y)) for edge in EDGES]
    solution = solve_polynomial_problem(conditions, RECTANGLE, x, y,
                                        min_degree=degree, max_degree=degree)
    assert sp.expand(solution.U - exact) == 0
    for name in stress:
        assert sp.expand(getattr(solution, name) - stress[name]) == 0
    assert solution.diagnostics.max_symbolic_residual == 0


def test_auto_degree_search_and_insufficient_degree():
    exact = x*y**3
    stress = stress_from_airy(exact, x, y)
    conditions = [TractionBC(edge, *edge_traction(stress, RECTANGLE, edge, x, y)) for edge in EDGES]
    solution = solve_polynomial_problem(conditions, RECTANGLE, x, y, max_degree=5)
    assert solution.U == exact
    assert solution.diagnostics.highest_degree == 4
    with pytest.raises(InconsistentBoundaryError, match="No polynomial solution"):
        solve_polynomial_problem(conditions, RECTANGLE, x, y, max_degree=3)


def test_surface_tractions_plus_three_section_resultants():
    exact = sp.expand(x*(y**3 - 3*y))
    stress = stress_from_airy(exact, x, y)
    conditions = [TractionBC(edge, *edge_traction(stress, RECTANGLE, edge, x, y))
                  for edge in ("top", "bottom")]

    def constraints(candidate):
        return [IntegralConstraint(f.__name__, f(candidate, RECTANGLE, 0, x, y),
                                   f(stress, RECTANGLE, 0, x, y))
                for f in (axial_resultant, shear_resultant, bending_moment)]

    solution = solve_polynomial_problem(conditions, RECTANGLE, x, y,
                                        min_degree=4, max_degree=4, extra_constraints=constraints)
    assert solution.U == exact
    assert solution.diagnostics.n_integral_constraints == 3


def test_partial_components_and_free_parameter_diagnostic():
    conditions = [TractionBC("right", 2, None), TractionBC("top", 0, 0)]
    assert solve_polynomial_problem(conditions, RECTANGLE, x, y).sigma_x == 2
    with pytest.raises(UnderdeterminedSystemError, match=r"free parameters=\(c_2,\)"):
        solve_polynomial_problem([TractionBC("right", 2, 0)], RECTANGLE, x, y)


def test_conflicting_duplicate_conditions():
    with pytest.raises(InconsistentBoundaryError):
        solve_polynomial_problem([TractionBC("top", 0, 0), TractionBC("top", 0, 1)],
                                 RECTANGLE, x, y, max_degree=4)


def test_reject_non_biharmonic_custom_basis():
    with pytest.raises(ValueError, match="biharmonic"):
        solve_linear_airy_problem([x**4], [], RECTANGLE, x, y)


def test_gauge_terms_not_duplicated():
    basis = biharmonic_basis(4, x, y, min_degree=0, include_gauge=True)
    assert len(basis) == len(set(basis)) == 14


def test_safe_exact_expression_parser():
    assert parse_expression("0.1*x^2 + 1/3") == x**2/10 + sp.Rational(1, 3)
    assert parse_expression("y") is y
    for text in ("sin(x)", "__import__('os')", "x.real", "1/0", "x**100", "c_0"):
        with pytest.raises(ValueError):
            parse_expression(text)


def test_interactive_fourth_degree_solution(monkeypatch, capsys):
    # U=x*y**3 implies tau=-3*y**2, independent of x.
    lines = iter(["", "left,r,,3*y**2", "right,r,6*L*y,-3*y**2",
                  "top,r,3*h**2/4,0", "bottom,r,-3*h**2/4,0", "done"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(lines))
    interactive()
    output = capsys.readouterr().out
    assert "U = x*y**3" in output
    assert "双调和残差 = 0" in output
    assert "平衡方程残差 = (0, 0)" in output