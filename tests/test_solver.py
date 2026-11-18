import pytest
import sympy as sp

from airy_solver.boundary import TractionBC
from airy_solver.constraints import IntegralConstraint
from airy_solver.diagnostics import (
    InconsistentBoundaryError,
    UnderdeterminedSystemError,
)
from airy_solver.geometry import BOTTOM, LEFT, RIGHT, TOP, Rectangle
from airy_solver.polynomial import homogeneous_biharmonic_basis
from airy_solver.solver import assemble_linear_system, solve_linear_airy_problem
from airy_solver.symbols import x, y


RECTANGLE = Rectangle(0, 1, -1, 1)


def test_assemble_linear_system_uses_exact_symbolic_arithmetic():
    a, b = sp.symbols("a b")
    matrix, rhs = assemble_linear_system([2 * a + b - 3, a - b], [a, b])

    assert matrix == sp.Matrix([[2, 1], [1, -1]])
    assert rhs == sp.Matrix([3, 0])


def test_solve_unique_constant_sigma_x_problem():
    basis = homogeneous_biharmonic_basis(2, x, y)
    conditions = [
        TractionBC(RIGHT, 2, 0),
        TractionBC(LEFT, -2, 0),
        TractionBC(TOP, 0, 0),
        TractionBC(BOTTOM, 0, 0),
    ]

    solution = solve_linear_airy_problem(
        basis, conditions, RECTANGLE, x, y
    )

    assert solution.diagnostics.status == "unique"
    assert solution.diagnostics.matrix_rank == solution.diagnostics.n_unknowns
    assert solution.sigma_x == 2
    assert solution.sigma_y == 0
    assert solution.tau_xy == 0
    assert solution.diagnostics.max_symbolic_residual == 0


def test_extra_integral_constraint_is_added_to_linear_system():
    basis = homogeneous_biharmonic_basis(2, x, y)
    conditions = [
        TractionBC(RIGHT, 2, 0),
        TractionBC(LEFT, -2, 0),
    ]
    # The user may provide any explicitly derived resultant or couple
    # equation. This independent equation fixes the remaining c_2.
    c_2 = sp.Symbol("c_2")
    constraint = IntegralConstraint("user_applied_end_couple", c_2, 1)

    solution = solve_linear_airy_problem(
        basis, conditions, RECTANGLE, x, y, extra_constraints=[constraint]
    )

    assert solution.diagnostics.n_integral_constraints == 1
    assert solution.sigma_x == 2


def test_inconsistent_boundary_conditions_raise():
    basis = homogeneous_biharmonic_basis(2, x, y)
    conditions = [
        TractionBC(RIGHT, 1, 0),
        TractionBC(RIGHT, 2, 0),
        TractionBC(TOP, 0, 0),
        TractionBC(BOTTOM, 0, 0),
    ]

    with pytest.raises(InconsistentBoundaryError):
        solve_linear_airy_problem(basis, conditions, RECTANGLE, x, y)


def test_underdetermined_system_raises_with_diagnostic_exception():
    basis = homogeneous_biharmonic_basis(2, x, y)
    conditions = [TractionBC(RIGHT, 2, 0)]

    with pytest.raises(UnderdeterminedSystemError):
        solve_linear_airy_problem(basis, conditions, RECTANGLE, x, y)
