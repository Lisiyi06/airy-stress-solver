"""Exact linear solving for polynomial Airy traction problems."""

import sympy as sp

from .boundary import (
    polynomial_coefficient_equations,
    traction_residuals,
)
from .constraints import constraint_equations
from .diagnostics import (
    AirySolution,
    InconsistentBoundaryError,
    SolveDiagnostics,
    UnderdeterminedSystemError,
)
from .stress import stress_from_airy, is_biharmonic


def assemble_linear_system(equations, unknowns):
    """Return the exact matrix ``A`` and vector ``b`` for expressions == 0."""
    equations = [sp.expand(sp.sympify(equation)) for equation in equations]
    matrix, vector = sp.linear_eq_to_matrix(equations, unknowns)
    return matrix, vector


def _boundary_equations(stress, rectangle, boundary_conditions, x, y):
    equations = []
    for condition in boundary_conditions:
        residual_tx, residual_ty = traction_residuals(
            stress, rectangle, condition, x, y
        )
        _, tangent = _edge_coordinates(rectangle, condition.edge, x, y)
        equations.extend(polynomial_coefficient_equations(residual_tx, tangent))
        equations.extend(polynomial_coefficient_equations(residual_ty, tangent))
    return equations


def _edge_coordinates(rectangle, edge, x, y):
    from .boundary import _edge_value

    return _edge_value(rectangle, edge, x, y)


def _max_residual(equations, substitutions):
    residuals = [sp.simplify(expr.subs(substitutions)) for expr in equations]
    nonzero = [expr for expr in residuals if expr != 0]
    return max(nonzero, key=lambda expr: len(str(expr))) if nonzero else sp.Integer(0)


def solve_linear_airy_problem(
    basis,
    boundary_conditions,
    rectangle,
    x,
    y,
    coefficient_prefix="c",
    extra_constraints=None,
):
    """Solve for coefficients of a polynomial Airy basis using traction BCs."""
    basis = list(basis)
    if not basis or any(not sp.sympify(f).is_polynomial(x, y)
                        or not is_biharmonic(f, x, y) for f in basis):
        raise ValueError("basis must contain biharmonic polynomials")
    unknowns = list(sp.symbols(f"{coefficient_prefix}_0:{len(basis)}"))
    U = sp.expand(sum(coefficient * function
                      for coefficient, function in zip(unknowns, basis)))
    stress = stress_from_airy(U, x, y)
    equations = _boundary_equations(
        stress, rectangle, boundary_conditions, x, y
    )
    n_boundary_equations = len(equations)
    # A factory avoids requiring callers to know internal coefficient ordering.
    integral_constraints = list(
        extra_constraints(stress) if callable(extra_constraints)
        else (extra_constraints or [])
    )
    equations.extend(constraint_equations(integral_constraints))
    matrix, rhs = assemble_linear_system(equations, unknowns)
    augmented = matrix.row_join(rhs)
    matrix_rank = matrix.rank()
    augmented_rank = augmented.rank()
    n_unknowns = len(unknowns)

    if matrix_rank < augmented_rank:
        raise InconsistentBoundaryError(
            f"inconsistent boundary system: rank={matrix_rank}, "
            f"augmented_rank={augmented_rank}"
        )
    if matrix_rank < n_unknowns:
        pivots = matrix.rref()[1]
        free = tuple(symbol for i, symbol in enumerate(unknowns) if i not in pivots)
        raise UnderdeterminedSystemError(
            f"underdetermined boundary system: rank={matrix_rank}, "
            f"unknowns={n_unknowns}; free parameters={free}"
        )

    solution_set = sp.linsolve((matrix, rhs), unknowns)
    solution_tuple = next(iter(solution_set))
    substitutions = dict(zip(unknowns, solution_tuple))
    solved_U = sp.expand(U.subs(substitutions))
    solved_stress = {
        key: sp.simplify(value.subs(substitutions))
        for key, value in stress.items()
    }
    residual = _max_residual(equations, substitutions)
    if residual != 0 or not is_biharmonic(solved_U, x, y):
        raise ValueError("symbolic verification failed; no solution returned")
    diagnostics = SolveDiagnostics(
        status="unique",
        n_unknowns=n_unknowns,
        n_boundary_equations=n_boundary_equations,
        matrix_rank=matrix_rank,
        augmented_rank=augmented_rank,
        free_parameters=(),
        max_symbolic_residual=residual,
        highest_degree=max(
            (sp.Poly(function, x, y).total_degree() for function in basis),
            default=None,
        ),
        n_basis_functions=len(basis),
        n_integral_constraints=len(integral_constraints),
    )
    return AirySolution(
        U=solved_U,
        sigma_x=solved_stress["sigma_x"],
        sigma_y=solved_stress["sigma_y"],
        tau_xy=solved_stress["tau_xy"],
        coefficients=substitutions,
        diagnostics=diagnostics,
    )


def solve_polynomial_problem(boundary_conditions, rectangle, x, y,
                             max_degree=8, min_degree=2,
                             extra_constraints=None):
    """Find the first uniquely matched polynomial within a bounded search.

    Uniqueness refers to the chosen polynomial space, not to an unspecified
    continuum boundary-value problem. An underdetermined space is reported
    immediately: adding more basis functions cannot remove its nullspace.
    Constraint factories receive the current symbolic stress dictionary.
    """
    from .polynomial import biharmonic_basis

    if any(not isinstance(n, int) or isinstance(n, bool)
           for n in (min_degree, max_degree)) or not 2 <= min_degree <= max_degree:
        raise ValueError("require integer degrees: 2 <= min_degree <= max_degree")
    conditions = list(boundary_conditions)
    attempts = []
    for degree in range(min_degree, max_degree + 1):
        try:
            return solve_linear_airy_problem(
                biharmonic_basis(degree, x, y), conditions, rectangle, x, y,
                extra_constraints=extra_constraints,
            )
        except InconsistentBoundaryError as exc:
            attempts.append(f"degree {degree}: {exc}")
    raise InconsistentBoundaryError(
        f"No polynomial solution found for degrees {min_degree}..{max_degree}. "
        "Conditions may conflict or require a higher/non-polynomial solution.\n"
        + "\n".join(attempts)
    )
