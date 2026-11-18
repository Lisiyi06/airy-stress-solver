"""Symbolic polynomial bases for biharmonic Airy stress functions."""

import sympy as sp

from .stress import biharmonic


def _validate_degree(degree):
    if not isinstance(degree, int) or isinstance(degree, bool) or degree < 0:
        raise ValueError("degree must be a non-negative integer")


def homogeneous_monomials(degree, x, y):
    """Return all monomials whose total degree is ``degree``.

    The ordering is ``y**degree, x*y**(degree-1), ..., x**degree``.
    """
    _validate_degree(degree)
    return [x**i * y ** (degree - i) for i in range(degree + 1)]


def homogeneous_polynomial(degree, x, y, prefix="a"):
    """Construct an unknown-coefficient homogeneous polynomial."""
    monomials = homogeneous_monomials(degree, x, y)
    coefficients = sp.symbols(f"{prefix}_0:{len(monomials)}")
    expression = sum(
        coefficient * monomial
        for coefficient, monomial in zip(coefficients, monomials)
    )
    return sp.expand(expression), list(coefficients), monomials


def biharmonic_constraint_matrix(degree, x, y):
    """Build the exact matrix enforcing biharmonicity at one degree.

    For ``U = sum(c_i*m_i)``, the returned matrix ``A`` satisfies
    ``A * coefficients == 0`` exactly when ``U`` is biharmonic.
    """
    expression, coefficients, monomials = homogeneous_polynomial(degree, x, y)
    residual = sp.Poly(biharmonic(expression, x, y), x, y)

    if residual.is_zero:
        matrix = sp.zeros(0, len(coefficients))
    else:
        equations = [
            residual.coeff_monomial(monomial)
            for monomial in residual.monoms()
        ]
        matrix, _ = sp.linear_eq_to_matrix(equations, coefficients)

    return matrix, coefficients, monomials


def homogeneous_biharmonic_basis(degree, x, y):
    """Return a basis of homogeneous biharmonic polynomials."""
    matrix, _, monomials = biharmonic_constraint_matrix(degree, x, y)
    basis = []
    for vector in matrix.nullspace():
        basis.append(sp.expand(sum(component * monomial
                                   for component, monomial in zip(vector, monomials))))
    return basis


def biharmonic_basis(
    max_degree,
    x,
    y,
    min_degree=2,
    include_lower_degrees=True,
    include_gauge=False,
):
    """Return a basis through ``max_degree`` for polynomial Airy functions.

    By default, constant and linear gauge terms are omitted because they do
    not contribute to stress. Set ``include_gauge=True`` to include them.
    """
    _validate_degree(max_degree)
    _validate_degree(min_degree)
    if min_degree > max_degree and not include_gauge:
        return []

    start = min_degree if include_lower_degrees else max_degree
    start = max(start, 0 if include_gauge else 2)
    degrees = range(start, max_degree + 1) if include_lower_degrees else [max_degree]

    result = []
    for degree in degrees:
        if degree <= max_degree and (include_gauge or degree >= 2):
            result.extend(homogeneous_biharmonic_basis(degree, x, y))

    if include_gauge:
        gauge = []
        if max_degree >= 0:
            gauge.append(sp.Integer(1))
        if max_degree >= 1:
            gauge.extend([x, y])
        if include_lower_degrees or min_degree <= 1:
            result = gauge + [f for f in result if f not in gauge]

    return result


def make_unknown_airy_from_basis(basis, x, y, prefix="c"):
    """Combine basis functions with independent unknown coefficients."""
    coefficients = sp.symbols(f"{prefix}_0:{len(basis)}")
    expression = sp.expand(sum(c * B for c, B in zip(coefficients, basis)))
    return expression, list(coefficients)
