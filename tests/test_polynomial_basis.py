import sympy as sp

from airy_solver.polynomial import (
    biharmonic_basis,
    homogeneous_monomials,
    homogeneous_polynomial,
    make_unknown_airy_from_basis,
)
from airy_solver.stress import is_biharmonic
from airy_solver.symbols import x, y


def test_homogeneous_monomials_are_in_expected_order():
    assert homogeneous_monomials(3, x, y) == [y**3, x * y**2, x**2 * y, x**3]


def test_homogeneous_polynomial_returns_coefficients_and_monomials():
    expression, coefficients, monomials = homogeneous_polynomial(2, x, y)

    assert coefficients == list(sp.symbols("a_0:3"))
    assert monomials == [y**2, x * y, x**2]
    assert expression == sum(c * m for c, m in zip(coefficients, monomials))


def test_biharmonic_basis_excludes_gauge_by_default():
    basis = biharmonic_basis(5, x, y)

    assert len(basis) == 3 + 4 + 4 + 4
    assert all(sp.Poly(expr, x, y).total_degree() >= 2 for expr in basis)
    assert all(is_biharmonic(expr, x, y) for expr in basis)


def test_biharmonic_basis_can_include_gauge_terms():
    basis = biharmonic_basis(2, x, y, include_gauge=True)

    assert 1 in basis
    assert x in basis
    assert y in basis
    assert len(basis) == 6


def test_make_unknown_airy_from_basis():
    basis = biharmonic_basis(2, x, y)
    expression, coefficients = make_unknown_airy_from_basis(basis, x, y, prefix="c")

    assert coefficients == list(sp.symbols("c_0:3"))
    assert expression == sum(c * B for c, B in zip(coefficients, basis))
