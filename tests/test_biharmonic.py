import sympy as sp

from airy_solver.polynomial import (
    biharmonic_constraint_matrix,
    homogeneous_biharmonic_basis,
)
from airy_solver.stress import is_biharmonic
from airy_solver.symbols import x, y


def test_degree_two_basis_is_biharmonic():
    basis = homogeneous_biharmonic_basis(2, x, y)

    assert len(basis) == 3
    for expr in basis:
        assert is_biharmonic(expr, x, y)


def test_degree_three_basis_is_biharmonic():
    basis = homogeneous_biharmonic_basis(3, x, y)

    assert len(basis) == 4
    for expr in basis:
        assert is_biharmonic(expr, x, y)


def test_higher_degree_basis_is_biharmonic():
    for degree in (4, 5):
        basis = homogeneous_biharmonic_basis(degree, x, y)
        # In two dimensions, the homogeneous biharmonic space has
        # dimension four for every degree n >= 4.
        assert len(basis) == 4
        assert all(is_biharmonic(expr, x, y) for expr in basis)


def test_low_degree_constraint_matrix_has_no_rows():
    matrix, coefficients, monomials = biharmonic_constraint_matrix(3, x, y)

    assert matrix.shape == (0, 4)
    assert len(coefficients) == len(monomials) == 4
    assert matrix == sp.zeros(0, 4)
