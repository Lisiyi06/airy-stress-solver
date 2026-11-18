import pytest
import sympy as sp

from airy_solver.catalog import CatalogEntry, build_polynomial_catalog, print_catalog
from airy_solver.stress import is_biharmonic
from airy_solver.symbols import x, y


def test_catalog_contains_expected_degrees_and_basis_counts():
    catalog = build_polynomial_catalog(5, x, y)

    assert list(catalog) == [2, 3, 4, 5]
    assert [len(catalog[degree]) for degree in catalog] == [3, 4, 4, 4]
    assert all(isinstance(entry, CatalogEntry)
               for entries in catalog.values() for entry in entries)


def test_catalog_entries_have_consecutive_indices_and_valid_stresses():
    catalog = build_polynomial_catalog(4, x, y)

    for degree, entries in catalog.items():
        assert [entry.index for entry in entries] == list(range(1, len(entries) + 1))
        for entry in entries:
            assert entry.degree == degree
            assert is_biharmonic(entry.U, x, y)
            assert sp.simplify(entry.sigma_x - sp.diff(entry.U, y, 2)) == 0
            assert sp.simplify(entry.sigma_y - sp.diff(entry.U, x, 2)) == 0
            assert sp.simplify(entry.tau_xy + sp.diff(entry.U, x, y)) == 0


def test_catalog_does_not_include_gauge_terms():
    catalog = build_polynomial_catalog(2, x, y)
    expressions = [entry.U for entry in catalog[2]]

    assert sp.Integer(1) not in expressions
    assert x not in expressions
    assert y not in expressions
    assert all(sp.Poly(expr, x, y).total_degree() == 2 for expr in expressions)


def test_print_catalog_outputs_required_fields(capsys):
    print_catalog(build_polynomial_catalog(2, x, y))
    output = capsys.readouterr().out

    assert "degree = 2" in output
    assert "basis_index = 1" in output
    assert "U =" in output
    assert "sigma_x =" in output
    assert "sigma_y =" in output
    assert "tau_xy =" in output


def test_catalog_rejects_degree_below_stressful_range():
    with pytest.raises(ValueError):
        build_polynomial_catalog(1, x, y)
