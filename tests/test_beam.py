import pytest
import sympy as sp

from airy_solver.beam import EndLoad, solve_beam, end_constraints
from airy_solver.boundary import TractionBC, edge_traction
from airy_solver.cli import interactive, parse_expression
from airy_solver.geometry import default_beam, L, h
from airy_solver.stress import biharmonic, stress_from_airy
from airy_solver.symbols import x, y


def test_symbolic_geometry_and_normals():
    rect = default_beam()
    assert rect.top_y == -h/2
    assert rect.bottom_y == h/2
    assert parse_expression("L/h") == L/h
    stress = dict(sigma_x=1, sigma_y=2, tau_xy=3)
    assert edge_traction(stress, rect, "top", x, y) == (-3, -2)
    assert edge_traction(stress, rect, "bottom", x, y) == (3, 2)


def test_end_force_fourth_degree_solution():
    P = sp.Symbol("P", real=True)
    conditions = [TractionBC("top", 0, 0), TractionBC("bottom", 0, 0)]
    loads = [EndLoad("right", 0, P, 0), EndLoad("left", 0, -P, -P*L)]
    solution = solve_beam(conditions, loads, max_degree=4)
    assert solution.diagnostics.highest_degree == 4
    assert sp.simplify(solution.tau_xy - 3*P/(2*h)*(1-4*y*y/h**2)) == 0
    assert sp.simplify(solution.sigma_x - 12*P*(x-L)*y/h**3) == 0
    assert biharmonic(solution.U, x, y) == 0
    stress = stress_from_airy(solution.U, x, y)
    assert all(sp.simplify(c.expression-c.target) == 0
               for c in end_constraints(stress, default_beam(), loads))


def test_end_couple_third_degree_solution():
    M = sp.Symbol("M", real=True)
    solution = solve_beam([TractionBC("top", 0, 0), TractionBC("bottom", 0, 0)],
                          [EndLoad("right", 0, 0, M), EndLoad("left", 0, 0, -M)],
                          max_degree=3)
    assert sp.simplify(solution.sigma_x + 12*M*y/h**3) == 0
    assert solution.tau_xy == 0


def test_uniform_surface_load_requires_fifth_degree():
    q = sp.Symbol("q", real=True)
    solution = solve_beam([TractionBC("top", 0, q), TractionBC("bottom", 0, 0)],
                          [EndLoad("right", 0, 0, 0)], max_degree=5)
    assert solution.diagnostics.highest_degree == 5
    stress = stress_from_airy(solution.U, x, y)
    assert tuple(sp.simplify(v) for v in edge_traction(stress, default_beam(), "top", x, y)) == (0, q)


def test_interactive_end_force(monkeypatch, capsys):
    lines = iter(["4", "top,r,0,0", "bottom,r,0,0",
                  "right,t,0,P,0", "done"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(lines))
    interactive()
    output = capsys.readouterr().out
    assert "highest_degree=4" in output
    assert "right.Fy 积分残差 = 0" in output


def test_surface_resultant_rejected(monkeypatch):
    lines = iter(["4", "top,t,0,P,0"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(lines))
    with pytest.raises(ValueError, match="上下表面禁止"):
        interactive()