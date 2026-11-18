"""Ready-to-run symbolic beam examples."""

import sympy as sp

from .beam import EndLoad, solve_beam
from .boundary import TractionBC
from .geometry import BOTTOM, TOP, L


def point_load_problem(F=None):
    """Right-supported cantilever with a downward force at x=0."""
    F = sp.Symbol("F", real=True) if F is None else sp.sympify(F)
    tractions = [TractionBC(TOP, 0, 0), TractionBC(BOTTOM, 0, 0)]
    loads = [EndLoad("left", 0, F, 0), EndLoad("right", 0, -F, F * L)]
    return tractions, loads, 4


def cantilever_uniform_problem(q=None):
    """Cantilever with a downward uniform load on the top edge."""
    q = sp.Symbol("q", real=True) if q is None else sp.sympify(q)
    tractions = [TractionBC(TOP, 0, q), TractionBC(BOTTOM, 0, 0)]
    loads = [EndLoad("right", 0, 0, 0),
             EndLoad("left", 0, -q * L, -q * L**2 / 2)]
    return tractions, loads, 5


def simply_supported_uniform_problem(q=None):
    """Simply supported beam with a downward uniform top load."""
    q = sp.Symbol("q", real=True) if q is None else sp.sympify(q)
    tractions = [TractionBC(TOP, 0, q), TractionBC(BOTTOM, 0, 0)]
    reaction = -q * L / 2
    loads = [EndLoad("left", 0, reaction, 0),
             EndLoad("right", 0, reaction, 0)]
    return tractions, loads, 5


def cantilever_point_load(F=None):
    tractions, loads, degree = point_load_problem(F)
    return solve_beam(tractions, loads, max_degree=degree)


def cantilever_uniform_load(q=None):
    tractions, loads, degree = cantilever_uniform_problem(q)
    return solve_beam(tractions, loads, max_degree=degree)


def simply_supported_uniform_load(q=None):
    tractions, loads, degree = simply_supported_uniform_problem(q)
    return solve_beam(tractions, loads, max_degree=degree)


def get_case_problem(name):
    """Return the actual input data shared by the solver and report."""
    factories = {
        cantilever_point_load: point_load_problem,
        cantilever_uniform_load: cantilever_uniform_problem,
        simply_supported_uniform_load: simply_supported_uniform_problem,
    }
    return factories[CASES[str(name).strip().lower()]]()


CASES = {
    "1": cantilever_point_load,
    "2": cantilever_uniform_load,
    "3": simply_supported_uniform_load,
    "cantilever-point": cantilever_point_load,
    "cantilever-uniform": cantilever_uniform_load,
    "simply-supported-uniform": simply_supported_uniform_load,
}


def solve_case(name, **parameters):
    """Run a case by number or name, e.g. ``solve_case("3", q=10)``."""
    try:
        factory = CASES[str(name).strip().lower()]
    except KeyError as exc:
        raise ValueError("案例只能是 1、2、3，或对应的英文名称") from exc
    return factory(**parameters)


__all__ = ["CASES", "cantilever_point_load", "cantilever_uniform_load",
           "simply_supported_uniform_load", "solve_case"]