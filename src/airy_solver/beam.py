"""Symbolic beam model with exact tractions and Saint-Venant end loads."""
from dataclasses import dataclass
import sympy as sp

from .boundary import edge_traction
from .constraints import IntegralConstraint
from .geometry import default_beam
from .solver import solve_polynomial_problem
from .symbols import x, y


@dataclass(frozen=True)
class EndLoad:
    """External end resultants; positive M is clockwise for y downward.

    M is taken about the end midpoint. A point force at offset y0 adds
    -y0*Fx to M. None leaves a resultant unspecified; zero prescribes zero.
    """
    edge: str
    Fx: object = 0
    Fy: object = 0
    M: object = 0


def end_constraints(stress, rectangle, loads):
    constraints = []
    for load in loads:
        if load.edge not in ("left", "right"):
            raise ValueError("集中力/力偶仅允许在左右端面，以圣维南结果量匹配")
        tx, ty = edge_traction(stress, rectangle, load.edge, x, y)
        for name, expression, target in (
            ("Fx", tx, load.Fx), ("Fy", ty, load.Fy), ("M", -y*tx, load.M)
        ):
            if target is not None:
                target = sp.sympify(target)
                if target.has(x, y):
                    raise ValueError("端面合力和力偶不能含坐标 x、y")
                constraints.append(IntegralConstraint(
                    f"{load.edge}.{name}",
                    sp.integrate(expression, (y, rectangle.y_min, rectangle.y_max)), target,
                ))
    return constraints


def solve_beam(tractions, end_loads=(), max_degree=8, min_degree=2):
    """Solve cumulative degree spaces on the default symbolic beam."""
    rectangle = default_beam()
    end_loads = list(end_loads)
    return solve_polynomial_problem(
        tractions, rectangle, x, y, max_degree=max_degree, min_degree=min_degree,
        extra_constraints=lambda stress: end_constraints(stress, rectangle, end_loads),
    )