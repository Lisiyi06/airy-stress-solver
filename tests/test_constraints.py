import sympy as sp

from airy_solver.constraints import (
    IntegralConstraint,
    axial_resultant,
    bending_moment,
    constraint_equations,
    shear_resultant,
)
from airy_solver.geometry import Rectangle
from airy_solver.symbols import x, y


RECTANGLE = Rectangle(0, 10, -1, 1)


def test_section_resultants_use_exact_symbolic_integration():
    stress = {
        "sigma_x": 2 + 3 * y,
        "sigma_y": 0,
        "tau_xy": 4 * y,
    }

    assert axial_resultant(stress, RECTANGLE, 0, x, y) == 4
    assert shear_resultant(stress, RECTANGLE, 0, x, y) == 0
    assert bending_moment(stress, RECTANGLE, 0, x, y) == 2


def test_section_resultants_substitute_symbolic_section_coordinate():
    a = sp.symbols("a")
    stress = {
        "sigma_x": a * x + y,
        "sigma_y": 0,
        "tau_xy": x,
    }

    assert axial_resultant(stress, RECTANGLE, 2, x, y) == 4 * a
    assert shear_resultant(stress, RECTANGLE, 2, x, y) == 4


def test_integral_constraint_is_user_defined_without_beam_type_detection():
    constraint = IntegralConstraint(
        name="user_applied_end_couple",
        expression=sp.Symbol("M_user"),
        target=0,
    )

    assert constraint_equations([constraint]) == [sp.Symbol("M_user")]
    assert constraint.name == "user_applied_end_couple"


def test_constraint_equations_subtract_targets():
    a, target = sp.symbols("a target")
    constraint = IntegralConstraint("reaction_force", a, target)

    assert constraint_equations([constraint]) == [a - target]
