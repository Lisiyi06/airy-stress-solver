"""Section resultants and user-specified integral constraints."""

from dataclasses import dataclass

import sympy as sp


@dataclass(frozen=True)
class IntegralConstraint:
    """An equation imposed on a user-provided resultant or moment."""

    name: str
    expression: object
    target: object


def _stress_component(stress, name, x, x_value, y):
    return sp.sympify(stress[name]).subs(x, x_value)


def axial_resultant(stress, rectangle, x_value, x, y):
    """Return ``N_x = integral(sigma_x dy)`` at ``x=x_value``."""
    return sp.integrate(
        _stress_component(stress, "sigma_x", x, x_value, y),
        (y, rectangle.y_min, rectangle.y_max),
    )


def shear_resultant(stress, rectangle, x_value, x, y):
    """Return ``V_y = integral(tau_xy dy)`` at ``x=x_value``."""
    return sp.integrate(
        _stress_component(stress, "tau_xy", x, x_value, y),
        (y, rectangle.y_min, rectangle.y_max),
    )


def bending_moment(stress, rectangle, x_value, x, y):
    """Return ``M_z = integral(y*sigma_x dy)`` at ``x=x_value``.

    The positive sign follows the project's right-handed ``x`` right,
    ``y`` upward convention and the stated section-moment definition.
    """
    sigma_x = _stress_component(stress, "sigma_x", x, x_value, y)
    return sp.integrate(
        y * sigma_x,
        (y, rectangle.y_min, rectangle.y_max),
    )


def constraint_equations(constraints):
    """Convert user constraints into expressions that must equal zero."""
    return [
        sp.expand(sp.sympify(constraint.expression) - sp.sympify(constraint.target))
        for constraint in constraints
    ]
