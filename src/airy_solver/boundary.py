"""Traction boundary conditions on rectangular domains."""

from dataclasses import dataclass

import sympy as sp

from .geometry import BOTTOM, LEFT, RIGHT, TOP, Rectangle
from .diagnostics import AirySolverError


class NonPolynomialBoundaryError(AirySolverError):
    """Raised when a boundary expression is not polynomial."""


@dataclass(frozen=True)
class TractionBC:
    """Prescribed traction on one rectangle edge."""

    edge: str
    tx: object
    ty: object


def _edge_substitution(rectangle, edge):
    if edge == LEFT:
        return {"coordinate": "x", "fixed": rectangle.x_min}
    if edge == RIGHT:
        return {"coordinate": "x", "fixed": rectangle.x_max}
    if edge == BOTTOM:
        return {"coordinate": "y", "fixed": rectangle.bottom_y}
    if edge == TOP:
        return {"coordinate": "y", "fixed": rectangle.top_y}
    raise ValueError(f"unknown rectangle edge: {edge!r}")


def _edge_value(rectangle, edge, x, y):
    if edge == LEFT:
        return {x: rectangle.x_min}, y
    if edge == RIGHT:
        return {x: rectangle.x_max}, y
    if edge == BOTTOM:
        return {y: rectangle.bottom_y}, x
    if edge == TOP:
        return {y: rectangle.top_y}, x
    raise ValueError(f"unknown rectangle edge: {edge!r}")


def edge_traction(stress, rectangle, edge, x=None, y=None):
    """Return ``(tx, ty)`` induced by ``stress`` on an edge.

    If coordinates are supplied, the fixed normal coordinate is substituted
    so the returned expressions are functions of the edge's tangent variable.
    """
    if x is None or y is None:
        raise TypeError("edge_traction requires the symbolic coordinates x and y")

    substitution, _ = _edge_value(rectangle, edge, x, y)
    sigma_x = sp.sympify(stress["sigma_x"]).subs(substitution)
    sigma_y = sp.sympify(stress["sigma_y"]).subs(substitution)
    tau_xy = sp.sympify(stress["tau_xy"]).subs(substitution)

    if edge == LEFT:
        return -sigma_x, -tau_xy
    if edge == RIGHT:
        return sigma_x, tau_xy
    if edge == BOTTOM:
        sign = 1 if rectangle.y_down else -1
        return sign*tau_xy, sign*sigma_y
    if edge == TOP:
        sign = -1 if rectangle.y_down else 1
        return sign*tau_xy, sign*sigma_y
    raise ValueError(f"unknown rectangle edge: {edge!r}")


def _polynomial_coefficients(expr, variable):
    try:
        return sp.Poly(sp.expand(expr), variable).all_coeffs()
    except (sp.PolynomialError, TypeError) as exc:
        raise NonPolynomialBoundaryError(
            f"boundary expression is not polynomial in {variable}"
        ) from exc


def polynomial_coefficient_equations(expr, variable):
    """Return coefficient equations for the polynomial identity ``expr == 0``."""
    coefficients = _polynomial_coefficients(expr, variable)
    return [sp.simplify(coefficient) for coefficient in coefficients]


def traction_residuals(stress, rectangle, bc, x, y):
    """Return actual-minus-target traction residuals on ``bc.edge``."""
    if not isinstance(bc, TractionBC):
        raise TypeError("bc must be a TractionBC instance")

    actual_tx, actual_ty = edge_traction(stress, rectangle, bc.edge, x, y)
    substitution, tangent = _edge_value(rectangle, bc.edge, x, y)
    # None is unspecified, not a traction-free component.
    target_tx = actual_tx if bc.tx is None else sp.sympify(bc.tx).subs(substitution)
    target_ty = actual_ty if bc.ty is None else sp.sympify(bc.ty).subs(substitution)

    residuals = (
        sp.expand(actual_tx - target_tx),
        sp.expand(actual_ty - target_ty),
    )
    for residual in residuals:
        _polynomial_coefficients(residual, tangent)
    return tuple(sp.simplify(residual) for residual in residuals)
