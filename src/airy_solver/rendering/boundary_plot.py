"""Boundary traction arrow plots."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from ..boundary import edge_traction
from ..geometry import BOTTOM, EDGES, LEFT, RIGHT, TOP


def _edge_samples(rectangle, edge, n_arrows):
    tangent = np.linspace(0.0, 1.0, n_arrows)
    if edge in (LEFT, RIGHT):
        y_values = float(rectangle.y_min) + tangent * float(rectangle.height)
        x_value = float(rectangle.x_min if edge == LEFT else rectangle.x_max)
        return np.full(n_arrows, x_value), y_values
    x_values = float(rectangle.x_min) + tangent * float(rectangle.length)
    y_value = float(rectangle.y_min if edge == BOTTOM else rectangle.y_max)
    return x_values, np.full(n_arrows, y_value)


def _target_for_edge(boundary_conditions, edge, x_symbol, y_symbol):
    for condition in boundary_conditions or ():
        if condition.edge == edge:
            return condition.tx, condition.ty
    return None


def plot_boundary_tractions(
    stress,
    rectangle,
    boundary_conditions=None,
    x=None,
    y=None,
    n_arrows=15,
    scale=None,
    save_path=None,
):
    """Plot actual or prescribed traction arrows around a rectangle."""
    if x is None or y is None:
        raise TypeError("x and y symbols are required")
    if n_arrows < 2:
        raise ValueError("n_arrows must be at least 2")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    condition_list = boundary_conditions or ()
    x_min = float(rectangle.x_min)
    x_max = float(rectangle.x_max)
    y_min = float(rectangle.y_min)
    y_max = float(rectangle.y_max)
    span = max(float(rectangle.length), float(rectangle.height), 1.0)
    margin = 0.22 * span
    max_magnitude = 0.0
    for edge in EDGES:
        x_points, y_points = _edge_samples(rectangle, edge, n_arrows)
        target = _target_for_edge(condition_list, edge, x, y)
        if target is None:
            tx_expr, ty_expr = edge_traction(stress, rectangle, edge, x, y)
        else:
            tx_expr, ty_expr = target
            if edge in (LEFT, RIGHT):
                substitution = {x: rectangle.x_min if edge == LEFT else rectangle.x_max}
            else:
                substitution = {y: rectangle.y_min if edge == BOTTOM else rectangle.y_max}
            tx_expr = sp.sympify(tx_expr).subs(substitution)
            ty_expr = sp.sympify(ty_expr).subs(substitution)
        tx_func = sp.lambdify((x, y), tx_expr, modules="numpy")
        ty_func = sp.lambdify((x, y), ty_expr, modules="numpy")
        tx_values = np.asarray(tx_func(x_points, y_points), dtype=float)
        ty_values = np.asarray(ty_func(x_points, y_points), dtype=float)
        max_magnitude = max(
            max_magnitude,
            float(np.max(np.hypot(tx_values, ty_values))),
        )
        if not (np.allclose(tx_values, 0.0) and np.allclose(ty_values, 0.0)):
            arrow_scale = scale
            if arrow_scale is None:
                arrow_scale = max_magnitude * 5.0 or 1.0
            ax.quiver(
                x_points,
                y_points,
                tx_values,
                ty_values,
                angles="xy",
                scale_units="xy",
                scale=arrow_scale,
                color="tab:red",
                width=0.003,
            )

    # Leave room outside the specimen: the traction vectors are outward and
    # must not be clipped at the rectangle boundary.
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.plot(
        [x_min, x_max, x_max, x_min, x_min],
        [y_min, y_min, y_max, y_max, y_min],
        color="black",
        linewidth=1.8,
    )
    ax.axhline(0.0, color="0.35", linewidth=0.7) if y_min < 0 < y_max else None
    ax.axvline(0.0, color="0.35", linewidth=0.7) if x_min < 0 < x_max else None
    ax.grid(color="0.85", linewidth=0.35, linestyle="--")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Boundary tractions")
    fig.tight_layout()
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
    return fig, ax
