"""Scalar-field evaluation and contour plotting."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp


def evaluate_on_grid(expr, rectangle, x, y, nx=200, ny=80):
    """Evaluate a SymPy expression on a rectangular NumPy grid."""
    if nx < 2 or ny < 2:
        raise ValueError("nx and ny must be at least 2")
    free_symbols = sp.sympify(expr).free_symbols - {x, y}
    if free_symbols:
        names = ", ".join(sorted((str(symbol) for symbol in free_symbols)))
        raise ValueError(f"绘图前还有未赋值符号：{names}")
    x_values = np.linspace(float(rectangle.x_min), float(rectangle.x_max), nx)
    y_values = np.linspace(float(rectangle.y_min), float(rectangle.y_max), ny)
    X, Y = np.meshgrid(x_values, y_values)
    function = sp.lambdify((x, y), expr, modules="numpy")
    Z = np.asarray(function(X, Y), dtype=float)
    if Z.shape == ():
        Z = np.full(X.shape, float(Z))
    else:
        Z = np.broadcast_to(Z, X.shape).astype(float)
    return X, Y, Z


def plot_scalar_field(
    expr,
    rectangle,
    x,
    y,
    title,
    xlabel="x",
    ylabel="y",
    cmap="coolwarm",
    save_path=None,
):
    """Plot a scalar field in a textbook-style rectangular specimen view.

    Non-constant fields use a light contour fill and contour lines.  Constant
    fields are rendered as a visible uniform specimen with a value annotation;
    this is important for elementary states such as ``sigma_x = 2``.
    """
    X, Y, Z = evaluate_on_grid(expr, rectangle, x, y)
    fig, ax = plt.subplots(figsize=(8, 5.2))
    value_min = float(np.nanmin(Z))
    value_max = float(np.nanmax(Z))
    value_range = value_max - value_min
    if value_range == 0:
        # Use a deliberately non-degenerate colour range, while displaying
        # the actual constant value in the specimen.
        half_range = max(1.0, abs(value_min) * 0.15)
        image = ax.imshow(
            Z,
            extent=(float(rectangle.x_min), float(rectangle.x_max),
                    float(rectangle.y_min), float(rectangle.y_max)),
            origin="lower",
            aspect="auto",
            cmap=cmap,
            vmin=value_min - half_range,
            vmax=value_max + half_range,
        )
        contour = image
        ax.text(
            0.5,
            0.5,
            f"{value_min:g}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=16,
            color="black",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "black"},
        )
    else:
        contour = ax.contourf(X, Y, Z, levels=30, cmap=cmap)
        ax.contour(X, Y, Z, levels=8, colors="0.25", linewidths=0.45)

    ax.plot(
        [rectangle.x_min, rectangle.x_max, rectangle.x_max, rectangle.x_min,
         rectangle.x_min],
        [rectangle.y_min, rectangle.y_min, rectangle.y_max, rectangle.y_max,
         rectangle.y_min],
        color="black",
        linewidth=1.4,
    )
    fig.colorbar(contour, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal", adjustable="box")
    if rectangle.y_down:
        ax.invert_yaxis()
    ax.grid(color="0.75", linewidth=0.35, linestyle="--")
    fig.tight_layout()
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
    return fig, ax


def plot_surface(
    expr,
    rectangle,
    x,
    y,
    title="U(x, y)",
    save_path=None,
    cmap="viridis",
):
    """Plot a symbolic scalar expression as a basic 3D surface."""
    X, Y, Z = evaluate_on_grid(expr, rectangle, x, y)
    fig = plt.figure(figsize=(8, 5.8))
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(X, Y, Z, cmap=cmap, linewidth=0, antialiased=True)
    fig.colorbar(surface, ax=ax, shrink=0.72, pad=0.1, label="value")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("U")
    if rectangle.y_down:
        ax.invert_yaxis()
    fig.tight_layout()
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
    return fig, ax
