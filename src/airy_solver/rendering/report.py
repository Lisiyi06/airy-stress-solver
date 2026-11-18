"""Convenience functions for complete stress-image reports."""

from pathlib import Path

from .boundary_plot import plot_boundary_tractions
from .fields import plot_scalar_field, plot_surface


def _stress_value(solution, name):
    if isinstance(solution, dict):
        return solution[name]
    return getattr(solution, name)


def plot_stress_components(solution, rectangle, x, y, output_dir):
    """Write ``sigma_x.png``, ``sigma_y.png`` and ``tau_xy.png``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    titles = {
        "sigma_x": "sigma_x",
        "sigma_y": "sigma_y",
        "tau_xy": "tau_xy",
    }
    for name, title in titles.items():
        path = output_dir / f"{name}.png"
        plot_scalar_field(
            _stress_value(solution, name),
            rectangle,
            x,
            y,
            title=title,
            save_path=path,
        )
        paths[name] = path
    return paths


def plot_stress_component(solution, name, rectangle, x, y, output_path):
    """Draw one stress component and return its output path.

    ``name`` must be one of ``sigma_x``, ``sigma_y`` or ``tau_xy``.
    """
    if name not in {"sigma_x", "sigma_y", "tau_xy"}:
        raise ValueError("name must be sigma_x, sigma_y or tau_xy")
    output_path = Path(output_path)
    plot_scalar_field(
        _stress_value(solution, name),
        rectangle,
        x,
        y,
        title=name,
        save_path=output_path,
    )
    return output_path


def plot_boundary_loads(solution, rectangle, x, y, output_path):
    """Write a boundary-traction plot derived from a solution."""
    stress = {
        "sigma_x": _stress_value(solution, "sigma_x"),
        "sigma_y": _stress_value(solution, "sigma_y"),
        "tau_xy": _stress_value(solution, "tau_xy"),
    }
    plot_boundary_tractions(stress, rectangle, x=x, y=y, save_path=output_path)
    return Path(output_path)


def plot_solution_report(solution, rectangle, x, y, output_dir):
    """Write the three stress plots and one boundary-load plot."""
    output_dir = Path(output_dir)
    paths = plot_stress_components(solution, rectangle, x, y, output_dir)
    paths["boundary_loads"] = plot_boundary_loads(
        solution, rectangle, x, y, output_dir / "boundary_loads.png"
    )
    return paths


def plot_basic_solution_report(solution, rectangle, x, y, output_dir):
    """Write one Airy surface and the three basic stress heatmaps."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {"U": output_dir / "U_surface.png"}
    airy_function = solution["U"] if isinstance(solution, dict) else solution.U
    plot_surface(airy_function, rectangle, x, y,
                 title="Airy stress function U(x, y)", save_path=paths["U"])
    paths.update(plot_stress_components(solution, rectangle, x, y, output_dir))
    return paths
