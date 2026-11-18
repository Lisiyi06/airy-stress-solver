import matplotlib

matplotlib.use("Agg")

import sympy as sp
import pytest

from airy_solver.geometry import Rectangle
from airy_solver.rendering.boundary_plot import plot_boundary_tractions
from airy_solver.rendering.fields import evaluate_on_grid, plot_scalar_field
from airy_solver.rendering.report import (
    plot_boundary_loads,
    plot_solution_report,
    plot_stress_components,
)
from airy_solver.symbols import x, y


RECTANGLE = Rectangle(0, 2, -1, 1)
STRESS = {"sigma_x": 2 * y, "sigma_y": x, "tau_xy": x + y}


def test_evaluate_on_grid_uses_expected_shape_and_values():
    X, Y, Z = evaluate_on_grid(x + 2 * y, RECTANGLE, x, y, nx=5, ny=3)

    assert X.shape == Y.shape == Z.shape == (3, 5)
    assert X[0, 0] == 0
    assert Y[0, 0] == -1
    assert Z[0, 0] == -2
    assert Z[-1, -1] == 4


def test_evaluate_on_grid_rejects_unassigned_parameters():
    load = sp.Symbol("P")

    with pytest.raises(ValueError, match="未赋值符号：P"):
        evaluate_on_grid(load * y, RECTANGLE, x, y, nx=5, ny=3)


def test_plot_scalar_field_writes_file(tmp_path):
    output = tmp_path / "field.png"
    fig, ax = plot_scalar_field(
        x**2 + y**2, RECTANGLE, x, y, "field", save_path=output
    )

    assert output.exists()
    assert fig.axes[0] is ax


def test_beam_plots_follow_downward_y():
    import matplotlib.pyplot as plt
    from airy_solver.rendering.fields import plot_surface

    rectangle = Rectangle(0, 10, -1.5, 1.5, y_down=True)
    for plot in (
        lambda: plot_scalar_field(y, rectangle, x, y, "stress"),
        lambda: plot_surface(x*y, rectangle, x, y),
    ):
        fig, ax = plot()
        assert ax.yaxis_inverted()
        plt.close(fig)


def test_boundary_traction_plot_writes_file(tmp_path):
    output = tmp_path / "tractions.png"
    fig, ax = plot_boundary_tractions(
        STRESS, RECTANGLE, x=x, y=y, n_arrows=4, save_path=output
    )

    assert output.exists()
    assert ax.get_title() == "Boundary tractions"
    assert len(ax.collections) > 0


def test_component_and_report_helpers_write_expected_files(tmp_path):
    component_dir = tmp_path / "components"
    paths = plot_stress_components(STRESS, RECTANGLE, x, y, component_dir)

    assert set(paths) == {"sigma_x", "sigma_y", "tau_xy"}
    assert all(path.exists() for path in paths.values())

    boundary_path = tmp_path / "boundary.png"
    assert plot_boundary_loads(STRESS, RECTANGLE, x, y, boundary_path) == boundary_path
    assert boundary_path.exists()

    report_dir = tmp_path / "report"
    report_paths = plot_solution_report(STRESS, RECTANGLE, x, y, report_dir)
    assert set(report_paths) == {"sigma_x", "sigma_y", "tau_xy", "boundary_loads"}
    assert all(path.exists() for path in report_paths.values())
