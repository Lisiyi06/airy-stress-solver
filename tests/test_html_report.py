import matplotlib
matplotlib.use("Agg")

from types import SimpleNamespace

from airy_solver.beam_cases import get_case_problem, solve_case
from airy_solver.boundary import TractionBC
from airy_solver.rendering.html_report import write_html_report


def test_report_contains_real_inputs_expressions_and_embedded_images(tmp_path):
    conditions, loads, _ = get_case_problem("1")
    solution = solve_case("1")
    import sympy as sp
    from airy_solver.geometry import h, default_beam
    from airy_solver.symbols import x, y
    from airy_solver.beam import end_constraints
    from airy_solver.stress import stress_from_airy

    F = sp.Symbol("F", real=True)
    assert sp.simplify(solution.sigma_x + 12*F*x*y/h**3) == 0
    assert sp.simplify(solution.tau_xy + 3*F*(h**2-4*y**2)/(2*h**3)) == 0
    assert all(sp.simplify(c.expression-c.target) == 0 for c in
               end_constraints(stress_from_airy(solution.U, x, y), default_beam(), loads))
    path = write_html_report(solution, conditions, loads, tmp_path, title="<demo>")
    text = path.read_text(encoding="utf-8")
    assert "&lt;demo&gt;" in text
    assert "left,t,0,F,0" in text
    assert "right,t,0,-F,F*L" in text
    assert "F = 1" in text
    assert "<th>类型</th>" not in text
    assert text.count("q<sub>x</sub>") == 2
    assert text.count("q<sub>y</sub>") == 2
    assert text.count('src="data:image/png;base64,') == 4
    assert "<math " in text
    from html import escape
    for name in ("U", "sigma_x", "sigma_y", "tau_xy"):
        expanded = sp.expand(getattr(solution, name))
        assert f'<pre>{escape(str(expanded))}</pre>' in text
    assert "<mo>(</mo>" not in text
    assert all((tmp_path / name).stat().st_size > 0 for name in
               ("U_surface.png", "sigma_x.png", "sigma_y.png", "tau_xy.png"))


def test_report_preserves_unspecified_components(tmp_path):
    solution = SimpleNamespace(U=0, sigma_x=0, sigma_y=0, tau_xy=0,
                               diagnostics=SimpleNamespace(highest_degree=2))
    path = write_html_report(solution, [TractionBC("top", None, 0)], [], tmp_path)
    text = path.read_text(encoding="utf-8")
    assert "未指定" in text
    assert "top,r,?,0" in text


def test_interactive_report_receives_user_conditions(monkeypatch, tmp_path):
    from airy_solver.cli import interactive
    from airy_solver.rendering import html_report

    lines = iter(["4", "top,r,0,0", "bottom,r,0,0",
                  "right,t,0,P,0", "done"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(lines))
    received = {}

    def record(solution, conditions, loads, output_dir):
        received.update(conditions=conditions, loads=loads, output_dir=output_dir)
        return tmp_path / "report.html"

    monkeypatch.setattr(html_report, "write_html_report", record)
    interactive(report_dir=tmp_path)
    assert [bc.edge for bc in received["conditions"]] == ["top", "bottom"]
    assert received["loads"][0].edge == "right"
    assert str(received["loads"][0].Fy) == "P"
    assert received["output_dir"] == tmp_path