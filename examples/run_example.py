from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse

import sympy as sp

from airy_solver.beam_cases import CASES, solve_case, get_case_problem


def print_expressions(solution):
    print("\n========== 解析表达式 ==========")
    for name in ("U", "sigma_x", "sigma_y", "tau_xy"):
        print(f"{name} = {sp.expand(getattr(solution, name))}")


def main():
    parser = argparse.ArgumentParser(description="运行 Airy 梁案例")
    parser.add_argument("case", nargs="?", default="1", choices=sorted(CASES),
                        help="1/2/3，或案例英文名称（默认 1）")
    parser.add_argument("--render", action="store_true",
                        help="按 L=10、h=3 生成 U 曲面和三张应力图")
    parser.add_argument("--output", default=None,
                        help="图片输出目录（默认 outputs/case_<case>）")
    parser.add_argument("--open", action="store_true", help="生成报告并在浏览器中打开")
    args = parser.parse_args()
    solution = solve_case(args.case)
    print(f"案例：{args.case}")
    print(f"搜索到的最高次数：{solution.diagnostics.highest_degree}")
    print_expressions(solution)
    if args.render or args.open:
        output_dir = args.output or (
            Path(__file__).resolve().parents[1] / "outputs" / f"case_{args.case}"
        )
        from airy_solver.rendering.html_report import write_html_report
        conditions, loads, _ = get_case_problem(args.case)
        path = write_html_report(solution, conditions, loads, output_dir,
                                 title=f"Airy 梁案例 {args.case}")
        print(f"HTML 报告：{path}")
        if args.open:
            import webbrowser
            webbrowser.open(path.as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



