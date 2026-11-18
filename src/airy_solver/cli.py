"""Text-only interactive interface; arithmetic is parsed without eval."""

import ast
import argparse
import sympy as sp

from .boundary import TractionBC, edge_traction, traction_residuals
from .diagnostics import AirySolverError
from .geometry import EDGES, L, h, default_beam
from .beam import EndLoad, end_constraints, solve_beam
from .stress import biharmonic
from .symbols import x, y


def parse_expression(text):
    """Accept only numbers, parameter names and arithmetic; decimals are exact."""
    text = text.strip().replace("^", "**")
    if len(text) > 2000:
        raise ValueError("表达式过长")

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return sp.Rational(ast.get_source_segment(text, node))
        if isinstance(node, ast.Name) and not node.id.startswith("_"):
            if node.id.startswith("c_"):
                raise ValueError("c_ 前缀保留给内部未知系数")
            return {"x": x, "y": y, "L": L, "h": h}.get(node.id, sp.Symbol(node.id, real=True))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div) and right != 0:
                return left / right
            if isinstance(node.op, ast.Pow) and right.is_Integer and abs(right) <= 30:
                return left ** right
        raise ValueError("只允许数字、参数名、括号和 + - * / **；幂须为绝对值不超过30的整数")

    try:
        result = visit(ast.parse(text, mode="eval").body)
    except (SyntaxError, TypeError, ZeroDivisionError) as exc:
        raise ValueError("表达式格式错误，例如 3*x**2/2 或 -q") from exc
    if result.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
        raise ValueError("表达式必须有限")
    return result


def print_solution(solution, rectangle, conditions):
    """Print exact expressions and independent physical residual checks."""
    for name in ("U", "sigma_x", "sigma_y", "tau_xy"):
        print(f"{name} = {sp.expand(getattr(solution, name))}")
    print("诊断 =", solution.diagnostics)
    stress = {name: getattr(solution, name) for name in ("sigma_x", "sigma_y", "tau_xy")}
    print("双调和残差 =", biharmonic(solution.U, x, y))
    print("平衡方程残差 =", (
        sp.simplify(sp.diff(solution.sigma_x, x) + sp.diff(solution.tau_xy, y)),
        sp.simplify(sp.diff(solution.tau_xy, x) + sp.diff(solution.sigma_y, y)),
    ))
    for edge in EDGES:
        print(f"{edge} 实际 (tx, ty) = {edge_traction(stress, rectangle, edge, x, y)}")
    for bc in conditions:
        residuals = traction_residuals(stress, rectangle, bc, x, y)
        print(f"{bc.edge} 指定分量残差 =", tuple(
            r if target is not None else "未指定" for r, target in zip(residuals, (bc.tx, bc.ty))
        ))


def interactive(report_dir=None, open_report=False):
    print("\n" + "=" * 64)
    print("  Airy 应力函数梁求解器  |  交互模式")
    print("=" * 64)
    print("坐标：0 <= x <= L，-h/2 <= y <= h/2（x 向右，y 向下）")
    print("说明：L、h 保持为符号；单位厚度；只求应力，不求位移。")
    print("方法：逐级增加多项式次数，匹配边界面力和端面结果量。")
    print("输入：数字/参数/括号及 + - * / **，例如 q、q*x/L、-F*L。")
    print("类型：r = 分布面力，t = 端面集中力/结果量；traction/resultant 也兼容。")
    print("提示：空值按 0；? 表示不指定；输入 done 开始求解。")
    print("-" * 64)
    raw_degree = input("[1/2] 最高搜索次数（2～8，默认 6）：").strip() or "6"
    if not raw_degree.isdigit() or not 2 <= int(raw_degree) <= 8:
        raise ValueError("最高搜索次数必须是 2～8 的整数。")
    maximum = int(raw_degree)
    print("\n[2/2] 输入边界条件（每行一条）")
    print("  分布面力  top,r,0,q")
    print("  端面集中力 right,t,0,P,0")
    print("  边：top / bottom / left / right")
    print("  Fx 向右、Fy 向下、M 顺时针为正（关于端面中点）")
    print("-" * 64)
    conditions, loads = [], []
    while True:
        line = input("  bc > ").strip()
        if line.lower() == "done":
            break
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2 or parts[0] not in EDGES:
            raise ValueError("边必须是 left/right/top/bottom")
        edge, kind = parts[:2]
        if kind in ("r", "traction") and len(parts) == 4:
            values = [None if s == "?" else parse_expression(s or "0") for s in parts[2:]]
            tangent, normal = (y, x) if edge in ("left", "right") else (x, y)
            for value in values:
                if value is not None and (value.has(normal) or not value.is_polynomial(tangent)):
                    raise ValueError(f"{edge} 面力必须为关于 {tangent} 的多项式")
            conditions.append(TractionBC(edge, *values))
        elif kind in ("t", "resultant") and len(parts) == 5:
            if edge not in ("left", "right"):
                raise ValueError("上下表面禁止集中力/力偶或结果量替代分布载荷")
            values = [None if s == "?" else parse_expression(s or "0") for s in parts[2:]]
            loads.append(EndLoad(edge, *values))
        else:
            raise ValueError("格式：top,r,0,q 或 right,t,0,P,0")
    solution = solve_beam(conditions, loads, max_degree=maximum)
    print("找到当前多项式空间内唯一解；端面结果量条件仅表示圣维南等效。")
    rectangle = default_beam()
    print_solution(solution, rectangle, conditions)
    stress = {name: getattr(solution, name) for name in ("sigma_x", "sigma_y", "tau_xy")}
    for constraint in end_constraints(stress, rectangle, loads):
        print(f"{constraint.name} 积分残差 =", sp.simplify(constraint.expression - constraint.target))
    if report_dir is not None:
        from .rendering.html_report import write_html_report
        path = write_html_report(solution, conditions, loads, report_dir)
        print(f"HTML 报告：{path}")
        if open_report:
            import webbrowser
            webbrowser.open(path.as_uri())


def main():
    parser = argparse.ArgumentParser(description="交互输入矩形面力与截面约束，输出精确解析解")
    parser.add_argument("--render", action="store_true", help="生成含输入、表达式和图像的 HTML 报告")
    parser.add_argument("--open", action="store_true", help="生成报告并打开浏览器")
    parser.add_argument("--output", help="报告输出目录")
    args = parser.parse_args()
    try:
        from pathlib import Path
        report_dir = None
        if args.render or args.open or args.output:
            report_dir = args.output or Path(__file__).resolve().parents[2] / "outputs" / "interactive"
        interactive(report_dir, args.open)
    except (AirySolverError, ValueError, EOFError) as exc:
        print("未返回解析解：", exc)
        return 1
    except KeyboardInterrupt:
        print("\n已取消")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())