"""Offline, self-contained HTML reports using native MathML and embedded PNGs."""

import base64
from html import escape
from pathlib import Path

import matplotlib.pyplot as plt
import sympy as sp

from ..geometry import L, h, Rectangle
from ..symbols import x, y
from .fields import plot_scalar_field, plot_surface


def formula(value):
    if value is None:
        return '<span class="muted">未指定</span>'
    markup = sp.mathml(sp.sympify(value), printer="presentation")
    return f'<math xmlns="http://www.w3.org/1998/Math/MathML">{markup}</math>'


def write_html_report(solution, conditions, loads, output_dir, title="Airy 应力函数求解报告"):
    """Render symbolic inputs/outputs and a unit-parameter 10-by-3 preview."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    names = ("U", "sigma_x", "sigma_y", "tau_xy")
    labels = ("U", "σx", "σy", "τxy")
    expressions = {name: sp.sympify(getattr(solution, name)) for name in names}
    # Include parameters from input data even if they cancel from the result.
    targets = [v for bc in conditions for v in (bc.tx, bc.ty) if v is not None]
    targets += [v for load in loads for v in (load.Fx, load.Fy, load.M) if v is not None]
    symbols = set().union(*(sp.sympify(v).free_symbols for v in [*expressions.values(), *targets]))
    parameters = {L: sp.Integer(10), h: sp.Integer(3)}
    parameters.update({s: sp.Integer(1) for s in sorted(symbols - {L, h, x, y}, key=str)})
    rectangle = Rectangle(0, 10, -sp.Rational(3, 2), sp.Rational(3, 2), y_down=True)
    rows = []
    raw_lines = []
    for bc in conditions:
        prefix = 'q' if bc.edge in ('top', 'bottom') else 't'
        rows.append(f'<tr><td>{escape(bc.edge)}</td>'
                    f'<td>{prefix}<sub>x</sub> = {formula(bc.tx)}</td>'
                    f'<td>{prefix}<sub>y</sub> = {formula(bc.ty)}</td><td>—</td></tr>')
        raw_lines.append(','.join([bc.edge, 'r', *('?' if v is None else str(v) for v in (bc.tx, bc.ty))]))
    for load in loads:
        rows.append(f'<tr><td>{escape(load.edge)}</td>'
                    f'<td>Fx = {formula(load.Fx)}</td><td>Fy = {formula(load.Fy)}</td>'
                    f'<td>M = {formula(load.M)}</td></tr>')
        raw_lines.append(','.join([load.edge, 't', *('?' if v is None else str(v) for v in (load.Fx, load.Fy, load.M))]))
    cards, figures = [], []
    for name, label in zip(names, labels):
        expression = sp.expand(expressions[name])
        cards.append(f'<article><h3>{label}</h3><div class="formula">{formula(expression)}</div>'
                     f'<details><summary>可复制表达式</summary><pre>{escape(str(expression))}</pre></details></article>')
        numeric = sp.simplify(expression.subs(parameters))
        path = output_dir / ("U_surface.png" if name == "U" else f"{name}.png")
        if name == "U":
            fig, _ = plot_surface(numeric, rectangle, x, y, save_path=path)
        else:
            fig, _ = plot_scalar_field(numeric, rectangle, x, y, name, save_path=path)
        plt.close(fig)
        image = base64.b64encode(path.read_bytes()).decode("ascii")
        figures.append(f'<article><h3>{label} · 数值分布</h3>'
                       f'<img alt="{label}" src="data:image/png;base64,{image}"></article>')
    parameter_text = '，'.join(f'{escape(str(k))} = {escape(str(v))}' for k, v in parameters.items())
    document = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>''' + escape(title) + '''</title><style>
*{box-sizing:border-box}body{margin:0;background:#f3f6fa;color:#203047;font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif}
main{max-width:1200px;margin:auto;padding:32px 24px}header{background:#18334f;color:white;padding:28px;border-radius:16px}
h1{margin:0;font-size:28px}h2{margin-top:32px}h3{margin:0 0 16px}header p{margin-bottom:0;color:#d6e6f5}
article,.panel{background:white;border:1px solid #dce4ed;border-radius:12px;padding:24px;margin:16px 0}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.grid article{margin:0}
.formula{overflow-x:auto;padding:12px 0;font-size:22px}math{font-family:Cambria Math,serif}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f6fa;padding:12px}
table{border-collapse:collapse;width:100%}td,th{padding:14px 10px;text-align:left;border-bottom:1px solid #e4eaf1}
.scroll{overflow:auto}img{width:100%;height:auto}summary{cursor:pointer;color:#32678d}.muted,footer{color:#62748a}
@media(max-width:800px){.grid{grid-template-columns:1fr}main{padding:16px}.formula{font-size:18px}}
@media print{body{background:white}article{break-inside:avoid}}
</style></head><body><main>'''
    document += f'<header><h1>{escape(title)}</h1><p>输入边界 → 多项式 Airy 求解 → 应力分布</p></header>'
    document += f'<div class="panel">求解次数：{solution.diagnostics.highest_degree}；单位厚度；x 向右、y 向下。<br>'
    document += '符号区域：0 ≤ x ≤ L，−h/2 ≤ y ≤ h/2；上表面 top 位于 y = −h/2。</div>'
    document += '<h2>01 · 输入边界条件</h2><div class="panel"><div class="scroll"><table>'
    document += '<thead><tr><th>边</th><th>x 分量</th><th>y 分量</th><th>力矩</th></tr></thead><tbody>'
    document += ''.join(rows) + '</tbody></table></div><p class="muted">面力按外法向定义；Fx 向右、Fy 向下、M 顺时针为正，M 关于端面中点。未指定不等于零，未输入的边不自动视为自由边。</p>'
    document += '<details><summary>查看可复制的边界输入</summary><pre>' + escape('\n'.join(raw_lines + ['done'])) + '</pre></details></div>'
    document += '<h2>02 · 符号解析结果</h2>' + ''.join(cards)
    document += '<h2>03 · 数值图像</h2><div class="panel">绘图参数：' + parameter_text
    document += '。<br>除尺寸外，未赋值的符号参数默认取 1，仅用于演示；以上解析表达式保留原符号。各应力图使用独立颜色条。</div>'
    document += '<div class="grid">' + ''.join(figures) + '</div>'
    document += '<footer><p>端面结果量仅为圣维南等效，不描述集中力附近的局部奇异应力；不求位移。U 可加任意一次函数而不改变应力，本图采用求解器的规范选择。此报告离线可用，图像已嵌入文件。</p></footer></main></body></html>'
    report_path = output_dir / 'report.html'
    report_path.write_text(document, encoding='utf-8')
    return report_path