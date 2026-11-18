# Airy 应力函数梁求解器

使用 SymPy 在矩形梁上求解二维多项式 Airy 应力函数的符号解析解。

坐标约定：`0 <= x <= L`、`-h/2 <= y <= h/2`，x 向右、y 向下。应力定义为：

```text
sigma_x = U_yy
sigma_y = U_xx
tau_xy  = -U_xy
```

程序构造满足双调和方程的多项式基，匹配边界面力、端面合力和力矩，逐级搜索可行次数并返回解析解。端面集中力按 Saint-Venant 结果量处理；程序不求位移，也不描述加载点附近的局部奇异应力。

## 安装

建议使用 Python 3.11 或更高版本。在项目根目录执行：

```bat
python -m pip install -r requirements.txt
```

## 调用方式

### 普通交互调用

```bat
python solve.py
```

程序会提示输入多项式最高次数和边界条件，求解后在终端输出 `U`、`sigma_x`、`sigma_y`、`tau_xy`。边界输入格式如下：

```text
边,r,tx,ty       # 分布面力
边,t,Fx,Fy,M     # 端面集中力/结果量
```

例如：

```text
top,r,0,q
right,t,0,-q*L,q*L**2/2
```

`r` 和 `t` 是推荐写法；为兼容旧输入，`traction` 和 `resultant` 仍然有效。

边名为 `top`、`bottom`、`left`、`right`。表达式支持数字、参数、`x`、`y`、`L`、`h` 以及 `+ - * / **`。Fx 向右、Fy 向下、M 顺时针为正；端面力矩关于端面中点。

### 直接运行内置案例

```bat
python examples\run_example.py 1
python examples\run_example.py 2
python examples\run_example.py 3
```

案例编号：

| 编号 | 内容 |
| --- | --- |
| 1 | 右端固定、左端向下集中力 |
| 2 | 左端固定、上表面向下均布载荷 |
| 3 | 两端简支、上表面向下均布载荷 |

### 生成图像和 HTML 报告

```bat
python examples\run_example.py 1 --render
```

生成报告后自动打开浏览器：

```bat
python examples\run_example.py 1 --open
```

交互模式也支持生成报告：

```bat
python solve.py --render
python solve.py --open
```

默认输出路径：

```text
outputs\case_1\
outputs\case_2\
outputs\case_3\
outputs\interactive\
```

每个输出目录可能包含：

```text
report.html       离线 HTML 汇总报告
U_surface.png     Airy 函数曲面图
sigma_x.png       sigma_x 应力图
sigma_y.png       sigma_y 应力图
tau_xy.png        tau_xy 应力图
```

也可以使用 `--output` 指定输出目录：

```bat
python examples\run_example.py 1 --render --output my_results
```

渲染时使用 `L=10`、`h=3` 和单位载荷参数，仅用于生成示意图；符号解析结果不受影响。HTML 报告内嵌图片，可以离线打开。

### Python API

```python
import sys
sys.path.insert(0, "src")

from airy_solver import solve_case, cantilever_point_load

solution = solve_case("3")
print(solution.sigma_x)

solution = cantilever_point_load(F=100)
```

## 测试

```bat
set "PYTHONPATH=src"
python -B -m pytest -q -p no:cacheprovider
```

## 限制

- 只支持矩形区域、二维无体力 Airy 应力函数和多项式边界数据；
- 不求位移，不处理严格的固支或简支位移边界；
- 不保证任意边界数据都能在有限阶多项式空间中精确表示。

## 项目架构

```text
airy-stress-solver/
├── README.md                 使用说明
├── requirements.txt          项目依赖
├── .gitignore                缓存和生成文件的忽略规则
├── solve.py                  交互式入口
├── examples/
│   └── run_example.py        内置案例入口
├── src/airy_solver/
│   ├── __init__.py           对外 API
│   ├── cli.py                输入解析、交互命令行和结果输出
│   ├── beam_cases.py         梁案例边界条件
│   ├── beam.py               梁模型、端面结果量约束
│   ├── geometry.py           矩形区域和坐标约定
│   ├── symbols.py            共用符号
│   ├── polynomial.py         多项式双调和基
│   ├── catalog.py            基函数和应力目录工具
│   ├── stress.py             应力求导和双调和计算
│   ├── boundary.py           边界面力与多项式匹配
│   ├── constraints.py        积分约束
│   ├── solver.py             线性方程求解和次数搜索
│   ├── diagnostics.py        求解诊断和异常
│   └── rendering/
│       ├── __init__.py       绘图 API
│       ├── fields.py         数值网格、云图和曲面图
│       ├── html_report.py    离线 HTML 报告
│       ├── boundary_plot.py  边界面力示意图
│       └── report.py         图片批量输出
└── tests/                    正式回归测试
```

主流程：入口 → 边界条件 → 双调和多项式基 → 线性方程 → Airy 函数与应力 → 终端或 HTML 报告。

`outputs/` 为运行时生成目录，不随源码保留；其中的报告和图片可以删除并重新生成。