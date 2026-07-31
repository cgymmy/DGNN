"""
经典间断有限元 (Discontinuous Galerkin) 方法模块
============================================

包含以下求解器:
- Poisson1D_DG: 1D Poisson 方程 DG 求解器
- Burgers1D_DG: 1D Burgers 方程 DG 求解器
- Poisson2D_DG: 2D Poisson 方程 DG 求解器 (不规则边界)

以及基础组件:
- QuadratureRule: 高斯正交规则
- TestFunction: 测试函数空间
- DGSolver1D: 1D DG 求解器基类
"""

from .dg_solver import (
    QuadratureRule,
    TestFunction,
    DGSolver1D,
    Poisson1D_DG,
    Burgers1D_DG,
    Poisson2D_DG,
    demo_1d_poisson,
    demo_2d_poisson,
)

__all__ = [
    "QuadratureRule",
    "TestFunction",
    "DGSolver1D",
    "Poisson1D_DG",
    "Burgers1D_DG",
    "Poisson2D_DG",
    "demo_1d_poisson",
    "demo_2d_poisson",
]
