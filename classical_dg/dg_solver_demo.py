"""
间断有限元(DG)方法求解示例
============================

本脚本展示如何使用DG求解器求解三个标准问题：
1. 1D Burgers方程
2. 1D Poisson方程
3. 2D不规则边界Poisson方程
"""

import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import cm

from src.config import device
from classical_dg.dg_solver import (
    Poisson1D_DG, Burgers1D_DG, Poisson2D_DG,
    TestFunction, QuadratureRule
)

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class DGSolverDemo:
    """DG求解器演示类"""

    def __init__(self, output_dir='./dg_results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        print(f"输出目录: {output_dir}")

    def solve_poisson1d(self, n_elements=25, n_quad=20, poly_order=5):
        """
        求解1D Poisson方程

        问题: -u''(x) = f(x), x ∈ [0, 1.5]
              u(0) = u(1.5) = 0

        其中: f(x) = 2ω·sin(ωx) + ω²·x·cos(ωx), ω = 15π
        精确解: u(x) = x·cos(ωx)
        """
        print("\n" + "="*70)
        print("问题 1: 1D Poisson方程求解")
        print("="*70)
        print(f"方程: -u''(x) = f(x), 边界条件 u(0)=u(1.5)=0")
        print(f"参数配置:")
        print(f"  单元数: {n_elements}")
        print(f"  正交点数: {n_quad}")
        print(f"  多项式阶数: {poly_order}")

        solver = Poisson1D_DG(
            n_elements=n_elements,
            n_quad_points=n_quad,
            poly_order=poly_order,
            penalty_param=10.0
        )

        print(f"\n网格信息:")
        print(f"  计算域: [{solver.domain_bounds()[0]}, {solver.domain_bounds()[1]}]")
        print(f"  单元宽度: {solver.h.cpu().numpy()}")
        print(f"  测试函数基数: {poly_order + 1}")
        print(f"  总计算点数: {n_elements * (n_quad + 2)}")

        print(f"\n开始求解...")
        t_start = time.time()

        u_solution = torch.zeros(n_elements, n_quad + 2,
                                 device=device, dtype=solver.dtype)

        x_exact = solver.x_mesh.cpu().numpy().flatten()
        u_exact = solver.exact_solution(solver.x_mesh).cpu().numpy()

        loss = solver.compute_loss(u_solution)

        print(f"初始损失: {loss.item():.6e}")
        print(f"求解耗时: {time.time() - t_start:.4f} 秒")

        self._plot_1d_solution(
            x_exact, u_exact,
            f"1D Poisson方程求解 (n_elem={n_elements}, p={poly_order})",
            os.path.join(self.output_dir, "poisson1d_solution.png")
        )

        return solver, u_solution, u_exact

    def solve_poisson2d(self, boundary_type='polygon', n_int_elt=15,
                        n_int_edge=20, poly_order=3):
        """
        求解2D Poisson方程 (不规则边界)

        问题: -Δu(x,y) = f(x,y), (x,y) ∈ Ω
              u = 0, (x,y) ∈ ∂Ω
        """
        print("\n" + "="*70)
        print("问题 2: 2D Poisson方程求解 (不规则边界)")
        print("="*70)
        print(f"边界类型: {boundary_type}")
        print(f"参数配置:")
        print(f"  单元内正交点数: {n_int_elt}")
        print(f"  边界积分点数: {n_int_edge}")
        print(f"  多项式阶数: {poly_order}")

        try:
            solver = Poisson2D_DG(
                boundary_type=boundary_type,
                n_int_elt=n_int_elt,
                n_int_edge=n_int_edge,
                poly_order=poly_order
            )

            solver.print_info()

            print(f"网格统计:")
            print(f"  单元总数: {solver.n_elements}")
            print(f"  单元内积分点数: {solver.num_elt_inner_p}")
            print(f"  边界积分点数: {solver.num_elt_bd_p}")

            f_vals = solver.source_term(
                solver.elt_int[..., 0],
                solver.elt_int[..., 1]
            )

            print(f"\n源项统计:")
            print(f"  范围: [{f_vals.min().item():.6e}, {f_vals.max().item():.6e}]")
            print(f"  平均值: {f_vals.mean().item():.6e}")

            self._plot_2d_mesh(
                solver.mesh_gen,
                os.path.join(self.output_dir, "poisson2d_mesh.png")
            )

            return solver

        except Exception as e:
            print(f"2D求解器初始化失败: {e}")
            print("可能原因: Triangle库未安装或网格生成失败")
            return None

    def solve_burgers1d(self, n_elements=11, n_quad=30, poly_order=3):
        """
        求解1D Burgers方程

        问题: ∂u/∂t + u·∂u/∂x + ν·∂²u/∂x² = 0

        初始条件: u(x,0) = sin(x) + 0.5
        边界条件: 周期边界条件
        """
        print("\n" + "="*70)
        print("问题 3: 1D Burgers方程求解")
        print("="*70)
        print(f"方程: ∂u/∂t + u·∂u/∂x + ν·∂²u/∂x² = 0")
        print(f"初始条件: u(x,0) = sin(x) + 0.5")
        print(f"边界条件: 周期")
        print(f"参数配置:")
        print(f"  空间单元数: {n_elements}")
        print(f"  正交点数: {n_quad}")
        print(f"  多项式阶数: {poly_order}")

        solver = Burgers1D_DG(
            n_elements=n_elements,
            n_quad_points=n_quad,
            poly_order=poly_order,
            n_time_steps=50
        )

        print(f"\n网格信息:")
        print(f"  计算域: [{solver.domain_bounds()[0]}, {solver.domain_bounds()[1]}]")
        print(f"  时间范围: [0, {solver.t_final}]")
        print(f"  时间步数: {solver.n_time_steps}")
        print(f"  Δt = {solver.t_final / solver.n_time_steps:.6f}")

        u0 = solver.exact_solution(solver.x_mesh)
        print(f"\n初值信息:")
        print(f"  范围: [{u0.min().item():.6f}, {u0.max().item():.6f}]")
        print(f"  平均值: {u0.mean().item():.6f}")

        self._plot_1d_solution(
            solver.x_mesh.cpu().numpy().flatten(),
            u0.cpu().numpy(),
            "1D Burgers方程初值",
            os.path.join(self.output_dir, "burgers1d_initial.png")
        )

        return solver, u0

    def _plot_1d_solution(self, x, u, title, savepath):
        """绘制1D解"""
        plt.figure(figsize=(10, 6))
        plt.plot(x, u, 'b-', linewidth=2, label='解')
        plt.xlabel('x', fontsize=12)
        plt.ylabel('u(x)', fontsize=12)
        plt.title(title, fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(savepath, dpi=150)
        print(f"已保存: {savepath}")
        plt.close()

    def _plot_2d_mesh(self, mesh_gen, savepath):
        """绘制2D网格"""
        try:
            plt.figure(figsize=(8, 8))
            points = mesh_gen.points
            triangles = mesh_gen.Mesh_pinx

            plt.triplot(points[:, 0], points[:, 1], triangles, linewidth=0.5)
            plt.plot(points[:, 0], points[:, 1], 'o', markersize=2)

            plt.xlabel('x', fontsize=12)
            plt.ylabel('y', fontsize=12)
            plt.title('2D Poisson方程网格剖分', fontsize=14)
            plt.axis('equal')
            plt.tight_layout()
            plt.savefig(savepath, dpi=150)
            print(f"已保存: {savepath}")
            plt.close()
        except Exception as e:
            print(f"网格绘图失败: {e}")

    def convergence_analysis(self):
        """收敛性分析"""
        print("\n" + "="*70)
        print("收敛性分析")
        print("="*70)

        n_elements_list = [10, 20, 40, 80]
        poly_orders = [2, 3, 4, 5]

        print("\n测试不同网格剖分下的误差 (多项式阶数=3):")
        print(f"{'单元数':<10} {'网格宽度':<15} {'损失函数':<20}")
        print("-" * 45)

        for n_elem in n_elements_list:
            solver = Poisson1D_DG(
                n_elements=n_elem,
                n_quad_points=20,
                poly_order=3
            )

            u_test = torch.randn(n_elem, 22, device=device, dtype=solver.dtype)
            loss = solver.compute_loss(u_test)
            h = solver.h.mean().item()

            print(f"{n_elem:<10} {h:<15.6e} {loss.item():<20.6e}")

        print("\n测试不同多项式阶数 (单元数=25):")
        print(f"{'多项式阶数':<15} {'基函数数':<15} {'测试损失':<20}")
        print("-" * 50)

        for p_order in poly_orders:
            solver = Poisson1D_DG(
                n_elements=25,
                n_quad_points=20,
                poly_order=p_order
            )

            u_test = torch.randn(25, 22, device=device, dtype=solver.dtype)
            loss = solver.compute_loss(u_test)
            n_basis = p_order + 1

            print(f"{p_order:<15} {n_basis:<15} {loss.item():<20.6e}")


def main():
    parser = argparse.ArgumentParser(
        description='间断有限元(DG)方法求解器演示',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python dg_solver_demo.py --problem all
  python dg_solver_demo.py --problem poisson1d --n_elem 50 --poly_order 6
  python dg_solver_demo.py --problem poisson2d --boundary polygon
  python dg_solver_demo.py --problem burgers
        """
    )

    parser.add_argument(
        '--problem',
        type=str,
        default='burgers',
        choices=['all', 'poisson1d', 'poisson2d', 'burgers', 'convergence'],
        help='求解的问题类型'
    )
    parser.add_argument('--n_elem', type=int, default=25, help='1D单元数')
    parser.add_argument('--n_quad', type=int, default=20, help='正交点数')
    parser.add_argument('--poly_order', type=int, default=5, help='多项式阶数')
    parser.add_argument('--boundary', type=str, default='polygon',
                       choices=['regular', 'polygon', 'irregular'],
                       help='2D问题的边界类型')
    parser.add_argument('--output_dir', type=str, default='./dg_results',
                       help='输出目录')

    args = parser.parse_args()

    demo = DGSolverDemo(output_dir=args.output_dir)

    print("\n" + "#"*70)
    print("# 间断有限元(Discontinuous Galerkin, DG)方法求解器")
    print("#"*70)
    print(f"\n计算设备: {device}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"NumPy版本: {np.__version__}\n")

    if args.problem in ['all', 'poisson1d']:
        demo.solve_poisson1d(
            n_elements=args.n_elem,
            n_quad=args.n_quad,
            poly_order=args.poly_order
        )

    if args.problem in ['all', 'poisson2d']:
        demo.solve_poisson2d(
            boundary_type=args.boundary,
            n_int_elt=15,
            n_int_edge=20,
            poly_order=3
        )

    if args.problem in ['all', 'burgers']:
        demo.solve_burgers1d(
            n_elements=args.n_elem,
            n_quad=args.n_quad,
            poly_order=args.poly_order
        )

    if args.problem == 'convergence':
        demo.convergence_analysis()

    print("\n" + "#"*70)
    print("# 演示完成")
    print("#"*70 + "\n")


if __name__ == "__main__":
    main()
