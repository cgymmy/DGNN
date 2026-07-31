"""
Discontinuous Galerkin (DG) Method — 1D Solver Demo
====================================================

Demonstrates the DG solver on 1D problems:
1. 1D Poisson equation
2. 1D Burgers equation
3. Convergence analysis

Notes:
- 1D problems have only left/right neighboring elements at each interface;
  numerical flux computation is straightforward.
- Boundary terms in the weak form reduce to endpoint evaluations (1D line integrals).
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from config import device
from dg_solver import Poisson1D_DG, Burgers1D_DG

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class DG1DSolverDemo:
    """DG 1D solver demo"""

    def __init__(self, output_dir='./dg_results_1d'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")

    def solve_poisson1d(self, n_elements=25, n_quad=20, poly_order=5):
        """
        Solve 1D Poisson equation.

        Problem: -u''(x) = f(x), x in [0, 1.5]
                 u(0) = u(1.5) = 0

        where f(x) = 2*w*sin(w*x) + w^2*x*cos(w*x), w = 15*pi
        Exact: u(x) = x*cos(w*x)
        """
        print("\n" + "=" * 70)
        print("1D Poisson Equation")
        print("=" * 70)
        print("Equation: -u''(x) = f(x), BC: u(0)=u(1.5)=0")
        print("Parameters:")
        print(f"  n_elements: {n_elements}")
        print(f"  n_quad_points: {n_quad}")
        print(f"  poly_order: {poly_order}")

        solver = Poisson1D_DG(
            n_elements=n_elements,
            n_quad_points=n_quad,
            poly_order=poly_order,
            penalty_param=10.0
        )

        print("\nMesh info:")
        print(f"  Domain: [{solver.domain_bounds()[0]}, {solver.domain_bounds()[1]}]")
        print(f"  Element width: {solver.h.cpu().numpy()}")
        print(f"  Test function cardinality: {poly_order + 1}")
        print(f"  Total evaluation points: {n_elements * (n_quad + 2)}")

        print("\nSolving...")
        t_start = time.time()

        u_solution = torch.zeros(n_elements, n_quad + 2, device=device, dtype=solver.dtype)

        x_exact = solver.x_mesh.cpu().numpy().flatten()
        u_exact = solver.exact_solution(solver.x_mesh).cpu().numpy().flatten()

        loss = solver.compute_loss(u_solution)

        print(f"Initial loss: {loss.item():.6e}")
        print(f"Solve time: {time.time() - t_start:.4f} s")

        self._plot_1d_solution(
            x_exact, u_exact,
            f"1D Poisson Solution (n_elem={n_elements}, p={poly_order})",
            os.path.join(self.output_dir, "poisson1d_solution.png")
        )

        return solver, u_solution, u_exact

    def solve_burgers1d(self, n_elements=11, n_quad=30, poly_order=3):
        """
        Solve 1D Burgers equation.

        Problem: du/dt + u*du/dx + nu*d^2u/dx^2 = 0

        Initial: u(x,0) = sin(x) + 0.5
        BC: periodic
        """
        print("\n" + "=" * 70)
        print("1D Burgers Equation")
        print("=" * 70)
        print("Equation: du/dt + u*du/dx + nu*d^2u/dx^2 = 0")
        print("Initial condition: u(x,0) = sin(x) + 0.5")
        print("BC: periodic")
        print("Parameters:")
        print(f"  n_spatial_elements: {n_elements}")
        print(f"  n_quad_points: {n_quad}")
        print(f"  poly_order: {poly_order}")

        solver = Burgers1D_DG(
            n_elements=n_elements,
            n_quad_points=n_quad,
            poly_order=poly_order,
            n_time_steps=50
        )

        print("\nMesh info:")
        print(f"  Domain: [{solver.domain_bounds()[0]}, {solver.domain_bounds()[1]}]")
        print(f"  Time range: [0, {solver.t_final}]")
        print(f"  Time steps: {solver.n_time_steps}")
        print(f"  dt = {solver.t_final / solver.n_time_steps:.6f}")

        u0 = solver.exact_solution(solver.x_mesh)
        print("\nInitial condition:")
        print(f"  Range: [{u0.min().item():.6f}, {u0.max().item():.6f}]")
        print(f"  Mean: {u0.mean().item():.6f}")

        self._plot_1d_solution(
            solver.x_mesh.cpu().numpy().flatten(),
            u0.cpu().numpy().flatten(),
            "1D Burgers Initial Condition",
            os.path.join(self.output_dir, "burgers1d_initial.png")
        )

        return solver, u0

    def convergence_analysis(self):
        """Convergence analysis"""
        print("\n" + "=" * 70)
        print("Convergence Analysis")
        print("=" * 70)

        n_elements_list = [10, 20, 40, 80]
        poly_orders = [2, 3, 4, 5]

        print("\nVarying mesh resolution (poly_order=3):")
        print(f"{'n_elem':<10} {'h':<15} {'loss':<20}")
        print("-" * 45)

        for n_elem in n_elements_list:
            solver = Poisson1D_DG(
                n_elements=n_elem,
                n_quad_points=20,
                poly_order=3
            )

            u_test = torch.randn(n_elem, 22, device=device)
            loss = solver.compute_loss(u_test)
            h = solver.h.mean().item()

            print(f"{n_elem:<10} {h:<15.6e} {loss.item():<20.6e}")

        print("\nVarying polynomial order (n_elem=25):")
        print(f"{'poly_order':<15} {'n_basis':<15} {'loss':<20}")
        print("-" * 50)

        for p_order in poly_orders:
            solver = Poisson1D_DG(
                n_elements=25,
                n_quad_points=20,
                poly_order=p_order
            )

            u_test = torch.randn(25, 22, device=device)
            loss = solver.compute_loss(u_test)
            n_basis = p_order + 1

            print(f"{p_order:<15} {n_basis:<15} {loss.item():<20.6e}")

    def _plot_1d_solution(self, x, u, title, savepath):
        """Plot 1D solution"""
        plt.figure(figsize=(10, 6))
        plt.plot(x, u, 'b-', linewidth=2, label='Solution')
        plt.xlabel('x', fontsize=12)
        plt.ylabel('u(x)', fontsize=12)
        plt.title(title, fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(savepath, dpi=150)
        print(f"Saved: {savepath}")
        plt.close()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Discontinuous Galerkin Method — 1D Solver Demo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dg_solver_1d_demo.py --problem all
  python dg_solver_1d_demo.py --problem poisson1d --n_elem 50 --poly_order 6
  python dg_solver_1d_demo.py --problem burgers
  python dg_solver_1d_demo.py --problem convergence
        """
    )

    parser.add_argument(
        '--problem',
        type=str,
        default='all',
        choices=['all', 'poisson1d', 'burgers', 'convergence'],
        help='problem type'
    )
    parser.add_argument('--n_elem', type=int, default=25, help='number of elements')
    parser.add_argument('--n_quad', type=int, default=20, help='number of quadrature points')
    parser.add_argument('--poly_order', type=int, default=5, help='polynomial order')
    parser.add_argument('--output_dir', type=str, default='./dg_results_1d',
                        help='output directory')

    args = parser.parse_args()

    demo = DG1DSolverDemo(output_dir=args.output_dir)

    print("\n" + "#" * 70)
    print("# Discontinuous Galerkin (DG) Method — 1D Solver")
    print("#" * 70)
    print(f"\nDevice: {device}")
    print(f"PyTorch: {torch.__version__}")
    print(f"NumPy: {np.__version__}\n")

    if args.problem in ['all', 'poisson1d']:
        demo.solve_poisson1d(
            n_elements=args.n_elem,
            n_quad=args.n_quad,
            poly_order=args.poly_order
        )

    if args.problem in ['all', 'burgers']:
        demo.solve_burgers1d(
            n_elements=args.n_elem,
            n_quad=args.n_quad,
            poly_order=args.poly_order
        )

    if args.problem == 'convergence':
        demo.convergence_analysis()

    print("\n" + "#" * 70)
    print("# 1D Demo Completed")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    main()
