"""
Discontinuous Galerkin (DG) Method — 2D Solver Demo
====================================================

Demonstrates the DG solver on 2D problems:
1. 2D Poisson equation with irregular boundaries

Notes:
- 2D requires integration over triangular/quadrilateral elements; boundaries are
  closed curves in the 2D plane.
- Boundary terms in the weak form are line integrals along the domain boundary,
  requiring proper handling of irregular geometries.
- Mesh generation (e.g. Delaunay triangulation) is a critical prerequisite that
  fundamentally distinguishes 2D from 1D.
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
from dg_solver import Poisson2D_DG

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class DG2DSolverDemo:
    """DG 2D solver demo"""

    def __init__(self, output_dir='./dg_results_2d'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")

    def solve_poisson2d(self, boundary_type='polygon', n_int_elt=15,
                        n_int_edge=20, poly_order=3):
        """
        Solve 2D Poisson equation (irregular boundary).

        Problem: -Δu(x,y) = f(x,y), (x,y) in Ω
                 u = 0, (x,y) on ∂Ω
        """
        print("\n" + "=" * 70)
        print("2D Poisson Equation (irregular boundary)")
        print("=" * 70)
        print(f"Boundary type: {boundary_type}")
        print("Parameters:")
        print(f"  interior quadrature points: {n_int_elt}")
        print(f"  edge quadrature points: {n_int_edge}")
        print(f"  polynomial order: {poly_order}")

        try:
            solver = Poisson2D_DG(
                boundary_type=boundary_type,
                n_int_elt=n_int_elt,
                n_int_edge=n_int_edge,
                poly_order=poly_order
            )

            solver.print_info()

            print("\nMesh statistics:")
            print(f"  Total elements: {solver.n_elements}")
            print(f"  Interior quadrature points: {solver.num_elt_inner_p}")
            print(f"  Edge quadrature points: {solver.num_elt_bd_p}")

            # Compute source term
            f_vals = solver.source_term(
                solver.elt_int[..., 0],
                solver.elt_int[..., 1]
            )

            print("\nSource term statistics:")
            print(f"  Range: [{f_vals.min().item():.6e}, {f_vals.max().item():.6e}]")
            print(f"  Mean: {f_vals.mean().item():.6e}")

            # Visualize mesh
            self._plot_2d_mesh(
                solver.mesh_gen,
                os.path.join(self.output_dir, "poisson2d_mesh.png")
            )

            return solver

        except Exception as e:
            print(f"2D solver initialization failed: {e}")
            print("Possible cause: Triangle library not installed or mesh generation failed")
            return None

    def _plot_2d_mesh(self, mesh_gen, savepath):
        """Plot 2D mesh"""
        try:
            plt.figure(figsize=(8, 8))
            points = mesh_gen.points
            triangles = mesh_gen.Mesh_pinx

            plt.triplot(points[:, 0], points[:, 1], triangles, linewidth=0.5)
            plt.plot(points[:, 0], points[:, 1], 'o', markersize=2)

            plt.xlabel('x', fontsize=12)
            plt.ylabel('y', fontsize=12)
            plt.title('2D Poisson Mesh', fontsize=14)
            plt.axis('equal')
            plt.tight_layout()
            plt.savefig(savepath, dpi=150)
            print(f"Saved: {savepath}")
            plt.close()
        except Exception as e:
            print(f"Mesh plotting failed: {e}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Discontinuous Galerkin Method — 2D Solver Demo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dg_solver_2d_demo.py
  python dg_solver_2d_demo.py --boundary polygon
  python dg_solver_2d_demo.py --boundary irregular --n_int_elt 20 --poly_order 4
        """
    )

    parser.add_argument('--boundary', type=str, default='irregular',
                        choices=['regular', 'polygon', 'irregular'],
                        help='boundary type')
    parser.add_argument('--n_int_elt', type=int, default=15,
                        help='quadrature points per element')
    parser.add_argument('--n_int_edge', type=int, default=20,
                        help='quadrature points per edge')
    parser.add_argument('--poly_order', type=int, default=3,
                        help='polynomial order')
    parser.add_argument('--output_dir', type=str, default='./dg_results_2d',
                        help='output directory')

    args = parser.parse_args()

    demo = DG2DSolverDemo(output_dir=args.output_dir)

    print("\n" + "#" * 70)
    print("# Discontinuous Galerkin (DG) Method — 2D Solver")
    print("#" * 70)
    print(f"\nDevice: {device}")
    print(f"PyTorch: {torch.__version__}")
    print(f"NumPy: {np.__version__}\n")

    demo.solve_poisson2d(
        boundary_type=args.boundary,
        n_int_elt=args.n_int_elt,
        n_int_edge=args.n_int_edge,
        poly_order=args.poly_order
    )

    print("\n" + "#" * 70)
    print("# 2D Demo Completed")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    main()
