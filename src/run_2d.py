"""
DGNN 2D Training Entry Point
=============================

Supported problems:
1. 2D Poisson equation: -Δu = f(x,y), Dirichlet BC (regular/polygon/irregular)

Notes:
- 2D requires integration over triangular elements; boundaries are closed curves in 2D.
- Mesh generation (Delaunay triangulation) is a key step distinct from 1D.
- Model input: (x,y), one independent MLP per element.
"""

import argparse

from problems.poisson2d_dg import Poisson2d_dg


def main():
    parser = argparse.ArgumentParser(
        description='DGNN 2D Training Entry Point',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_2d.py --problem poisson2d
  python run_2d.py --problem poisson2d --boundary irregular --partition 0.1 --num_layers 3 --hidden_size 50
  python run_2d.py --problem poisson2d --boundary regular --order 4
        """
    )
    parser.add_argument('--problem', type=str, default='poisson2d',
                        choices=['poisson2d'],
                        help='problem type')
    # Mesh / domain
    parser.add_argument('--boundary', type=str, default='polygon',
                        choices=['regular', 'polygon', 'irregular'],
                        help='boundary type')
    parser.add_argument('--partition', type=float, default=0.003,
                        help='min element size')
    parser.add_argument('--Nint_elt', type=int, default=5, help='quadrature points per element')
    parser.add_argument('--Nint_edge', type=int, default=5, help='quadrature points per edge')
    parser.add_argument('--order', type=int, default=2, help='polynomial order of test functions')
    # Network
    parser.add_argument('--num_layers', type=int, default=2, help='number of layers')
    parser.add_argument('--hidden_size', type=int, default=20, help='hidden layer size')
    parser.add_argument('--act', type=str, default='tanh', help='activation function')
    # Loss weights
    parser.add_argument('--sigma_eq', type=int, default=1, help='PDE residual loss weight')
    parser.add_argument('--sigma_flux', type=int, default=1, help='flux loss weight')
    parser.add_argument('--sigma_bd', type=int, default=1, help='boundary loss weight')

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print(f"2D Poisson DGNN Training (boundary: {args.boundary})")
    print("=" * 60)

    P = Poisson2d_dg(
        boundary_type=args.boundary,
        Nint_elt=args.Nint_elt,
        Nint_edge=args.Nint_edge,
        order=args.order,
        partition=args.partition,
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        act=args.act,
        sigma_eq=args.sigma_eq,
        sigma_bd=args.sigma_bd,
        sigma_flux=args.sigma_flux,
    )
    P.load()
    P.train()
    P.load()


if __name__ == '__main__':
    main()
