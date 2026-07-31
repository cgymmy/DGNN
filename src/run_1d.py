"""
DGNN 1D Training Entry Point
=============================

Supported problems:
1. 1D Poisson equation: -u'' = f(x), Dirichlet BC
2. 1D Burgers equation: ∂u/∂t + u·∂u/∂x + ν·∂²u/∂x² = 0, periodic BC

Notes:
- 1D problems have only two neighboring elements at each interface; numerical flux is simple.
- Each element = interval; basis functions defined on reference element [-1,1].
- Model input: (x,) or (x,t).
"""

import argparse

from problems.poisson1d_dg import Poisson1D_dg
from problems.burgers_dg import Burgers_dg


def main():
    parser = argparse.ArgumentParser(
        description='DGNN 1D Training Entry Point',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_1d.py --problem poisson1d
  python run_1d.py --problem burgers --N_x 30 --N_t 100
  python run_1d.py --problem poisson1d --deg 6 --N_x 50
        """
    )
    parser.add_argument('--problem', type=str, default='poisson1d',
                        choices=['poisson1d', 'burgers'],
                        help='problem type')
    # Mesh
    parser.add_argument('--N_x', type=int, default=25, help='number of spatial elements')
    parser.add_argument('--N_int', type=int, default=20, help='quadrature points per element')
    parser.add_argument('--deg', type=int, default=5, help='polynomial degree of test functions')
    # Burgers specific
    parser.add_argument('--N_t', type=int, default=50, help='number of time steps (Burgers only)')
    # Network
    parser.add_argument('--num_layers', type=int, default=2, help='number of layers')
    parser.add_argument('--hidden_size', type=int, default=20, help='hidden layer size')
    parser.add_argument('--act', type=str, default='tanh', help='activation function')
    # Training
    parser.add_argument('--adam_lr', type=float, default=1e-4, help='Adam learning rate')
    parser.add_argument('--maxiter', type=int, default=70000, help='max training iterations')
    parser.add_argument('--threshold', type=float, default=1e-6, help='convergence threshold (MAE)')

    args = parser.parse_args()

    if args.problem == 'poisson1d':
        print("\n" + "=" * 60)
        print("1D Poisson DGNN Training")
        print("=" * 60)
        P = Poisson1D_dg(
            N_x=args.N_x,
            N_int=args.N_int,
            deg=args.deg,
            act=args.act,
            adam_lr=args.adam_lr,
            maxiter=args.maxiter,
            convergence_threshold=args.threshold,
        )
        P.load()
        P.train()

    elif args.problem == 'burgers':
        print("\n" + "=" * 60)
        print("1D Burgers DGNN Training")
        print("=" * 60)
        P = Burgers_dg(
            N_x=args.N_x,
            N_t=args.N_t,
            Nint_x=args.N_int,
            deg=args.deg,
            num_layers=args.num_layers,
            hidden_size=args.hidden_size,
            act=args.act,
        )
        P.load()
        P.train()


if __name__ == '__main__':
    main()
