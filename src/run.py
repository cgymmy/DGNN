import argparse

from problems.poisson2d_dg import Poisson2d_dg


def main():
    parser = argparse.ArgumentParser(description="DGNN training entrypoint")
    parser.add_argument("--partition", type=float, default=0.05, help="min size of element")
    parser.add_argument("--num_layers", type=int, default=2, help="number of layers")
    parser.add_argument("--hidden_size", type=int, default=20, help="hidden size")
    args = parser.parse_args()

    P_dgnet = Poisson2d_dg(
        boundary_type='polygon',
        partition=args.partition,
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        act='tanh',
    )
    for _ in range(2):
        P_dgnet.train()
        P_dgnet.load()


if __name__ == '__main__':
    main()
