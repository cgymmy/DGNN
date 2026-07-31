"""
DGNN 2D Training with AdamW + CosineAnnealing
==============================================
Variant of run_2d.py: replaces L-BFGS + Adam with AdamW + CosineAnnealingLR.
All other logic (mesh, model, loss) is inherited unchanged from Poisson2d_dg.
"""

import argparse
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from problems.poisson2d_dg import Poisson2d_dg


class Poisson2d_dg_adamw(Poisson2d_dg):
    def __init__(self, lr: float = 1e-3, weight_decay: float = 1e-5,
                 maxiter: int = 20000, **kwargs):
        super().__init__(**kwargs)
        self.name = self.name + '_adamw'

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=maxiter, eta_min=1e-6
        )
        self.maxiter = maxiter

    def train(self):
        print(f'*********** Started training {self.name} (AdamW) ...... ***************')
        self.writer = SummaryWriter(f'./logs/poisson2d/AdamW')
        self.t = time.time()
        best_loss = float('inf')

        for i in range(self.maxiter):
            self.optimizer.zero_grad()
            loss, mse, mae = self.loss()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            if not torch.isnan(mse):
                self.writer.add_scalar("mse_vs_iter", mse, i)
                self.writer.add_scalar("mse_vs_time", mse, time.time() - self.t)

            if i % 100 == 0:
                print(f"Iter: {i:5d}, loss: {loss.item():.6f}, mse: {mse.item():.6f}, mae: {mae.item():.6f}")

            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(self.model.state_dict(), f'./models/DGNet/{self.name}.pth')

        print(f'Finished training in {time.time() - self.t:.4f} seconds')
        with open('train_times.txt', 'a') as f:
            f.write(f'{self.name}, {time.time() - self.t:.4f} seconds\n')


def main():
    parser = argparse.ArgumentParser(
        description='DGNN 2D Training with AdamW',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_2d_adamw.py --boundary regular --partition 0.05
  python run_2d_adamw.py --boundary polygon --lr 1e-3 --maxiter 30000
  python run_2d_adamw.py --boundary irregular --hidden_size 50 --num_layers 3
        """
    )
    parser.add_argument('--problem', type=str, default='poisson2d', choices=['poisson2d'])
    parser.add_argument('--boundary', type=str, default='regular',
                        choices=['regular', 'polygon', 'irregular'])
    parser.add_argument('--partition', type=float, default=0.01)
    parser.add_argument('--Nint_elt', type=int, default=15)
    parser.add_argument('--Nint_edge', type=int, default=20)
    parser.add_argument('--order', type=int, default=3)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_size', type=int, default=20)
    parser.add_argument('--act', type=str, default='tanh')
    parser.add_argument('--sigma_eq', type=int, default=1)
    parser.add_argument('--sigma_flux', type=int, default=1)
    parser.add_argument('--sigma_bd', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--maxiter', type=int, default=20000)

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print(f"2D Poisson DGNN Training with AdamW (boundary: {args.boundary})")
    print("=" * 60)

    P = Poisson2d_dg_adamw(
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
        lr=args.lr,
        weight_decay=args.weight_decay,
        maxiter=args.maxiter,
    )
    P.load()
    P.train()
    P.load()


if __name__ == '__main__':
    main()
