import argparse
import os
import sys
import time
from math import pi

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

try:
    from ..config import device
    from ..nn.dgnet import DGNet
    from ..testfuncs.testfunc1d import TestFunction1D
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import device
    from nn.dgnet import DGNet
    from testfuncs.testfunc1d import TestFunction1D


class Poisson1D_dg:
    def __init__(self, N_x: int, N_int: int, deg: int, act: str = 'tanh',
                 logs_dir: str = './logs/poisson1d/DGNet',
                 models_dir: str = './models/DGNet',
                 adam_lr: float = 1e-4, maxiter: int = 70000,
                 convergence_threshold: float = 1e-6) -> None:
        self.name = f'DG_1dpoisson_{N_x}_{N_int}_{deg}_{act}'
        self.a = 0.0
        self.b = 1.5
        self.np_dtype = np.float64
        self.torch_dtype = torch.float64
        self.N_x = N_x
        self.N_int = N_int
        self.deg = deg
        self.logs_dir = logs_dir
        self.models_dir = models_dir
        self.convergence_threshold = convergence_threshold

        self.testfunc = TestFunction1D(func_type='Polynomial')

        self.x, self.xc, self.h, self.Mesh, self.weights = self.get_mesh()
        self.v, self.dv = self.test_data()
        self.model = DGNet(num_modules=self.N_x, input_size=1, hidden_size=20, output_size=1,
                           num_layers=2, act=act).to(device).to(self.torch_dtype)
        # case 1
        # self.f = lambda x: 10 * torch.ones_like(x)
        # case 2
        self.w = 15 * pi
        self.f = lambda x: 2 * self.w * torch.sin(self.w * x) + self.w ** 2 * x * torch.cos(self.w * x)
        self.exact = lambda x: x * torch.cos(self.w * x)
        self.fu = self.f(self.Mesh[:, 1:-1]).squeeze(-1)

        self.Lfbgs = torch.optim.LBFGS(self.model.parameters(), lr=1., max_iter=50000,
                                       max_eval=50000, history_size=50, tolerance_grad=1e-6,
                                       tolerance_change=1.0 * np.finfo(float).eps,
                                       line_search_fn='strong_wolfe')
        self.lfbgsiter = 0
        self.Adam = torch.optim.Adam(self.model.parameters(), lr=adam_lr)
        self.maxiter = maxiter
        self.adamiter = 0

    def test_data(self):
        v_list = []
        dv_list = []
        for i in range(self.deg + 1):
            v, dv = self.testfunc.get_value(x=self.Mesh, x_mid=self.xc[:, None], h=self.h[:, None], order=i)
            v_list.append(v)
            dv_list.append(dv)
        v = torch.stack(v_list, dim=0).to(device)
        dv = torch.stack(dv_list, dim=0).to(device)
        return v, dv

    def get_mesh(self):
        x = np.linspace(self.a, self.b, self.N_x + 1, dtype=self.np_dtype)
        xc = (x[:-1] + x[1:]) / 2.0
        h = (x[1:] - x[:-1])
        nodes, weights = np.polynomial.legendre.leggauss(self.N_int)
        mesh = 0.5 * (nodes[None, :] + 1.) * h[:, None] + x[:-1, None]
        weights = 0.5 * weights[None, :] * h[:, None]
        Mesh = np.zeros((self.N_x, self.N_int + 2))
        Mesh[:, 1:-1] = mesh
        Mesh[:, 0] = x[:-1]
        Mesh[:, -1] = x[1:]
        return torch.tensor(x, dtype=self.torch_dtype).to(device), \
            torch.tensor(xc, dtype=self.torch_dtype).to(device), \
            torch.tensor(h, dtype=self.torch_dtype).to(device), \
            torch.tensor(Mesh, dtype=self.torch_dtype).to(device), \
            torch.tensor(weights, dtype=self.torch_dtype).to(device)

    def loss(self):
        Mesh = self.Mesh[..., None].clone().detach().requires_grad_(True).to(device)
        u = self.model(Mesh)
        gradu = torch.autograd.grad(u, Mesh, grad_outputs=torch.ones_like(u), create_graph=True)[0]

        lux = gradu[None, :, 0, 0]
        rux = gradu[None, :, -1, 0]
        lv = self.v[:, :, 0]
        rv = self.v[:, :, -1]
        bd = rux * rv - lux * lv

        flux_u = u[:-1, -1, 0] - u[1:, 0, 0]
        flux_ux = gradu[:-1, -1, 0] - gradu[1:, 0, 0]
        ux = gradu[:, 1:-1, 0]
        # compute local loss
        Int = torch.sum((ux[None, ...] * self.dv[:, :, 1:-1] - self.fu[None, ...] * self.v[:, :, 1:-1]) * self.weights[None, ...], dim=-1) - bd
        local_loss = torch.sum(Int ** 2)
        # compute boundary loss
        bd_loss = (u[0, 0, 0] ** 2 + u[-1, -1, 0] ** 2)
        # compute flux loss
        flux_loss = torch.sum((flux_u ** 2 + flux_ux ** 2))
        loss = local_loss + bd_loss + flux_loss
        mae = torch.max(abs(u - self.exact(Mesh)))
        mse = torch.mean(abs(u - self.exact(Mesh)) ** 2)
        return loss, mae, mse

    def exact_loss(self):
        x = self.Mesh
        u = self.exact(x)
        ux = torch.cos(self.w * x) - self.w * x * torch.sin(self.w * x)
        lux = ux[None, :, 0]
        rux = ux[None, :, -1]
        lv = self.v[:, :, 0]
        rv = self.v[:, :, -1]
        bd = rux * rv - lux * lv

        flux_u = u[:-1, -1] - u[1:, 0]
        flux_ux = ux[:-1, -1] - ux[1:, 0]
        ux = ux[:, 1:-1]
        Int = torch.sum((ux[None, ...] * self.dv[:, :, 1:-1] - self.fu[None, ...] * self.v[:, :, 1:-1]) * self.weights[None, ...], dim=-1) - bd
        local_loss = torch.sum(Int ** 2)
        bd_loss = (u[0, 0] ** 2 + u[-1, -1] ** 2)
        flux_loss = torch.sum((flux_u ** 2 + flux_ux ** 2))
        loss = local_loss + bd_loss + flux_loss
        return loss

    def loss_lfbgs(self):
        self.Lfbgs.zero_grad()
        loss, mae, mse = self.loss()
        loss.backward()
        self.writer.add_scalar(f"mse_vs_iter", mse, self.lfbgsiter)
        self.writer.add_scalar(f"mse_vs_time", mse, time.time() - self.t)
        self.lfbgsiter += 1
        if self.lfbgsiter % 500 == 0:
            print(f"LBFGS At iter: {self.lfbgsiter}, loss_train:{loss.item():.6f}, mae_train:{mae.item():.6f}, mse_train:{mse.item():.6f}")
        return loss

    def loss_adam(self):
        self.Adam.zero_grad()
        loss, mae, mse = self.loss()
        self.writer.add_scalar(f"mse_vs_iter", mse, self.lfbgsiter + self.adamiter)
        self.writer.add_scalar(f"mse_vs_time", mse, time.time() - self.t)
        self.adamiter += 1
        if self.adamiter % 500 == 0:
            print(f"Adam At iter: {self.adamiter}, loss_train:{loss.item():.6f}, mae_train:{mae.item():.6f}, mse_train:{mse.item():.6f}")
        return loss, mae, mse

    def train(self):
        t_start = time.time()
        print('*********** Started training ...... ***************')
        self.t = time.time()
        # Create directories if they don't exist
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        self.writer = SummaryWriter(self.logs_dir)
        self.Lfbgs.step(self.loss_lfbgs)
        model_path = os.path.join(self.models_dir, f'{self.name}.pth')
        torch.save(self.model.state_dict(), model_path)
        loss, mae, mse = self.loss_adam()
        best_loss = mae
        while mae > self.convergence_threshold:
            loss.backward()
            self.Adam.step()
            loss, mae, mse = self.loss_adam()
            if mae < best_loss:
                best_loss = mae
                torch.save(self.model.state_dict(), model_path)
            if self.adamiter + self.lfbgsiter > self.maxiter:
                break
        self.adamiter = 0
        self.writer.close()
        print(f'Finished training in {time.time() - t_start:.4f} seconds')

    def load(self, path=None):
        if path is None:
            path = os.path.join(self.models_dir, f'{self.name}.pth')
        if os.path.exists(path):
            print("Loading saved model...")
            model_dict = torch.load(path)
            self.model.load_state_dict(model_dict)
            return True
        else:
            print("No saved model found. Need to train")
            return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='1D Poisson Problem with DGNet')
    parser.add_argument('--N_x', type=int, default=25, help='Number of elements')
    parser.add_argument('--N_int', type=int, default=20, help='Number of integration points')
    parser.add_argument('--deg', type=int, default=5, help='Degree of test functions')
    parser.add_argument('--act', type=str, default='tanh', help='Activation function')
    parser.add_argument('--logs_dir', type=str, default='./logs/poisson1d/DGNet', help='Path to save logs')
    parser.add_argument('--models_dir', type=str, default='./models/DGNet', help='Path to save models')
    parser.add_argument('--adam_lr', type=float, default=1e-4, help='Adam optimizer learning rate')
    parser.add_argument('--maxiter', type=int, default=70000, help='Maximum iterations for training')
    parser.add_argument('--convergence_threshold', type=float, default=1e-6, help='MAE threshold for convergence')
    parser.add_argument('--train', type=bool, default=True, help='Whether to train the model')
    parser.add_argument('--load_path', type=str, default=None, help='Path to load a saved model')
    args = parser.parse_args()
    
    P = Poisson1D_dg(
        N_x=args.N_x, 
        N_int=args.N_int, 
        deg=args.deg,
        act=args.act,
        logs_dir=args.logs_dir,
        models_dir=args.models_dir,
        adam_lr=args.adam_lr,
        maxiter=args.maxiter,
        convergence_threshold=args.convergence_threshold
    )
    
    load_path = args.load_path if args.load_path is not None else os.path.join(args.models_dir, f'{P.name}.pth')
    
    if args.train:
        P.load(load_path)
        P.train()
    elif P.load(load_path) and not args.train:
        print("train loss: ", P.loss())

