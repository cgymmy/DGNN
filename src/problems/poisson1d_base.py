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
    from ..nn.mlp import MLP
    from ..nn.resnet import ResNet
    from ..testfuncs.testfunc1d import TestFunction1D
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import device
    from nn.mlp import MLP
    from nn.resnet import ResNet
    from testfuncs.testfunc1d import TestFunction1D


class Poisson1D_base:
    def __init__(self, method: str = 'PINN', N_x: int = 10, N_int: int = 10, deg: int = 3,
                 num_layers: int = 2, input_size: int = 1, hidden_size: int = 50, output_size: int = 1,
                 act: str = 'tanh', logs_dir: str = './logs/Poisson1D',
                 models_dir: str = './models', adam_lr: float = 1e-4,
                 maxiter: int = 40000, convergence_threshold: float = 1e-5) -> None:
        self.name = f'1dpoisson_{method}_{N_x}_{N_int}_{deg}_{num_layers}_{input_size}_{hidden_size}_{output_size}_{act}'

        self.a = 0.0
        self.b = 1.5
        self.np_dtype = np.float64
        self.torch_dtype = torch.float32
        self.method = method
        self.logs_dir = logs_dir
        self.models_dir = models_dir
        self.convergence_threshold = convergence_threshold

        self.N_x = N_x
        self.N_int = N_int
        self.deg = deg
        # case 1
        # self.f = lambda x: 10 * torch.ones_like(x)
        # case 2
        self.w = 3 * pi
        self.f = lambda x: 2 * self.w * torch.sin(self.w * x) + self.w ** 2 * x * torch.cos(self.w * x)
        self.exact = lambda x: x * torch.cos(self.w * x)

        if self.method == "PINN":
            self.mesh = torch.linspace(self.a, self.b, self.N_x + 1, device=device).view(-1, 1)
            self.model = MLP(input_size=input_size, hidden_size=hidden_size, output_size=output_size,
                             num_layers=num_layers, act=act).to(device).to(self.torch_dtype)

        elif self.method == "DeepRitz":
            nodes, weights = np.polynomial.legendre.leggauss(self.N_x)
            h = self.b - self.a
            mesh = 0.5 * (nodes + 1.) * h + self.a
            self.weights = torch.tensor(0.5 * weights * h).to(device)
            self.mesh = torch.tensor(mesh, device=device, dtype=self.torch_dtype).view(-1, 1)
            self.model = ResNet(input_size=input_size, hidden_size=hidden_size, output_size=output_size,
                                num_layers=num_layers, act=act).to(device).to(self.torch_dtype)

        elif self.method == "hpVPINN":
            self.testfunc = TestFunction1D(func_type='Legendre')
            self.x, self.xc, self.h, self.mesh, self.weights = self.get_mesh()
            self.v, self.dv = self.test_data()
            self.model = MLP(input_size=input_size, hidden_size=hidden_size, output_size=output_size,
                             num_layers=num_layers, act=act).to(device).to(self.torch_dtype)

        else:
            raise ValueError(f"Invalid method: {self.method}")

        self.fx = self.f(self.mesh)
        self.exactu = self.exact(self.mesh)

        self.Lfbgs = torch.optim.LBFGS(self.model.parameters(), lr=1.0, max_iter=20000,
                                       max_eval=50000, history_size=50, tolerance_grad=1e-6,
                                       tolerance_change=1.0 * np.finfo(float).eps,
                                       line_search_fn='strong_wolfe')
        self.lfbgsiter = 0
        self.Adam = torch.optim.Adam(self.model.parameters(), lr=adam_lr)
        self.maxiter = maxiter
        self.adamiter = 0

    def loss(self):
        if self.method == 'PINN':
            return self.pinn()
        elif self.method == 'DeepRitz':
            return self.deepritz()
        elif self.method == 'hpVPINN':
            return self.hpvpinn()
        else:
            raise ValueError(f"Invalid method: {self.method}")

    def pinn(self):
        self.mesh.requires_grad = True
        u = self.model(self.mesh)
        u_x = torch.autograd.grad(u, self.mesh, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, self.mesh, torch.ones_like(u_x), create_graph=True)[0]

        eq_loss = torch.mean((u_xx + self.fx) ** 2)
        bd_loss = torch.mean(u[[0, -1], 0] ** 2)
        loss = eq_loss + 10 * bd_loss
        mae = torch.max(abs(self.exactu - u))
        mse = torch.mean(abs(self.exactu - u) ** 2)
        return loss, mae, mse

    def deepritz(self):
        self.mesh.requires_grad = True
        u = self.model(self.mesh)
        u_x = torch.autograd.grad(u, self.mesh, torch.ones_like(u), create_graph=True)[0]
        energy = torch.sum((0.5 * u_x ** 2 - self.fx * u) * self.weights[:, None])
        xb = self.mesh[[0, -1]]
        output_b = u[[0, -1]]
        exact_b = self.exact(xb)
        bd_loss = torch.mean((output_b - exact_b) ** 2)
        loss = energy + 500 * bd_loss
        mae = torch.max(abs(self.exactu - u))
        mse = torch.mean(abs(self.exactu - u) ** 2)
        return loss, mae, mse

    def hpvpinn(self):
        Mesh = self.mesh[..., None].clone().detach().requires_grad_(True).to(device)
        u = self.model(Mesh)
        gradu = torch.autograd.grad(u, Mesh, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        lux = gradu[None, :, 0, 0]
        rux = gradu[None, :, -1, 0]
        lv = self.v[:, :, 0]
        rv = self.v[:, :, -1]
        bd = rux * rv - lux * lv

        ux = gradu[:, 1:-1, 0]
        Int = torch.sum((ux[None, ...] * self.dv[:, :, 1:-1] - self.fx[None, :, 1:-1] * self.v[:, :, 1:-1]) * self.weights[None, ...], dim=-1) - bd
        local_loss = torch.sum(Int ** 2)
        bd_loss = (u[0, 0, 0] ** 2 + u[-1, -1, 0] ** 2)
        loss = local_loss + 100 * bd_loss
        mae = torch.max(abs(u - self.exact(Mesh)))
        mse = torch.mean(abs(u - self.exact(Mesh)) ** 2)
        return loss, mae, mse

    def train(self):
        print('*********** Started training ...... ***************')
        t = time.time()
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(os.path.join(self.models_dir, self.method), exist_ok=True)
        self.writer = SummaryWriter(self.logs_dir)
        loss, mae, mse = self.loss()
        best_loss = loss
        epoch = 0
        model_path = os.path.join(self.models_dir, self.method, f'{self.name}.pth')
        while epoch < self.maxiter:
            self.Adam.zero_grad()
            loss, mae, mse = self.loss()
            self.writer.add_scalar(f"mse_vs_iter", mse, epoch)
            self.writer.add_scalar(f"mse_vs_time", mse, time.time() - t)
            loss.backward()
            self.Adam.step()
            epoch += 1
            if loss < best_loss:
                best_loss = loss
                torch.save(self.model.state_dict(), model_path)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item():.6f}, mse = {mse.item():.6f}, mae = {mae.item():.6f}")
            if mae <= self.convergence_threshold:
                break
        self.writer.close()
        print(f'Finished training in {time.time() - t:.4f} seconds')

    def load(self, path=None):
        if path is None:
            path = os.path.join(self.models_dir, self.method, f'{self.name}.pth')
        if os.path.exists(path):
            print("Loading saved model...")
            model_dict = torch.load(path)
            self.model.load_state_dict(model_dict)
            return True
        else:
            print("No saved model found. Need to train")
            return False

    def test_data(self):
        v_list = []
        dv_list = []
        for i in range(self.deg + 1):
            v, dv = self.testfunc.get_value(x=self.mesh, x_mid=self.xc[:, None], h=self.h[:, None], order=i)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='1D Poisson Problem base model')
    parser.add_argument('--method', type=str, default='PINN', choices=['PINN', 'DeepRitz', 'hpVPINN'], help='Training method')
    parser.add_argument('--N_x', type=int, default=10, help='Number of spatial intervals')
    parser.add_argument('--N_int', type=int, default=10, help='Number of integration points')
    parser.add_argument('--deg', type=int, default=3, help='Degree of test functions')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of hidden layers')
    parser.add_argument('--input_size', type=int, default=1, help='Input dimension')
    parser.add_argument('--hidden_size', type=int, default=50, help='Hidden layer width')
    parser.add_argument('--output_size', type=int, default=1, help='Output dimension')
    parser.add_argument('--act', type=str, default='tanh', help='Activation function')
    parser.add_argument('--logs_dir', type=str, default='./logs/Poisson1D', help='Directory for tensorboard logs')
    parser.add_argument('--models_dir', type=str, default='./models', help='Directory for saved models')
    parser.add_argument('--adam_lr', type=float, default=1e-4, help='Adam learning rate')
    parser.add_argument('--maxiter', type=int, default=40000, help='Maximum Adam iterations')
    parser.add_argument('--convergence_threshold', type=float, default=1e-5, help='Stopping threshold for MAE')
    parser.add_argument('--train', type=bool, default=True, help='Whether to train the model')
    parser.add_argument('--load_path', type=str, default=None, help='Custom model checkpoint path')
    args = parser.parse_args()

    P = Poisson1D_base(
        method=args.method,
        N_x=args.N_x,
        N_int=args.N_int,
        deg=args.deg,
        num_layers=args.num_layers,
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        output_size=args.output_size,
        act=args.act,
        logs_dir=args.logs_dir,
        models_dir=args.models_dir,
        adam_lr=args.adam_lr,
        maxiter=args.maxiter,
        convergence_threshold=args.convergence_threshold,
    )

    load_path = args.load_path if args.load_path is not None else os.path.join(args.models_dir, args.method, f'{P.name}.pth')

    if args.train:
        P.load(load_path)
        P.train()
    elif P.load(load_path) and not args.train:
        print("train loss: ", P.loss())
