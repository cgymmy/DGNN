import os
import time
from math import pi

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from config import device
from exact.burgers import burgers_exact
from nn.mlp import MLP
from testfuncs.testfunc1d import TestFunction1D


class Burgers_hpVPINN:
    def __init__(self, N_x: int, N_t: int, Nint_x: int, deg: int,
                 num_layers: int = 2, hidden_size: int = 20, act: str = 'tanh') -> None:
        self.name = f'1dburgers_{N_x}_{N_t}_{Nint_x}_{deg}_{num_layers}_{hidden_size}_{act}'
        self.np_dtype = np.float64
        self.torch_dtype = torch.float64
        self.save_path = f'./models/DGNet/{self.name}.pth'

        self.a = 0.0
        self.b = 2.0 * pi
        self.t0 = 0.0
        self.T = 1.5

        self.N_x = N_x
        self.N_t = N_t
        self.Nint_x = Nint_x
        self.deg = deg

        self.testfunc = TestFunction1D(func_type='Legendre')

        self.x, self.xc, self.h, self.t, self.Mesh, self.xmesh, self.weights = self.get_mesh()
        mesh = self.Mesh.cpu().detach().numpy()
        self.v, self.dv = self.test_data()
        self.model = MLP(input_size=2, hidden_size=hidden_size, output_size=1,
                         num_layers=num_layers, act=act).to(device).to(self.torch_dtype)

        self.f = lambda x: x ** 2 / 2
        self.init = lambda x: torch.sin(x) + 1 / 2
        self.exact_init = self.init(self.xmesh)
        self.u_exact = torch.tensor(burgers_exact(mesh[:, :, :, 0], mesh[:, :, :, 1])).to(device)

        self.Lfbgs = torch.optim.LBFGS(self.model.parameters(), lr=1.0, max_iter=40000,
                                       max_eval=50000, history_size=50, tolerance_grad=1e-7,
                                       tolerance_change=1.0 * np.finfo(float).eps,
                                       line_search_fn='strong_wolfe')
        self.Adam = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.maxiter = 70000
        self.adamiter = 0
        self.iter = 0

    def test_data(self):
        v_list = []
        dv_list = []
        for i in range(self.deg + 1):
            v, dv = self.testfunc.get_value(x=self.xmesh, x_mid=self.xc[:, None], h=self.h[:, None], order=i)
            v_list.append(v)
            dv_list.append(dv)
        v = torch.stack(v_list, dim=0).to(device)
        dv = torch.stack(dv_list, dim=0).to(device)
        return v, dv

    def get_mesh(self):
        x = np.linspace(self.a, self.b, self.N_x + 1)
        t = np.linspace(self.t0, self.T, self.N_t + 1)
        xc = (x[:-1] + x[1:]) / 2.0
        h = (x[1:] - x[:-1])
        nodes, weights = np.polynomial.legendre.leggauss(self.Nint_x)

        mesh = 0.5 * (nodes[None, :] + 1) * h[:, None] + x[:-1, None]
        weights = 0.5 * weights[None, :] * h[:, None]
        xmesh = np.zeros((self.N_x, self.Nint_x + 2))
        xmesh[:, 1:-1] = mesh
        xmesh[:, 0] = x[:-1]
        xmesh[:, -1] = x[1:]
        Mesh = []
        for i in range(self.N_x):
            xx, tt = np.meshgrid(xmesh[i, :], t)
            Mesh.append(np.stack([xx, tt], axis=-1))
        Mesh = np.array(Mesh)
        return torch.tensor(x, dtype=self.torch_dtype).to(device), \
            torch.tensor(xc, dtype=self.torch_dtype).to(device), \
            torch.tensor(h, dtype=self.torch_dtype).to(device), \
            torch.tensor(t, dtype=self.torch_dtype).to(device), \
            torch.tensor(Mesh, dtype=self.torch_dtype).to(device), \
            torch.tensor(xmesh, dtype=self.torch_dtype).to(device), \
            torch.tensor(weights, dtype=self.torch_dtype).to(device)

    def loss(self):
        Mesh = self.Mesh.clone().detach().requires_grad_(True).to(device)
        u = self.model(Mesh)
        fu = self.f(u)
        gradu = torch.autograd.grad(u, Mesh, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        ux = gradu[..., 0]
        ut = gradu[..., 1]

        lfu = self.f(u[:, :, 0, 0])
        rfu = self.f(u[:, :, -1, 0])
        lv = self.v[:, :, 0]
        rv = self.v[:, :, -1]
        bd = rfu[None, ...] * rv[:, :, None] - lfu[None, ...] * lv[:, :, None]

        lux = ux[0, :, 0]
        rux = ux[-1, :, -1]
        ux = ux[:, :, 1:-1]
        ut = ut[:, :, 1:-1]

        # compute local loss
        Int = torch.sum((ut[None, ...] * self.v[:, :, None, 1:-1] - fu[None, :, :, 1:-1, 0] * self.dv[:, :, None, 1:-1]) * self.weights[None, :, None, :], dim=-1) + bd
        local_loss = torch.sum(Int ** 2)
        # compute init_loss
        init_loss = torch.sum((u[:, 0, :, 0] - self.exact_init) ** 2)
        # compute bd_loss
        bd_loss = torch.sum((u[0, :, 0, 0] - u[-1, :, -1, 0]) ** 2 + (rux - lux) ** 2)
        loss = local_loss + init_loss + bd_loss
        mse = torch.mean((u.squeeze(-1) - self.u_exact) ** 2)
        mae = torch.max(torch.abs(u.squeeze(-1) - self.u_exact))
        return loss, mse, mae

    def train(self):
        print('*********** Started training ...... ***************')
        t = time.time()
        self.writer = SummaryWriter(f'./logs/burgers1d/hpVPINN')
        loss, mse, mae = self.loss()
        best_loss = loss
        epoch = 0
        while epoch < self.maxiter:
            self.Adam.zero_grad()
            loss, mse, mae = self.loss()
            self.writer.add_scalar(f"mse_vs_iter", mse, epoch)
            self.writer.add_scalar(f"mse_vs_time", mse, time.time() - t)
            loss.backward()
            self.Adam.step()
            epoch += 1
            if loss < best_loss:
                best_loss = loss
                torch.save(self.model.state_dict(), f'./models/hpVPINN/{self.name}.pth')
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item():.6f}, mse = {mse.item():.6f}, mae = {mae.item():.6f}")
        self.writer.close()
        print(f'Finished training in {time.time() - t:.4f} seconds')

    def load(self):
        path = f'./models/hpVPINN/{self.name}.pth'
        if os.path.exists(path):
            print("Loading saved model...")
            model_dict = torch.load(path)
            self.model.load_state_dict(model_dict)
            return True
        else:
            print("No saved model found. Need to train")
            return False
