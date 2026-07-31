import os
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from config import device
from exact.burgers import burgers_exact
from nn.mlp import MLP


class Burgers_pinn:
    def __init__(self, method: str = 'PINN', num_layers: int = 4, hidden_size: int = 128, act: str = 'tanh') -> None:
        self.name = f'1dburgers_{method}_{num_layers}_{hidden_size}_{act}'
        self.method = method
        self.torch_dtype = torch.float64
        x = np.linspace(0, 2 * np.pi, 1000)
        t = np.linspace(0, 1.5, 100)
        xx, tt = np.meshgrid(x, t)
        self.mesh = np.stack([xx, tt], axis=-1)
        self.model = MLP(input_size=2, output_size=1, hidden_size=hidden_size, num_layers=num_layers, act=act).to(device)
        self.init = lambda x: torch.sin(x) + 1 / 2
        self.Adam = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.maxiter = 40000
        self.u_exact = torch.tensor(burgers_exact(self.mesh[:, :, 0], self.mesh[:, :, 1])).to(device)

    def loss(self):
        if self.method == 'PINN':
            return self.pinn()
        else:
            pass

    def pinn(self):
        mesh = torch.tensor(self.mesh, dtype=torch.float32, requires_grad=True).to(device)
        u = self.model(mesh)
        grad_u = torch.autograd.grad(u, mesh, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        ux = grad_u[..., 0]
        ut = grad_u[..., 1]
        eq_loss = torch.sum((ut + u.squeeze(-1) * ux) ** 2)
        init_loss = torch.sum((self.init(mesh[0, :, 0]) - u[0, :, 0]) ** 2)
        bd_loss = torch.sum((u[:, 0, 0] - u[:, -1, 0]) ** 2) + torch.sum((ux[0, :] - ux[-1, :]) ** 2)
        loss = eq_loss + init_loss + bd_loss
        mse = torch.mean((u.squeeze(-1) - self.u_exact) ** 2)
        mae = torch.max(torch.abs(u.squeeze(-1) - self.u_exact))
        return loss, mse, mae

    def train(self):
        print('*********** Started training ...... ***************')
        t = time.time()
        self.writer = SummaryWriter(f'./logs/burgers1d/PINN')
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
                torch.save(self.model.state_dict(), f'./models/PINN/{self.name}.pth')
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item():.6f}, mse = {mse.item():.6f}, mae = {mae.item():.6f}")
        self.writer.close()
        print(f'Finished training in {time.time() - t:.4f} seconds')

    def load(self):
        path = f'./models/{self.method}/{self.name}.pth'
        if os.path.exists(path):
            print("Loading saved model...")
            model_dict = torch.load(path)
            self.model.load_state_dict(model_dict)
            return True
        else:
            print("No saved model found. Need to train")
            return False
