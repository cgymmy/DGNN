import os
import time

import numpy as np
import torch
from matplotlib.path import Path
from scipy.interpolate import griddata
from torch.utils.tensorboard import SummaryWriter

from config import device
from mesh.mesh2d import GenMesh2D
from nn.mlp import MLP
from nn.resnet import ResNet


class Poisson2d_base:
    def __init__(self, boundary_type: str = 'regular', N_points: int = 100, Nint_edge: int = 20,
                 method: str = 'PINN', num_layers: int = 3, hidden_size: int = 50, act: str = 'tanh') -> None:
        self.name = f'2dpoisson_{boundary_type}_{N_points}_{Nint_edge}_{num_layers}_{hidden_size}_{act}'
        self.genmesh = GenMesh2D(boundary_type=boundary_type, Nint_edge=20, Nint_elt=15, param=f'pq30a0.05e')

        self.Nelt = self.genmesh.Nelt
        self.method = method
        self.boundary_type = boundary_type
        self.Nint_edge = Nint_edge

        self.inner_p, self.edges_p = self.get_mesh(N_points, Nint_edge)
        self.Mesh = torch.cat((self.inner_p, self.edges_p), dim=0)
        self.N = self.inner_p.shape[0]
        if method == 'PINN':
            _, _, _, _, _, self.Mesh_test, _, _ = self.genmesh.get_mesh()
            self.model = MLP(input_size=2, hidden_size=hidden_size, output_size=1,
                             num_layers=num_layers, act=act).to(device).to(torch.float64)
        elif method == 'DeepRitz':
            self.Mesh_test, _, _, _, _, _, _, _ = self.genmesh.get_mesh()
            self.model = ResNet(input_size=2, hidden_size=hidden_size, output_size=1,
                                num_layers=num_layers, act=act).to(device).to(torch.float64)

        # case 1
        if boundary_type == 'regular':
            self.f = lambda x, y: ((4 * x ** 4 - 4 * x ** 3 + 10 * x ** 2 - 6 * x + 2) * (y - y ** 2) * torch.exp(x ** 2 + y ** 2) + (4 * y ** 4 - 4 * y ** 3 + 10 * y ** 2 - 6 * y + 2) * (x - x ** 2) * torch.exp(x ** 2 + y ** 2)) * 10
            self.exact = lambda x, y: 10 * x * (1 - x) * y * (1 - y) * torch.exp(x ** 2 + y ** 2)
            self.fxy = self.f(self.Mesh[..., 0], self.Mesh[..., 1])
            self.u_exact = self.exact(self.Mesh[..., 0], self.Mesh[..., 1]).unsqueeze(-1)
        # case 2
        elif boundary_type == 'polygon':
            self.f = lambda x, y: torch.ones_like(x) * 10
            self.fxy = self.f(self.Mesh[..., 0], self.Mesh[..., 1])
            self.get_exact_polygon()

        self.Adam = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.maxiter = 20000
        self.adamiter = 0

    def get_exact_polygon(self):
        Mesh = self.Mesh_test.clone().detach().cpu().numpy()
        exact_dict = np.load('./data/exact2dpolygon.npz')
        x = exact_dict['x']
        y = exact_dict['y']
        u_exact = exact_dict['u_exact']
        comsol_mesh = np.array([x, y]).T
        comsol_u = griddata(comsol_mesh, u_exact, Mesh, method='cubic')
        self.u_exact = torch.from_numpy(comsol_u).unsqueeze(-1).to(device)

    def get_mesh(self, N_points: int, Nint_edge: int):
        v = np.array(self.genmesh.vertices)
        e = np.array(self.genmesh.segments)
        t = np.linspace(0, 1, Nint_edge)
        x = (v[e[:, 0]] - v[e[:, 1]])[:, None, :] * t[None, :, None] + v[e[:, 1]][:, None, :]
        edges_p = torch.tensor(x.reshape(-1, 2)).to(device)
        if self.boundary_type == 'regular':
            x = np.linspace(0, 1, N_points)
            y = np.linspace(0, 1, N_points)
            X, Y = np.meshgrid(x, y)
            Mesh = np.vstack([X.ravel(), Y.ravel()]).T
        elif self.boundary_type == 'polygon':
            vtx = np.array(self.genmesh.vertices)
            path = Path(vtx)
            x = np.linspace(-1, 1, N_points)
            y = np.linspace(-0.9, 1.1, N_points)
            X, Y = np.meshgrid(x, y)
            mesh_all = np.vstack([X.ravel(), Y.ravel()]).T
            mask = path.contains_points(mesh_all)
            Mesh = mesh_all[mask]
        Mesh = torch.tensor(Mesh).to(device)
        return Mesh, edges_p

    def pinn(self):
        Mesh = self.Mesh.clone().detach().requires_grad_(True).to(device)
        u = self.model(Mesh)
        gradu = torch.autograd.grad(u, Mesh, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        uxx = torch.autograd.grad(gradu[..., 0], Mesh, grad_outputs=torch.ones_like(gradu[..., 0]), create_graph=True)[0][..., 0]
        uyy = torch.autograd.grad(gradu[..., 1], Mesh, grad_outputs=torch.ones_like(gradu[..., 1]), create_graph=True)[0][..., 1]
        eq_loss = torch.sum((uxx + uyy + self.f(Mesh[..., 0], Mesh[..., 1])) ** 2)
        u_edge = u[self.N:, :]
        bd_loss = torch.sum(u_edge ** 2)
        loss = eq_loss + 100 * bd_loss
        u_test = self.model(self.Mesh_test)
        mse = torch.mean((u_test - self.u_exact) ** 2)
        mae = torch.max(torch.abs(u_test - self.u_exact))
        return loss, mse, mae

    def deepritz(self):
        Mesh = self.Mesh.clone().detach().requires_grad_(True).to(device)
        u = self.model(Mesh)
        gradu = torch.autograd.grad(u, Mesh, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        eq_loss = torch.sum(0.5 * (gradu[..., 0] ** 2 + gradu[..., 1] ** 2) - self.f(Mesh[..., 0], Mesh[..., 1]) * u.squeeze(-1))
        u_edge = u[self.N:, :]
        bd_loss = torch.sum(u_edge ** 2)
        loss = eq_loss + 500 * bd_loss
        u_test = self.model(self.Mesh_test)
        mse = torch.mean((u_test - self.u_exact) ** 2)
        mae = torch.max(torch.abs(u_test - self.u_exact))
        return loss, mse, mae

    def loss(self):
        if self.method == 'PINN':
            return self.pinn()
        elif self.method == 'DeepRitz':
            return self.deepritz()
        else:
            raise ValueError(f"Invalid method: {self.method}")

    def train(self):
        print('*********** Started training ...... ***************')
        t = time.time()
        self.writer = SummaryWriter(f'./logs/poisson2d/{self.method}')
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
                torch.save(self.model.state_dict(), f'./models/{self.method}/{self.name}.pth')
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
