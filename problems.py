import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from triangle_gauss import *
from math import pi
import numpy as np
import scipy
import time
import os
from triangle import triangulate
from scipy.interpolate import griddata
from matplotlib.path import Path
from torch.utils.tensorboard import SummaryWriter
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from utilities import *
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class Possion2d_dg:
    def __init__(self, boundary_type: str = 'regular', Nint_elt: int=15, Nint_edge: int = 20, order: int=3, partition: float=0.2,\
                  num_layers: int=2, hidden_size: int=50, act: str='tanh',\
                  sigma_eq: int=1, sigma_bd: int=1, sigma_flux: int=1) -> None:
        self.name = f'DG_2dPossion_{boundary_type}_{Nint_elt}_{Nint_edge}_{order}_{partition}_{num_layers}_{hidden_size}_{act}'
        self.genmesh = GenMesh2D(boundary_type=boundary_type, Nint_edge=Nint_edge, Nint_elt=Nint_elt, param=f'pq30a{partition}e')
        self.torch_type = torch.float64
        self.sigma_eq = sigma_eq
        self.sigma_bd = sigma_bd
        self.sigma_flux = sigma_flux
        #
        self.Nelt = self.genmesh.Nelt
        self.K = int(self.Nelt * 0.5)
        #
        self.test_func = TestFunction2D()
        self.elt_int, self.elt_weights, self.edges_int, self.mesh_edges_w, self.mesh_normvec, self.Mesh, self.ref_Mesh, self.inv_matrix = self.genmesh.get_mesh()
        self.num_eltinnerp = self.genmesh.num_eltp
        self.num_eltbdp = self.genmesh.num_edgep
        # case 1
        if boundary_type == 'regular':
            self.f = lambda x, y: ((4*x**4-4*x**3+10*x**2-6*x+2)*(y-y**2)*torch.exp(x**2+y**2) + (4*y**4-4*y**3+10*y**2-6*y+2)*(x-x**2)*torch.exp(x**2+y**2))*10
            self.exact = lambda x, y: 10 * x*(1-x)*y*(1-y)*torch.exp(x**2 + y**2)
            self.fxy = self.f(self.elt_int[..., 0], self.elt_int[..., 1])
            self.u_exact = self.exact(self.Mesh[..., 0], self.Mesh[..., 1]).unsqueeze(-1)
        # case 2
        elif boundary_type == 'polygon':
            self.f = lambda x, y: torch.ones_like(x) * 10
            self.fxy = self.f(self.elt_int[..., 0], self.elt_int[..., 1])
            self.get_exact_polygon()
            
            
        #
        self.v_elt_inner, self.v_elt_bd, self.dv_elt_inner = self.get_test(order=order)
        #
        self.model = DGNet(num_modules=self.Nelt, input_size=2, hidden_size=hidden_size, output_size=1, num_layers=num_layers, act=act).to(device).to(self.torch_type)
        #
        self.Lfbgs = torch.optim.LBFGS(self.model.parameters(), lr=1.0, max_iter=50000, 
                                           max_eval=50000, history_size=50, tolerance_grad=1e-7, 
                                           tolerance_change=1.0 * np.finfo(float).eps, 
                                           line_search_fn='strong_wolfe'); self.lfbgsiter = 0
        self.Adam = torch.optim.Adam(self.model.parameters(), lr=1e-4); self.maxiter = 10000; self.adamiter = 0
    def get_test(self, order: int=2):
        v, dv = self.test_func.Poly(mesh=self.ref_Mesh, order=order)
        
        v_elt_inner, v_elt_bd = v[:, :self.num_eltinnerp], v[:, self.num_eltinnerp:].reshape(-1, 3, self.num_eltbdp)
        dv_elt_inner = dv[:, :self.num_eltinnerp, :]
        dv_elt_inner = torch.matmul(self.inv_matrix[None, :, None, :, :], dv_elt_inner[:, None, :, :].unsqueeze(-1)).squeeze(-1)
        return v_elt_inner, v_elt_bd, dv_elt_inner
    
    def get_exact_polygon(self):
        Mesh = self.Mesh.clone().detach().cpu().numpy()
        exact_dict = np.load('./exact2dpolygon.npz')
        x = exact_dict['x']; y = exact_dict['y']
        u_exact = exact_dict['u_exact']
        comsol_mesh = np.array([x, y]).T
        comsol_u = griddata(comsol_mesh, u_exact, Mesh, method='cubic')
        self.u_exact = torch.from_numpy(comsol_u).unsqueeze(-1).to(device)

    
    def loss(self):
        Mesh = self.Mesh.clone().detach().requires_grad_(True).to(device)
        u = self.model(Mesh)
        gradu = torch.autograd.grad(u, Mesh, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        # compute local loss
        u_inner, u_bd = u[:, :self.num_eltinnerp, 0], u[:, self.num_eltinnerp:, 0].reshape(self.Nelt, 3, self.num_eltbdp)
        du_inner, du_bd = gradu[:, :self.num_eltinnerp, :], gradu[:, self.num_eltinnerp:, :].reshape(self.Nelt, 3, self.num_eltbdp, 2)
        Int = torch.sum((torch.sum(du_inner[None, ...] * self.dv_elt_inner, dim=-1) - self.fxy[None, ...] * self.v_elt_inner[:, None, :]) * self.elt_weights[None, ...], dim=-1)
        bd = torch.sum(du_bd[None, ...] * self.mesh_normvec[None, :, :, None, :], dim=-1) * self.v_elt_bd[:, None,:, :] * self.mesh_edges_w[None, ...]
        Int = Int - torch.sum(torch.sum(bd, dim=-1), dim=-1) 
        # local_loss = torch.sum(Int**2)
        local_info = torch.sum(Int**2, dim=0)
        local_loss = torch.sum(torch.topk(local_info, k=self.K, largest=True)[0])
        # compute flux loss
        flux_u = u_bd[self.genmesh.inner_label[:, 0, 0], self.genmesh.inner_label[:, 0, 1], :] - torch.flip(u_bd[self.genmesh.inner_label[:, 1, 0], self.genmesh.inner_label[:, 1, 1], :], dims=[1])

        flux_du = (du_bd[self.genmesh.inner_label[:, 0, 0], self.genmesh.inner_label[:, 0, 1], :, :] - torch.flip(du_bd[self.genmesh.inner_label[:, 1, 0], self.genmesh.inner_label[:, 1, 1], :, :], dims=[1]))**2
        flux_loss = torch.sum(flux_u ** 2) + torch.sum(flux_du)
        # compute boundary loss
        bd_loss = torch.sum(u_bd[self.genmesh.bd_label[:, 0, 0], self.genmesh.bd_label[:, 0, 1], :] ** 2)
        # if self.lfbgsiter % 100 == 0:
        #     print(f"LBFGS At iter: {self.lfbgsiter}, local_loss:{local_loss.item():.6f}, flux_loss:{flux_loss.item():.6f}, bd_loss:{bd_loss.item():.6f}")
        loss = self.sigma_eq * local_loss + self.sigma_flux * flux_loss + self.sigma_bd * bd_loss
        mse = torch.mean((u - self.u_exact)**2)
        mae = torch.max(torch.abs(u - self.u_exact))
        return loss, mse, mae
    
    def exact_loss(self):
        Mesh = self.Mesh.clone().detach().requires_grad_(True).to(device)
        u = self.exact(Mesh[..., 0], Mesh[..., 1]).unsqueeze(-1)
        ux = lambda x, y: ((1-2*x+2*x**2-2*x**3)*y*(1-y)*torch.exp(x**2 + y**2))*10
        uy = lambda x, y: ((1-2*y+2*y**2-2*y**3)*x*(1-x)*torch.exp(x**2 + y**2))*10
        gradu = torch.stack((ux(Mesh[..., 0], Mesh[..., 1]), uy(Mesh[..., 0], Mesh[..., 1])), dim=-1)
        # compute local loss
        u_inner, u_bd = u[:, :self.num_eltinnerp, 0], u[:, self.num_eltinnerp:, 0].reshape(self.Nelt, 3, self.num_eltbdp)
        du_inner, du_bd = gradu[:, :self.num_eltinnerp, :], gradu[:, self.num_eltinnerp:, :].reshape(self.Nelt, 3, self.num_eltbdp, 2)
        Int = torch.sum((torch.sum(du_inner[None, ...] * self.dv_elt_inner, dim=-1) - self.fxy[None, ...] * self.v_elt_inner[:, None, :]) * self.elt_weights[None, ...], dim=-1)
        bd = torch.sum(du_bd[None, ...] * self.mesh_normvec[None, :, :, None, :], dim=-1) * self.v_elt_bd[:, None,:, :] * self.mesh_edges_w[None, ...]
        Int = Int - torch.sum(torch.sum(bd, dim=-1), dim=-1) 
        local_loss = torch.sum(Int**2)
        # compute flux loss
        flux_u = u_bd[self.genmesh.inner_label[:, 0, 0], self.genmesh.inner_label[:, 0, 1], :] - torch.flip(u_bd[self.genmesh.inner_label[:, 1, 0], self.genmesh.inner_label[:, 1, 1], :], dims=[1])

        flux_du = (du_bd[self.genmesh.inner_label[:, 0, 0], self.genmesh.inner_label[:, 0, 1], :, :] - torch.flip(du_bd[self.genmesh.inner_label[:, 1, 0], self.genmesh.inner_label[:, 1, 1], :, :], dims=[1]))**2
        flux_loss = torch.sum(flux_u ** 2) + torch.sum(flux_du)
        # compute boundary loss
        bd_loss = torch.sum(u_bd[self.genmesh.bd_label[:, 0, 0], self.genmesh.bd_label[:, 0, 1], :] ** 2)
        loss = self.sigma_eq * local_loss + self.sigma_flux * flux_loss + self.sigma_bd * bd_loss
        return loss
    
    def loss_lfbgs(self):
        self.Lfbgs.zero_grad()
        loss, mse, mae = self.loss()
        self.writer.add_scalar(f"mse_vs_iter", mse, self.lfbgsiter + self.adamiter)
        self.writer.add_scalar(f"mse_vs_time", mse, time.time() - self.t)
        loss.backward()
        self.lfbgsiter += 1
        if self.lfbgsiter % 100 == 0:
            print(f"LBFGS At iter: {self.lfbgsiter}, loss_train:{loss.item():.6f}, mse:{mse.item():.6f}, mae:{mae.item():.6f}")
        return loss
    
    def loss_adam(self):
        self.Adam.zero_grad()
        loss, mse, mae = self.loss()
        self.writer.add_scalar(f"mse_vs_iter", mse, self.lfbgsiter + self.adamiter)
        self.writer.add_scalar(f"mse_vs_time", mse, time.time() - self.t)
        self.adamiter += 1
        if self.adamiter % 100 == 0:
            print(f"Adam At iter: {self.adamiter}, loss_train:{loss.item():.6f}, mse:{mse.item():.6f}, mae:{mae.item():.6f}")
        return loss
    def train(self):
        print(f'*********** Started training {self.name}...... ***************')
        self.writer = SummaryWriter(f'./logs/possion2d/DGNet')
        self.t = time.time()
        self.Lfbgs.step(self.loss_lfbgs)
        torch.save(self.model.state_dict(), f'./models/DGNet/{self.name}.pth')
        loss = self.loss_adam()
        best_loss = loss
        while loss > 1e-5:
            loss.backward()
            self.Adam.step()
            loss = self.loss_adam()
            if loss < best_loss:
                best_loss = loss
                torch.save(self.model.state_dict(), f'./models/DGNet/{self.name}.pth')
            if self.adamiter > self.maxiter:
                break
        self.adamiter = 0
        print(f'Finished training in {time.time()-self.t:.4f} seconds')
        with open('train_times.txt', 'a') as f:
            f.write(f'{self.name}, {time.time()-self.t:.4f} seconds\n')
    def load(self):
        path = f'./models/DGNet/{self.name}.pth'
        if os.path.exists(path):
            print("Loading saved model...")
            model_dict = torch.load(path)
            self.model.load_state_dict(model_dict)
            return True
        else:
            print("No saved model found. Need to train")
            return False
