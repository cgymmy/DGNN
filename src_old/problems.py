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

class Poisson1D_dg:
    def __init__(self, N_x:int, N_int:int, deg:int, act:str='tanh') -> None:
        #
        self.name = f'DG_1dpoisson_{N_x}_{N_int}_{deg}_{act}'
        #
        self.a = 0.0
        self.b = 1.5
        self.np_dtype = np.float64
        self.torch_dtype = torch.float64
        #
        self.N_x = N_x
        self.N_int = N_int
        self.deg = deg
        #
        self.testfunc = TestFunction1D(func_type='Polynomial')
        #
        self.x, self.xc, self.h, self.Mesh, self.weights = self.get_mesh()
        self.v, self.dv = self.test_data()
        self.model = DGNet(num_modules=self.N_x, input_size=1, hidden_size=20, output_size=1, num_layers=2, act=act).to(device).to(self.torch_dtype)
        # case 1
        # self.f = lambda x: 10 * torch.ones_like(x)
        # case 2
        self.w = 15*pi
        self.f = lambda x: 2 * self.w * torch.sin(self.w * x) + self.w**2 * x * torch.cos(self.w * x)
        self.exact = lambda x: x * torch.cos(self.w * x)
        self.fu = self.f(self.Mesh[:, 1:-1]).squeeze(-1)
        #
        self.Lfbgs = torch.optim.LBFGS(self.model.parameters(), lr=1., max_iter=50000, 
                                           max_eval=50000, history_size=50, tolerance_grad=1e-6, 
                                           tolerance_change=1.0 * np.finfo(float).eps, 
                                           line_search_fn='strong_wolfe'); self.lfbgsiter = 0
        self.Adam = torch.optim.Adam(self.model.parameters(), lr=1e-4); self.maxiter = 70000; self.adamiter = 0

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
               torch.tensor(h, dtype=self.torch_dtype).to(device),\
               torch.tensor(Mesh, dtype=self.torch_dtype).to(device),\
               torch.tensor(weights, dtype=self.torch_dtype).to(device)


    def loss(self):
        Mesh = self.Mesh[..., None].clone().detach().requires_grad_(True).to(device)
        u = self.model(Mesh)
        gradu = torch.autograd.grad(u, Mesh, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        #
        lux = gradu[None, :, 0, 0]; rux = gradu[None, :, -1, 0]
        lv = self.v[:, :, 0]; rv = self.v[:, :, -1]
        bd = rux * rv - lux * lv
        #
        flux_u = u[:-1, -1, 0] - u[1:, 0, 0]
        flux_ux = gradu[:-1, -1, 0] - gradu[1:, 0, 0]
        ux = gradu[:, 1:-1, 0]
        # compute local loss
        Int = torch.sum((ux[None, ...] * self.dv[:, :, 1:-1] - self.fu[None, ...] * self.v[:, :, 1:-1]) * self.weights[None, ...], dim=-1) - bd # [deg + 1, N_x]
        local_loss = torch.sum(Int**2)
        # compute boundary loss
        bd_loss = (u[0, 0, 0]**2 + u[-1, -1, 0]**2)
        # compute flux loss
        flux_loss = torch.sum((flux_u**2 + flux_ux**2))
        loss = local_loss + bd_loss + flux_loss
        mae = torch.max(abs(u - self.exact(Mesh)))
        mse = torch.mean(abs(u - self.exact(Mesh))**2)
        return loss, mae, mse
    
    def exact_loss(self):
        x = self.Mesh
        u = self.exact(x)
        ux = torch.cos(self.w * x)  - self.w * x * torch.sin(self.w * x)
        lux = ux[None, :, 0]; rux = ux[None, :, -1]
        lv = self.v[:, :, 0]; rv = self.v[:, :, -1]
        bd = rux * rv - lux * lv
        #
        flux_u = u[:-1, -1] - u[1:, 0]
        flux_ux = ux[:-1, -1] - ux[1:, 0]
        ux = ux[:, 1:-1]
        # compute local loss
        Int = torch.sum((ux[None, ...] * self.dv[:, :, 1:-1] - self.fu[None, ...] * self.v[:, :, 1:-1]) * self.weights[None, ...], dim=-1) - bd # [deg + 1, N_x]
        local_loss = torch.sum(Int**2)
        # compute boundary loss
        bd_loss = (u[0, 0]**2 + u[-1, -1]**2)
        # compute flux loss
        flux_loss = torch.sum((flux_u**2 + flux_ux**2))
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
        self.writer = SummaryWriter(f'./logs/poisson1d/DGNet')
        self.Lfbgs.step(self.loss_lfbgs)
        torch.save(self.model.state_dict(), f'./models/DGNet/{self.name}.pth')
        loss, mae, mse = self.loss_adam()
        best_loss = mae
        while mae > 1e-6:
            loss.backward()
            self.Adam.step()
            loss, mae, mse = self.loss_adam()
            if mae < best_loss:
                best_loss = mae
                torch.save(self.model.state_dict(), f'./models/DGNet/{self.name}.pth')
            if self.adamiter + self.lfbgsiter > self.maxiter:
                break
        self.adamiter = 0
        self.writer.close()
        print(f'Finished training in {time.time()-t_start:.4f} seconds')
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

class Poisson1D_base:
    def __init__(self, method: str='PINN', N_x:int=10, N_int:int=10, deg:int=3, \
                 num_layers:int=2, input_size:int=1, hidden_size:int=50, output_size:int=1, act:str='tanh') -> None:
        #
        self.name = f'1dpoisson_{method}_{N_x}_{N_int}_{deg}_{num_layers}_{input_size}_{hidden_size}_{output_size}_{act}'
        #
        self.a = 0.0
        self.b = 1.5
        self.np_dtype = np.float64
        self.torch_dtype = torch.float32
        self.method = method
        #
        self.N_x = N_x
        self.N_int = N_int
        self.deg = deg
        # case 1
        # self.f = lambda x: 10 * torch.ones_like(x)
        # case 2
        self.w = 3 * pi
        self.f = lambda x: 2 * self.w * torch.sin(self.w * x) + self.w**2 * x * torch.cos(self.w * x)
        self.exact = lambda x: x * torch.cos(self.w * x)
        #
        if self.method == "PINN":
            self.mesh = torch.linspace(self.a, self.b, self.N_x + 1, device=device).view(-1, 1)
            self.model = MLP(input_size=input_size, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers, act=act).to(device).to(self.torch_dtype)
        
        elif self.method == "DeepRitz":
            nodes, weights = np.polynomial.legendre.leggauss(self.N_x)
            h = self.b - self.a
            mesh = 0.5 * (nodes + 1.) * h + self.a
            self.weights = torch.tensor(0.5 * weights * h).to(device)
            self.mesh = torch.tensor(mesh, device=device, dtype=self.torch_dtype).view(-1, 1)
            self.model = ResNet(input_size=input_size, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers, act=act).to(device).to(self.torch_dtype)
        
        elif self.method == "hpVPINN":
            self.testfunc = TestFunction1D(func_type='Legendre')
            self.x, self.xc, self.h, self.mesh, self.weights = self.get_mesh()
            self.v, self.dv = self.test_data()
            self.model = MLP(input_size=input_size, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers, act=act).to(device).to(self.torch_dtype)
        
        else:
            raise ValueError(f"Invalid method: {self.method}")
        self.fx = self.f(self.mesh)
        self.exactu = self.exact(self.mesh)

        #
        #
        self.Lfbgs = torch.optim.LBFGS(self.model.parameters(), lr=1.0, max_iter=20000, 
                                           max_eval=50000, history_size=50, tolerance_grad=1e-6, 
                                           tolerance_change=1.0 * np.finfo(float).eps, 
                                           line_search_fn='strong_wolfe'); self.lfbgsiter = 0
        self.Adam = torch.optim.Adam(self.model.parameters(), lr=1e-4); self.maxiter = 40000; self.adamiter = 0
    
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
        #
        eq_loss = torch.mean((u_xx + self.fx) ** 2) 
        bd_loss = torch.mean(u[[0, -1], 0] ** 2)
        loss = eq_loss + 10*bd_loss
        mae = torch.max(abs(self.exactu - u))
        mse = torch.mean(abs(self.exactu - u)**2)
        return loss, mae, mse
    
    def deepritz(self):
        self.mesh.requires_grad = True
        u = self.model(self.mesh)
        u_x = torch.autograd.grad(u, self.mesh, torch.ones_like(u), create_graph=True)[0]
        energy = torch.sum((0.5 * u_x**2 - self.fx * u) * self.weights[:, None])
        xb = self.mesh[[0, -1]] 
        output_b = u[[0, -1]] 
        exact_b = self.exact(xb)
        bd_loss = torch.mean((output_b - exact_b)**2)
        loss = energy + 500 * bd_loss
        mae = torch.max(abs(self.exactu - u))
        mse = torch.mean(abs(self.exactu - u)**2)
        return loss, mae, mse
    
    def hpvpinn(self):
        Mesh = self.mesh[..., None].clone().detach().requires_grad_(True).to(device)
        u = self.model(Mesh)
        gradu = torch.autograd.grad(u, Mesh, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        lux = gradu[None, :, 0, 0]; rux = gradu[None, :, -1, 0]
        lv = self.v[:, :, 0]; rv = self.v[:, :, -1]
        bd = rux * rv - lux * lv
        #
        ux = gradu[:, 1:-1, 0]
        # compute local loss
        Int = torch.sum((ux[None, ...] * self.dv[:, :, 1:-1] - self.fx[None, :, 1:-1] * self.v[:, :, 1:-1]) * self.weights[None, ...], dim=-1) - bd # [deg + 1, N_x]
        local_loss = torch.sum(Int**2)
        # compute boundary loss
        bd_loss = (u[0, 0, 0]**2 + u[-1, -1, 0]**2)
        loss = local_loss + 100*bd_loss
        mae = torch.max(abs(u - self.exact(Mesh)))
        mse = torch.mean(abs(u - self.exact(Mesh))**2)
        return loss, mae, mse
    
    def train(self):
        print('*********** Started training ...... ***************')
        t = time.time()
        self.writer = SummaryWriter(f'./logs/Poisson1D/{self.method}')
        loss, mae, mse = self.loss()
        best_loss = loss
        epoch = 0
        # while  mse.item() > 1e-3:
        while  epoch < self.maxiter:
            self.Adam.zero_grad()
            loss, mae, mse = self.loss()
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
            # if epoch > self.maxiter:
            #     break
            # if mae.item() < 1e-4:
            #     print(f"Epoch {epoch}: Loss = {loss.item():.6f}, Exact Loss = {mse.item():.6f}")
            #     break
        self.writer.close()
        print(f'Finished training in {time.time()-t:.4f} seconds')
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
               torch.tensor(h, dtype=self.torch_dtype).to(device),\
               torch.tensor(Mesh, dtype=self.torch_dtype).to(device),\
               torch.tensor(weights, dtype=self.torch_dtype).to(device)

class Poisson2d_dg:
    def __init__(self, boundary_type: str = 'regular', Nint_elt: int=15, Nint_edge: int = 20, order: int=3, partition: float=0.2,\
                  num_layers: int=2, hidden_size: int=50, act: str='tanh',\
                  sigma_eq: int=1, sigma_bd: int=1, sigma_flux: int=1) -> None:
        self.name = f'DG_2dpoisson_{boundary_type}_{Nint_elt}_{Nint_edge}_{order}_{partition}_{num_layers}_{hidden_size}_{act}'
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
        exact_dict = np.load('./data/exact2dpolygon.npz')
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
        self.writer = SummaryWriter(f'./logs/poisson2d/DGNet')
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

class Poisson2d_base:
    def __init__(self, boundary_type: str = 'regular', N_points: int=100, Nint_edge: int = 20,\
                 method: str = 'PINN', num_layers: int = 3, hidden_size: int = 50, act: str = 'tanh') -> None:
        self.name = f'2dpoisson_{boundary_type}_{N_points}_{Nint_edge}_{num_layers}_{hidden_size}_{act}'
        self.genmesh = GenMesh2D(boundary_type=boundary_type, Nint_edge=20, Nint_elt=15, param=f'pq30a0.05e')
        #
        self.Nelt = self.genmesh.Nelt
        self.method = method
        self.boundary_type = boundary_type
        self.Nint_edge = Nint_edge
        #
        self.inner_p, self.edges_p = self.get_mesh(N_points, Nint_edge)
        self.Mesh = torch.cat((self.inner_p, self.edges_p), dim=0)
        self.N = self.inner_p.shape[0]
        if method == 'PINN':
            _, _, _, _, _, self.Mesh_test, _, _ = self.genmesh.get_mesh()
            self.model = MLP(input_size=2, hidden_size=hidden_size, output_size=1, num_layers=num_layers, act=act).to(device).to(torch.float64)
        elif method == 'DeepRitz':
            self.Mesh_test, _, _, _, _, _, _, _ = self.genmesh.get_mesh()
            self.model = ResNet(input_size=2, hidden_size=hidden_size, output_size=1, num_layers=num_layers, act=act).to(device).to(torch.float64)
        
        # case 1
        if boundary_type == 'regular':
            self.f = lambda x, y: ((4*x**4-4*x**3+10*x**2-6*x+2)*(y-y**2)*torch.exp(x**2+y**2) + (4*y**4-4*y**3+10*y**2-6*y+2)*(x-x**2)*torch.exp(x**2+y**2))*10
            self.exact = lambda x, y: 10 * x*(1-x)*y*(1-y)*torch.exp(x**2 + y**2)
            self.fxy = self.f(self.Mesh[..., 0], self.Mesh[..., 1])
            self.u_exact = self.exact(self.Mesh[..., 0], self.Mesh[..., 1]).unsqueeze(-1)
        # case 2
        elif boundary_type == 'polygon':
            self.f = lambda x, y: torch.ones_like(x) * 10
            self.fxy = self.f(self.Mesh[..., 0], self.Mesh[..., 1])
            self.get_exact_polygon()
        #
        self.Adam = torch.optim.Adam(self.model.parameters(), lr=1e-3); self.maxiter = 20000; self.adamiter = 0

    def get_exact_polygon(self):
        Mesh = self.Mesh_test.clone().detach().cpu().numpy()
        exact_dict = np.load('./data/exact2dpolygon.npz')
        x = exact_dict['x']; y = exact_dict['y']
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
        eq_loss = torch.sum((uxx + uyy + self.f(Mesh[..., 0], Mesh[..., 1]))**2)
        u_edge = u[self.N:, :]
        bd_loss = torch.sum(u_edge**2)
        loss = eq_loss + 100 * bd_loss
        u_test = self.model(self.Mesh_test)
        mse = torch.mean((u_test - self.u_exact)**2)
        mae = torch.max(torch.abs(u_test - self.u_exact))
        return loss, mse, mae
    
    def deepritz(self):
        Mesh = self.Mesh.clone().detach().requires_grad_(True).to(device)
        u = self.model(Mesh)
        gradu = torch.autograd.grad(u, Mesh, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        eq_loss = torch.sum(0.5 * (gradu[..., 0]**2 + gradu[..., 1]**2) - self.f(Mesh[..., 0], Mesh[..., 1])*u.squeeze(-1))
        u_edge = u[self.N:, :]
        bd_loss = torch.sum(u_edge**2)
        loss = eq_loss + 500 * bd_loss
        u_test = self.model(self.Mesh_test)
        mse = torch.mean((u_test - self.u_exact)**2)
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
        # while  mse.item() > 1e-3:
        while  epoch < self.maxiter:
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
            # if epoch > self.maxiter:
            #     break
            # if mae.item() < 1e-4:
            #     print(f"Epoch {epoch}: Loss = {loss.item():.6f}, Exact Loss = {mse.item():.6f}")
            #     break
        self.writer.close()
        print(f'Finished training in {time.time()-t:.4f} seconds')
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

class Burgers_dg:
    def __init__(self, N_x:int, N_t:int, Nint_x:int, deg:int, num_layers:int=2, hidden_size:int=20, act:str='tanh') -> None:
        #
        self.name = f'DG_burgers_{N_x}_{N_t}_{Nint_x}_{deg}_{num_layers}_{hidden_size}_{act}'
        self.np_dtype = np.float64
        self.torch_dtype = torch.float64
        self.save_path = f'./models/DGNet/{self.name}.pth'
        #
        self.a = 0.0
        self.b = 2.0 * pi
        self.t0 = 0.0
        self.T = 1.5
        #
        self.N_x = N_x
        self.N_t = N_t
        self.Nint_x = Nint_x
        self.deg = deg
        #
        self.testfunc = TestFunction1D(func_type='Polynomial')
        #
        self.x, self.xc, self.h, self.t, self.Mesh, self.xmesh, self.weights = self.get_mesh()
        self.v, self.dv = self.test_data()
        self.model = DGNet(num_modules=self.N_x, input_size=2, hidden_size=hidden_size, output_size=1, num_layers=num_layers, act=act).to(device).to(self.torch_dtype)
        mesh = self.Mesh.cpu().detach().numpy()
        self.u_exact = torch.tensor(burgers_exact(mesh[:, :, :, 0], mesh[:, :, :, 1])).to(device)
        #
        self.f = lambda x: x ** 2 / 2
        self.init = lambda x: torch.sin(x) + 1/2
        self.flux = lambda x, y: x - y
        self.exact_init = self.init(self.xmesh)
        #
        self.Lfbgs = torch.optim.LBFGS(self.model.parameters(), lr=1.0, max_iter=20000, 
                                           max_eval=50000, history_size=50, tolerance_grad=1e-7, 
                                           tolerance_change=1.0 * np.finfo(float).eps, 
                                           line_search_fn='strong_wolfe')
        self.Adam = torch.optim.Adam(self.model.parameters(), lr=1e-4); self.maxiter = 10000; self.adamiter = 0
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
        #
        x = np.linspace(self.a, self.b, self.N_x + 1)
        t = np.linspace(self.t0, self.T, self.N_t + 1)
        xc = (x[:-1] + x[1:]) / 2.0
        h = (x[1:] - x[:-1]) 
        nodes, weights = np.polynomial.legendre.leggauss(self.Nint_x)
        #
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
               torch.tensor(h, dtype=self.torch_dtype).to(device),\
               torch.tensor(t, dtype=self.torch_dtype).to(device),\
               torch.tensor(Mesh, dtype=self.torch_dtype).to(device),\
               torch.tensor(xmesh, dtype=self.torch_dtype).to(device),\
               torch.tensor(weights, dtype=self.torch_dtype).to(device)

    
    def loss(self):
        Mesh = self.Mesh.clone().detach().requires_grad_(True).to(device)
        u = self.model(Mesh)
        fu = self.f(u)
        gradu = torch.autograd.grad(u, Mesh, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        ux = gradu[..., 0]; ut = gradu[..., 1]
        #
        lfu = self.f(u[:, :, 0, 0]); rfu = self.f(u[:, :, -1, 0])
        lv = self.v[:, :, 0]; rv = self.v[:, :, -1]
        bd = rfu[None, ...] * rv[:, :, None] - lfu[None, ...] * lv[:, :, None]
        #
        flux_u = self.flux(u[:-1, :, -1, 0], u[1:, :, 0, 0])
        # flux_ux = self.flux(ux[:-1, :, -1], ux[1:, :, 0]) 
        lux = ux[0, :, 0]; rux = ux[-1, :, -1]
        ux = ux[:, :, 1:-1]; ut = ut[:, :, 1:-1]
        

        # compute local loss
        Int = torch.sum((ut[None, ...] * self.v[:, :, None, 1:-1] - fu[None, :, :, 1:-1, 0] * self.dv[:, :, None, 1:-1]) * self.weights[None, :, None, :], dim=-1) + bd
        local_loss = torch.sum(Int**2)
        # compute init_loss
        init_loss = torch.sum((u[:, 0, :, 0] - self.exact_init)**2)
        # compute bd_loss
        bd_loss = torch.sum((u[0, :, 0, 0] - u[-1, :, -1, 0])**2 + (rux - lux)**2)
        # compute flux_loss
        flux_loss = torch.sum(flux_u**2)
        loss = local_loss + init_loss + bd_loss + flux_loss
        mse = torch.mean((u.squeeze(-1)-self.u_exact)**2)
        mae = torch.max(torch.abs(u.squeeze(-1)-self.u_exact))
        return loss, mse, mae
    
    def loss_lfbgs(self):
        self.Lfbgs.zero_grad()
        loss, mse, mae = self.loss()
        loss.backward()
        self.writer.add_scalar(f"mse_vs_iter", mse, self.iter + self.adamiter)
        self.writer.add_scalar(f"mse_vs_time", mse, time.time() - self.t)
        self.iter += 1
        if self.iter % 100 == 0:
            print(f"LBFGS At iter: {self.iter}, loss_train:{loss.item():.6f}, mse:{mse.item():.6f}, mae:{mae.item():.6f}")
        return loss
    
    def loss_adam(self):
        self.Adam.zero_grad()
        loss, mse, mae = self.loss()
        self.writer.add_scalar(f"mse_vs_iter", mse, self.iter + self.adamiter)
        self.writer.add_scalar(f"mse_vs_time", mse, time.time() - self.t)
        self.adamiter += 1
        if self.adamiter % 100 == 0:
            print(f"Adam At iter: {self.adamiter}, loss_train:{loss.item():.6f}, mse:{mse.item():.6f}, mae:{mae.item():.6f}")
        return loss
    def train(self):
        t_start = time.time()
        print('*********** Started training ...... ***************')
        self.writer = SummaryWriter(f'logs/burgers1d/DGNet')
        self.t = time.time()
        self.Lfbgs.step(self.loss_lfbgs)
        torch.save(self.model.state_dict(), self.save_path)
        loss = self.loss_adam()
        best_loss = loss
        while loss > 1e-3:
            loss.backward()
            self.Adam.step()
            loss = self.loss_adam()
            if loss < best_loss:
                best_loss = loss
                torch.save(self.model.state_dict(), self.save_path)
            if self.adamiter > self.maxiter:
                break
        self.adamiter = 0
        self.writer.close()
        print(f'Finished training in {time.time()-t_start:.4f} seconds')
    def load(self):
        if os.path.exists(self.save_path):
            print("Loading saved model...")
            model_dict = torch.load(self.save_path)
            self.model.load_state_dict(model_dict)
            return True
        else:
            print("No saved model found. Need to train...")
            return False

class Burgers_pinn():
    def __init__(self, method:str='PINN', num_layers:int=4, hidden_size:int=128, act:str='tanh') -> None:
        self.name = f'1dburgers_{method}_{num_layers}_{hidden_size}_{act}'
        self.method = method
        self.torch_dtype = torch.float64
        x = np.linspace(0, 2*np.pi, 1000)
        t = np.linspace(0, 1.5, 100)
        xx, tt = np.meshgrid(x, t)
        self.mesh = np.stack([xx, tt], axis=-1)
        self.model = MLP(input_size=2, output_size=1, hidden_size=hidden_size, num_layers=num_layers, act=act).to(device)
        self.init = lambda x: torch.sin(x) + 1/2
        self.Adam = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.maxiter = 40000
        self.u_exact = torch.tensor(burgers_exact(self.mesh[:,:, 0], self.mesh[:,:, 1])).to(device)

    def loss(self):
        if self.method == 'PINN':
            return self.pinn()
        else:
            pass
        
    def pinn(self):
        mesh = torch.tensor(self.mesh, dtype=torch.float32, requires_grad=True).to(device)
        u = self.model(mesh)
        grad_u = torch.autograd.grad(u, mesh, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        ux = grad_u[..., 0]; ut = grad_u[..., 1]
        eq_loss = torch.sum((ut+u.squeeze(-1)*ux)**2)
        init_loss = torch.sum((self.init(mesh[0, :, 0])-u[0, :, 0])**2)
        bd_loss = torch.sum((u[:, 0, 0]-u[:, -1, 0])**2) + torch.sum((ux[0, :]-ux[-1, :])**2)
        loss =  eq_loss + init_loss + bd_loss
        mse = torch.mean((u.squeeze(-1)-self.u_exact)**2)
        mae = torch.max(torch.abs(u.squeeze(-1)-self.u_exact))
        return loss, mse, mae


    def train(self):
        print('*********** Started training ...... ***************')
        t = time.time()
        self.writer = SummaryWriter(f'./logs/burgers1d/PINN')
        loss, mse, mae = self.loss()
        best_loss = loss
        epoch = 0
        while  epoch < self.maxiter:
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
        print(f'Finished training in {time.time()-t:.4f} seconds')
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
    
class Burgers_hpVPINN:
    def __init__(self, N_x:int, N_t:int, Nint_x:int, deg:int, num_layers:int=2, hidden_size:int=20, act:str='tanh') -> None:
        #
        self.name = f'1dburgers_{N_x}_{N_t}_{Nint_x}_{deg}_{num_layers}_{hidden_size}_{act}'
        self.np_dtype = np.float64
        self.torch_dtype = torch.float64
        self.save_path = f'./models/DGNet/{self.name}.pth'
        #
        self.a = 0.0
        self.b = 2.0 * pi
        self.t0 = 0.0
        self.T = 1.5
        #
        self.N_x = N_x
        self.N_t = N_t
        self.Nint_x = Nint_x
        self.deg = deg
        #
        self.testfunc = TestFunction1D(func_type='Legendre')
        #
        self.x, self.xc, self.h, self.t, self.Mesh, self.xmesh, self.weights = self.get_mesh()
        mesh = self.Mesh.cpu().detach().numpy()
        self.v, self.dv = self.test_data()
        self.model = MLP(input_size=2, hidden_size=hidden_size, output_size=1, num_layers=num_layers, act=act).to(device).to(self.torch_dtype)
        #
        self.f = lambda x: x ** 2 / 2
        self.init = lambda x: torch.sin(x) + 1/2
        self.exact_init = self.init(self.xmesh)
        self.u_exact = torch.tensor(burgers_exact(mesh[:,:, :, 0], mesh[:,:, :, 1])).to(device)
        #
        self.Lfbgs = torch.optim.LBFGS(self.model.parameters(), lr=1.0, max_iter=40000, 
                                           max_eval=50000, history_size=50, tolerance_grad=1e-7, 
                                           tolerance_change=1.0 * np.finfo(float).eps, 
                                           line_search_fn='strong_wolfe')
        self.Adam = torch.optim.Adam(self.model.parameters(), lr=1e-4); self.maxiter = 70000; self.adamiter = 0
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
        #
        x = np.linspace(self.a, self.b, self.N_x + 1)
        t = np.linspace(self.t0, self.T, self.N_t + 1)
        xc = (x[:-1] + x[1:]) / 2.0
        h = (x[1:] - x[:-1]) 
        nodes, weights = np.polynomial.legendre.leggauss(self.Nint_x)
        #
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
               torch.tensor(h, dtype=self.torch_dtype).to(device),\
               torch.tensor(t, dtype=self.torch_dtype).to(device),\
               torch.tensor(Mesh, dtype=self.torch_dtype).to(device),\
               torch.tensor(xmesh, dtype=self.torch_dtype).to(device),\
               torch.tensor(weights, dtype=self.torch_dtype).to(device)

    
    def loss(self):
        Mesh = self.Mesh.clone().detach().requires_grad_(True).to(device)
        u = self.model(Mesh)
        fu = self.f(u)
        gradu = torch.autograd.grad(u, Mesh, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        ux = gradu[..., 0]; ut = gradu[..., 1]
        #
        lfu = self.f(u[:, :, 0, 0]); rfu = self.f(u[:, :, -1, 0])
        lv = self.v[:, :, 0]; rv = self.v[:, :, -1]
        bd = rfu[None, ...] * rv[:, :, None] - lfu[None, ...] * lv[:, :, None]
        #
        lux = ux[0, :, 0]; rux = ux[-1, :, -1]
        ux = ux[:, :, 1:-1]; ut = ut[:, :, 1:-1]
        

        # compute local loss
        Int = torch.sum((ut[None, ...] * self.v[:, :, None, 1:-1] - fu[None, :, :, 1:-1, 0] * self.dv[:, :, None, 1:-1]) * self.weights[None, :, None, :], dim=-1) + bd
        local_loss = torch.sum(Int**2)
        # compute init_loss
        init_loss = torch.sum((u[:, 0, :, 0] - self.exact_init)**2)
        # compute bd_loss
        bd_loss = torch.sum((u[0, :, 0, 0] - u[-1, :, -1, 0])**2 + (rux - lux)**2)
        # compute flux_loss
        loss = local_loss + init_loss + bd_loss
        mse = torch.mean((u.squeeze(-1)-self.u_exact)**2)
        mae = torch.max(torch.abs(u.squeeze(-1)-self.u_exact))
        return loss, mse, mae

    def train(self):
        print('*********** Started training ...... ***************')
        t = time.time()
        self.writer = SummaryWriter(f'./logs/burgers1d/hpVPINN')
        loss, mse, mae = self.loss()
        best_loss = loss
        epoch = 0
        while  epoch < self.maxiter:
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
        print(f'Finished training in {time.time()-t:.4f} seconds')
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
 