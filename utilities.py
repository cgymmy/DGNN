import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from scipy.special import legendre
import matplotlib.pyplot as plt
from triangle_gauss import *
from triangle import triangulate
from scipy.optimize import fsolve
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class TestFunction1D:
    def __init__(self, func_type='Polynomial'):
        self.type = func_type
    def get_value(self, x: torch.tensor, order: int=0, x_mid=None, h=None):
        if self.type == 'Polynomial':
            return self.Poly(x=x, order=order, x_mid=x_mid, h=h)
        elif self.type == 'Legendre':   
            return self.Legendre(x=x, x_mid=x_mid, order=order, h=h)
        else:
            raise ValueError(f"Unsupported function type: {self.type}")
    
    def Poly(self, x: torch.tensor, order: int, x_mid, h):
        x = 2 * (x - x_mid)/h
        if order == 0:
            v = torch.ones_like(x)
            dv = torch.zeros_like(x)
        else:
            v = x ** order
            dv = order * x ** (order - 1) * 2 / h
        return v, dv
    
    def Legendre(self, x: torch.tensor, x_mid: torch.tensor, order: int, h: torch.tensor):
        original_shape = x.shape
        x = 2 * (x - x_mid)/h
        x_np = x.cpu().numpy().flatten()
        hp = h.cpu()
        legendre_poly = legendre(order)
        v = legendre_poly(x_np)
        dv = np.gradient(v, x_np)
        v = torch.tensor(v).reshape(*original_shape)
        dv = torch.tensor(dv).reshape(*original_shape) * 2 / hp
        return v, dv

class TestFunction2D:
    def __init__(self, func_type: str = 'Polynomial') -> None:
        self.type = func_type
    def get_value(self, mesh: torch.tensor, order: int=0):
        if self.type == 'Polynomial':
            return self.Poly(mesh=mesh, order=order)
        else:
            raise ValueError(f"Unsupported function type: {self.type}")
        
    def Poly(self, mesh: torch.tensor, order: int=0):
        x = mesh[..., 0]
        y = mesh[..., 1]
        v, dv =[], []
        for i in range(order + 1):
            for j in range(order + 1 - i):
                v.append(x**i * y**j)
                dv_x = i * x**(i-1) * y**j if i > 0 else torch.zeros_like(x)
                dv_y = j * x**i * y**(j-1) if j > 0 else torch.zeros_like(y)
                dv.append(torch.stack([dv_x, dv_y], dim=-1))
        v = torch.stack(v, dim=0)
        dv = torch.stack(dv, dim=0)
        return v, dv

class GenMesh2D:
    def __init__(self, boundary_type: str = 'regular', param: str = 'pq30a0.2e', Nint_edge: int = 20, Nint_elt: int = 10) -> None:
        #
        self.para = param
        self.Nint_edge = Nint_edge
        self.Nint_elt = Nint_elt
        self.np_dtype = np.float64
        self.torch_dtype = torch.float64
        #
        self.vertices, self.segments = self.set_grid_data(boundary_type)
        if self.vertices is None:
            raise NotImplementedError(f"Grid name '{boundary_type}' is not supported.")
        # 使用 triangulate 生成三角形网格
        t = triangulate({"vertices": self.vertices, 'segments': self.segments}, self.para)
        self.points = np.array(t["vertices"], dtype=self.np_dtype)
        self.edges = np.array(t["edges"])
        self.Mesh_pinx = np.array(t["triangles"])
        
        # 网格属性
        self.Nv = len(self.points)  # 顶点数量
        self.Nelt = len(self.Mesh_pinx)  # 元素数量
        self.Nedge = len(self.edges)  # 边数量
        self.process_edges()

    def set_grid_data(self, boundary_type: str):
        v, e = None, None
        if boundary_type == 'polygon':
            v, e = genpolygon()
        grid_data = {
            'regular': (
                [[0, 0], [0, 1], [1, 1], [1, 0]],
                [[0, 1], [1, 2], [2, 3], [3, 0]]
            ),
            'irregular': (
                [[0, 0], [1, 0], [1, 0.5], [0.5, 0.5], [0.5, 1], [0, 1]],
                [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]]
            ),
            'polygon': (
                v, e
            )
        }
        return grid_data.get(boundary_type, (None, None))

    def process_edges(self):
        tria_inx = []; label = [[] for _ in range(self.Nedge)]
        for i in range(self.Nelt):
            v1, v2, v3 = self.Mesh_pinx[i, :]
            tria = [(v1, v2), (v2, v3), (v3, v1)]
            edge_inx = [];
            for no, ei in enumerate(tria):
                e1 = ei; e2 = (ei[1], ei[0])
                for j in range(self.Nedge):
                    if np.array_equal(e1, self.edges[j]) or np.array_equal(e2, self.edges[j]):
                        inx = j
                        label[j].append((i, no))
                edge_inx.append(inx); 
            tria_inx.append(edge_inx); 
        self.Mesh_einx = np.array(tria_inx)
        self.label = label
        self.inner_label = []; self.bd_label = []
        self.bd_edge_inx = []
        for i, sublist in enumerate(label):
            if len(sublist) == 2:
                self.inner_label.append(sublist)
            elif len(sublist) == 1:
                self.bd_label.append(sublist)
                self.bd_edge_inx.append(i)
        self.inner_label = np.array(self.inner_label)
        self.bd_label = np.array(self.bd_label)
        self.bd_edge_inx = np.array(self.bd_edge_inx)

    
    def get_mesh(self):
        # integrate over the element
        ref_p, ref_w = rule(self.Nint_elt); ref_p = ref_p.T
        self.num_eltp = ref_p.shape[0]
        tri_p = np.stack((self.points[self.Mesh_pinx[:, 0]], self.points[self.Mesh_pinx[:, 1]], self.points[self.Mesh_pinx[:, 2]]), axis=1)
        matrix = np.stack((tri_p[:, 1, :] - tri_p[:, 0, :], tri_p[:, 2, :] - tri_p[:, 0, :]), axis=1)
        inv_matrix = np.linalg.inv(matrix)
        elt_int = np.sum(matrix[:, None, :, :] * ref_p[None, :, :, None], axis=-2) + tri_p[:, None, 0, :]
        tri_sqr = np.linalg.det(matrix) / 2
        elt_weights = ref_w[None, :] * tri_sqr[:, None] * 2
        # integrate over the edge
        edges_p = np.stack((self.points[self.edges[:, 0]], self.points[self.edges[:, 1]]), axis=1)
        edges_vec = edges_p[:, 1, :] - edges_p[:, 0, :]
        edges_l = np.linalg.norm(edges_vec, axis=-1)
        nodes, weights = np.polynomial.legendre.leggauss(self.Nint_edge)
        self.num_edgep = nodes.shape[0]
        nodes = (nodes + 1)/2
        edges_weights = weights[None, :] * edges_l[:, None] / 2
        edges_int = nodes[None, :, None] * edges_vec[:, None, :] + edges_p[:,None, 0, :]
        # compute mesh normvec
        mesh_normvec = np.zeros((self.Nelt, 3, 2))
        mesh_normvec[:, 0, :] = (self.points[self.Mesh_pinx[:, 1]] - self.points[self.Mesh_pinx[:, 0]])/edges_l[self.Mesh_einx[:, 0, None]]   
        mesh_normvec[:, 1, :] = (self.points[self.Mesh_pinx[:, 2]] - self.points[self.Mesh_pinx[:, 1]])/edges_l[self.Mesh_einx[:, 1, None]]   
        mesh_normvec[:, 2, :] = (self.points[self.Mesh_pinx[:, 0]] - self.points[self.Mesh_pinx[:, 2]])/edges_l[self.Mesh_einx[:, 2, None]]
        mesh_normvec = np.stack((mesh_normvec[:, :, 1], -mesh_normvec[:, :, 0]), axis=-1)
        # integrate the mesh
        # mesh_edges_p = edges_int[self.Mesh_einx]
        mesh_edges_w = edges_weights[self.Mesh_einx]
        # Mesh = np.concatenate((elt_int, mesh_edges_p.reshape(self.Nelt, -1, 2)), axis=1)
        #
        e1 = nodes[:, None] * np.array([1, 0])[None, :]
        e2 = nodes[:, None] * np.array([-1, 1])[None, :] + np.array([1, 0])[None, :]
        e3 = nodes[:, None] * np.array([0, -1])[None, :] + np.array([0, 1])[None, :]
        ref_Mesh = np.concatenate((ref_p, e1, e2, e3), axis=0)
        Mesh = np.sum(matrix[:, None, :, :] * ref_Mesh[None, :, :, None], axis=-2) + tri_p[:, None, 0, :]

        #
        return torch.tensor(elt_int, dtype=self.torch_dtype).to(device),\
            torch.tensor(elt_weights, dtype=self.torch_dtype).to(device),\
            torch.tensor(edges_int, dtype=self.torch_dtype).to(device),\
            torch.tensor(mesh_edges_w, dtype=self.torch_dtype).to(device),\
            torch.tensor(mesh_normvec, dtype=self.torch_dtype).to(device),\
            torch.tensor(Mesh, dtype=self.torch_dtype).to(device),\
            torch.tensor(ref_Mesh, dtype=self.torch_dtype).to(device),\
            torch.tensor(inv_matrix, dtype=self.torch_dtype).to(device)

    def print_grid_info(self):
        print(f'In the whole domain: ')
        print(f'{self.Nv} points')
        print(f'{self.Nelt} elements')
        print(f'{self.Nedge} faces/edges')
    
    def plot_mesh(self):
        print("Plot the mesh:")
        plt.figure(figsize=(6, 6))
        plt.triplot(self.points[:,0], self.points[:,1], self.Mesh_pinx) 
        plt.plot(self.points[:,0], self.points[:,1], 'o', markersize=1) 
        plt.axis('equal')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.savefig('mesh.png') 
        plt.show()

def genpolygon():
    R = 1.0
    r = R / (math.sin(math.pi/5.)*math.tan(2*math.pi/5.)+math.cos(math.pi/5.))  # cos(36°)
    # print(r, R)
    vertices = []
    segments = []
    for n in range(10):
        angle = math.pi / 2 + n * math.pi / 5  
        if n % 2 == 0:
            radius = R
        else:
            radius = r
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        vertices.append([x, y])
        segments.append([n, n+1])
    vertices.append(vertices[0])
    segments.append([9, 0])
    return vertices, segments

class ParallelLinear(nn.Module):
    def __init__(self, num_modules, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_modules, out_features, in_features))
        self.bias = nn.Parameter(torch.randn(num_modules, out_features))

    def forward(self, x):
        # x shape: [N, ..., in_features] (例如 [N, n, m, 2])
        original_shape = x.shape
        x_flat = x.reshape(original_shape[0], -1, original_shape[-1])  # [N, L, in_features]
        output = torch.einsum("noi,nli->nlo", self.weight, x_flat)  # [N, L, out_features]
        output += self.bias.unsqueeze(1)  # 添加偏置
        return output.reshape(*original_shape[:-1], -1)

class DGNet(nn.Module):
    def __init__(self, num_modules, input_size: int=2, hidden_size: int=50, output_size: int=1, num_layers: int=2, act: str='tanh'):
        super().__init__()
        
        self.num_layers = num_layers
        self.activations = {
            'tanh': F.tanh,
            'sigmoid': F.sigmoid,
            'gelu': F.gelu,
            'relu': F.relu,
            'elu': F.elu
        }

        if act not in self.activations:
            raise ValueError(f"Unsupported activation function: {act}")
        
        self.act = act
        self.layers = nn.ModuleList()
        self.layers.append(ParallelLinear(num_modules, input_size, hidden_size))
        for _ in range(num_layers - 1):
            self.layers.append(ParallelLinear(num_modules, hidden_size, hidden_size))
        self.layers.append(ParallelLinear(num_modules, hidden_size, output_size))

    def forward(self, x):
        # x shape: [N, input_size]
        for i in range(self.num_layers):
            x = self.layers[i](x)  # 通过 ParallelLinear 进行计算
            x = self.activations[self.act](x)  # 应用激活函数
        x = self.layers[self.num_layers](x)
        return x


class MLP(nn.Module):
    def __init__(self, num_layers: int=5, input_size: int=1, hidden_size: int=50, output_size: int=1, act="relu"):
        super().__init__()
        activations = {
            'tanh': F.tanh,
            'sigmoid': F.sigmoid,
            'gelu': F.gelu,
            'relu': F.relu,
            'elu': F.elu,
            'sine': Sine()
        }
        self.activation = activations[act]
        self.net = nn.ModuleList()
        self.net.append(nn.Linear(input_size, hidden_size))
        for _ in range(num_layers - 1): 
            self.net.append(nn.Linear(hidden_size, hidden_size))
        self.net.append(nn.Linear(hidden_size, output_size)) 

    def forward(self, x):
        for layer in self.net[:-1]:  # 经过隐藏层
            x = self.activation(layer(x))
        return self.net[-1](x) 
    
class ResNet(nn.Module):
    def __init__(self, num_layers: int = 4, input_size: int = 1, hidden_size: int = 128, output_size: int = 1, act="ReLU"):
        super(ResNet, self).__init__()
        activations = {
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'gelu': nn.GELU(),
            'relu': nn.ReLU(),
            'elu': nn.ELU()
        }
        self.activation = activations[act]
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                self.activation,
                nn.Linear(hidden_size, hidden_size)
            )
            for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        y = self.input_layer(x)
        for layer in self.hidden_layers:
            y = y + self.activation(layer(y))
        return self.output_layer(y)
    
class Sine(nn.Module):
    def __init__(self):
        super(Sine, self).__init__()

    def forward(self, x):
        return torch.sin(x)

def solve_w(x, t, w0):
    func = lambda w: w - np.sin(x - (w + 0.5) * t)
    w_solution = fsolve(func, w0)
    return w_solution[0] 

def burgers_exact_scalar(x, t):
    left = solve_w(x, t, 0.8) + 0.5
    right = solve_w(x, t, -0.8) + 0.5
    if x <= 0.5 * t + np.pi:
        return left
    elif x >= 0.5 * t + np.pi:
        return right
    else:
        return (left + right)/2

burgers_exact = np.vectorize(burgers_exact_scalar)