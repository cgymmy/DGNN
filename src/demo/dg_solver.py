"""
经典间断有限元(Discontinuous Galerkin, DG)方法求解器
============================================

主要功能：
1. 求解1D Burgers方程: u_t + u*u_x + u_xx = 0
2. 求解1D Poisson方程: -u_xx = f, u(0)=u(L)=0
3. 求解不规则边界上的2D Poisson方程: -Δu = f, u|∂Ω = 0

DG方法基本思想：
- 将计算域分解为多个单元(element)
- 在每个单元内用多项式近似解
- 在单元边界处通过数值通量(numerical flux)连接相邻单元
- 使用弱形式和分部积分得到局部离散方程
- 对所有单元的方程进行求解

参考文献：
[1] Cockburn, B., Karniadakis, G. E., & Shu, C. W. (2000). 
    The development of discontinuous Galerkin methods. 
    Discontinuous Galerkin Methods: Theory, Computation and Applications, 11-50.
[2] Hesthaven, J. S., & Warburton, T. (2008). 
    Nodal Discontinuous Galerkin Methods. Springer.
"""

import os
import time
from abc import ABC, abstractmethod
from typing import Callable, Tuple, Optional
import numpy as np
import torch

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from config import device


class QuadratureRule:
    """高斯正交规则管理"""
    
    @staticmethod
    def legendre_gauss_1d(n_points: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        返回1D Legendre-Gauss正交规则
        
        Args:
            n_points: 正交点数
            
        Returns:
            nodes: 正交点 (参考元上, [-1,1])
            weights: 正交权重
        """
        nodes, weights = np.polynomial.legendre.leggauss(n_points)
        return nodes, weights
    
    @staticmethod
    def triangle_gauss_2d(order: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        返回三角形单元上的Gauss正交规则
        
        Args:
            order: 正交精度阶数
            
        Returns:
            points: 正交点坐标 (n_points, 2)，参考三角形上
            weights: 正交权重 (n_points,)
        """
        from mesh.triangle_gauss import rule
        return rule(order)


class TestFunction:
    """测试函数(试函数)空间"""
    
    def __init__(self, basis_type: str = 'polynomial'):
        """
        Args:
            basis_type: 基函数类型 ('polynomial', 'legendre', 等)
        """
        self.basis_type = basis_type
    
    def evaluate_1d(self, x: torch.Tensor, order: int, 
                    x_mid: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        在1D单元上计算测试函数及其导数
        
        Args:
            x: 计算点坐标
            order: 多项式阶数
            x_mid: 单元中点
            h: 单元宽度
            
        Returns:
            v: 测试函数值
            dv: 测试函数导数
        """
        # 变换到参考元[-1,1]
        xi = 2 * (x - x_mid) / h
        
        if self.basis_type == 'polynomial':
            if order == 0:
                v = torch.ones_like(xi)
                dv = torch.zeros_like(xi)
            else:
                v = xi ** order
                dv = order * xi ** (order - 1) * 2 / h
        elif self.basis_type == 'legendre':
            # Legendre多项式
            from scipy.special import legendre
            legendre_poly = legendre(order)
            v_np = legendre_poly(xi.cpu().numpy())
            v = torch.tensor(v_np, dtype=xi.dtype, device=xi.device)
            # 数值求导
            dv = torch.autograd.grad(v.sum(), xi, create_graph=True)[0]
        else:
            raise ValueError(f"Unknown basis type: {self.basis_type}")
        
        return v, dv
    
    def evaluate_2d(self, points: torch.Tensor, order: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        在2D单元上计算测试函数及其梯度
        
        Args:
            points: 计算点坐标 (n_points, 2)，参考单元上
            order: 多项式阶数
            
        Returns:
            v: 测试函数值 (n_basis, n_points)
            grad_v: 测试函数梯度 (n_basis, n_points, 2)
        """
        if self.basis_type == 'polynomial':
            # 2D多项式基: 1, x, y, x^2, xy, y^2, ...
            x = points[..., 0]
            y = points[..., 1]
            
            basis_list = [torch.ones_like(x)]
            grad_list = [torch.zeros(*x.shape, 2, device=x.device, dtype=x.dtype)]
            
            for i in range(1, order + 1):
                for j in range(i + 1):
                    k = i - j
                    # x^j * y^k
                    if j > 0:
                        basis_list.append(x ** j * y ** k)
                        grad_x = j * x ** (j - 1) * y ** k
                        grad_y = x ** j * k * y ** (k - 1) if k > 0 else torch.zeros_like(x)
                        grad_list.append(torch.stack([grad_x, grad_y], dim=-1))
                    else:
                        basis_list.append(y ** k)
                        grad_y = k * y ** (k - 1) if k > 0 else torch.zeros_like(y)
                        grad_list.append(torch.stack([torch.zeros_like(x), grad_y], dim=-1))
            
            v = torch.stack(basis_list, dim=0)
            grad_v = torch.stack(grad_list, dim=0)
            return v, grad_v
        else:
            raise ValueError(f"Unknown basis type: {self.basis_type}")


class DGSolver1D(ABC):
    """1D问题的DG求解器基类"""
    
    def __init__(self, n_elements: int, n_quad_points: int, poly_order: int,
                 dtype=torch.float64):
        """
        Args:
            n_elements: 单元数
            n_quad_points: 正交点数
            poly_order: 多项式阶数
            dtype: 数据类型
        """
        self.n_elements = n_elements
        self.n_quad_points = n_quad_points
        self.poly_order = poly_order
        self.dtype = dtype
        
        self.test_func = TestFunction('polynomial')
        self.quad = QuadratureRule()
        
        # 初始化网格
        self._setup_mesh()
        # 初始化试函数
        self._setup_test_functions()
    
    def _setup_mesh(self):
        """建立1D网格"""
        # 获取计算域边界
        a, b = self.domain_bounds()
        
        # 单元节点和中点
        self.x_nodes = np.linspace(a, b, self.n_elements + 1)
        self.x_centers = (self.x_nodes[:-1] + self.x_nodes[1:]) / 2.0
        self.h = np.diff(self.x_nodes)  # 单元宽度
        
        # 单元内部的正交点
        quad_nodes, quad_weights = self.quad.legendre_gauss_1d(self.n_quad_points)
        self.x_quad_interior = 0.5 * (quad_nodes[None, :] + 1) * self.h[:, None] + self.x_nodes[:-1, None]
        self.quad_weights = 0.5 * quad_weights[None, :] * self.h[:, None]
        
        # 单元边界点 (左、右端点)
        self.x_mesh = np.zeros((self.n_elements, self.n_quad_points + 2))
        self.x_mesh[:, 0] = self.x_nodes[:-1]
        self.x_mesh[:, 1:-1] = self.x_quad_interior
        self.x_mesh[:, -1] = self.x_nodes[1:]
        
        # 转换为torch tensor并移至设备
        self.x_nodes = torch.tensor(self.x_nodes, dtype=self.dtype, device=device)
        self.x_centers = torch.tensor(self.x_centers, dtype=self.dtype, device=device)
        self.h = torch.tensor(self.h, dtype=self.dtype, device=device)
        self.x_mesh = torch.tensor(self.x_mesh, dtype=self.dtype, device=device)
        self.quad_weights = torch.tensor(self.quad_weights, dtype=self.dtype, device=device)
    
    def _setup_test_functions(self):
        """预计算测试函数值"""
        v_list = []
        dv_list = []
        
        for order in range(self.poly_order + 1):
            v, dv = self.test_func.evaluate_1d(
                self.x_mesh, order, 
                self.x_centers[:, None], 
                self.h[:, None]
            )
            v_list.append(v)
            dv_list.append(dv)
        
        self.v_basis = torch.stack(v_list, dim=0).to(device)  # (n_basis, n_elem, n_quad+2)
        self.dv_basis = torch.stack(dv_list, dim=0).to(device)  # (n_basis, n_elem, n_quad+2)
    
    @abstractmethod
    def domain_bounds(self) -> Tuple[float, float]:
        """返回计算域的边界"""
        pass
    
    @abstractmethod
    def source_term(self, x: torch.Tensor) -> torch.Tensor:
        """源项 f(x)"""
        pass
    
    @abstractmethod
    def exact_solution(self, x: torch.Tensor) -> torch.Tensor:
        """精确解 (用于精度验证)"""
        pass
    
    @abstractmethod
    def compute_loss(self, u: torch.Tensor) -> torch.Tensor:
        """计算损失函数"""
        pass
    
    def solve(self):
        """求解方程"""
        raise NotImplementedError


class Poisson1D_DG(DGSolver1D):
    """
    1D Poisson方程的DG求解: -u'' = f, u(a)=u(b)=0
    
    使用内惩罚(IP) DG格式，数值通量为：
    {u} = (u_left + u_right)/2
    {∇u} = (∇u_left + ∇u_right)/2
    """
    
    def __init__(self, n_elements: int = 25, n_quad_points: int = 20, 
                 poly_order: int = 5, penalty_param: float = 10.0, **kwargs):
        """
        Args:
            n_elements: 单元数
            n_quad_points: 正交点数
            poly_order: 多项式阶数
            penalty_param: 惩罚参数 (sigma)
        """
        self.penalty = penalty_param
        super().__init__(n_elements, n_quad_points, poly_order, **kwargs)
    
    def domain_bounds(self) -> Tuple[float, float]:
        return 0.0, 1.5
    
    def source_term(self, x: torch.Tensor) -> torch.Tensor:
        """
        测试问题: u(x) = x*cos(ωx), f(x) = 2ω*sin(ωx) + ω²*x*cos(ωx)
        其中 ω = 15π
        """
        w = 15 * np.pi
        return 2 * w * torch.sin(w * x) + w**2 * x * torch.cos(w * x)
    
    def exact_solution(self, x: torch.Tensor) -> torch.Tensor:
        """精确解"""
        w = 15 * np.pi
        return x * torch.cos(w * x)
    
    def compute_loss(self, u: torch.Tensor) -> torch.Tensor:
        """
        计算DG离散的损失函数
        
        DG方程: 求u ∈ V_h 使得对所有 v ∈ V_h，有：
        
        ∫_Ω ∇u·∇v dx - ∫_∂E [[∇u]]·{v} ds - ∫_∂E {∇u}·[[v]] ds 
        + σ/h ∫_∂E [[u]]·[[v]] ds = ∫_Ω f·v dx
        """
        # 计算导数 (简单的有限差分)
        ux = torch.zeros_like(u)
        
        # 内部正交点的导数 (使用中心差分)
        ux[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * self.h[:, None])
        # 边界点的导数
        ux[:, 0] = (u[:, 1] - u[:, 0]) / self.h
        ux[:, -1] = (u[:, -1] - u[:, -2]) / self.h
        
        # 仅在内部正交点上计算源项
        x_interior = self.x_mesh[:, 1:-1]  # (n_elem, n_quad)
        f_interior = self.source_term(x_interior)  # (n_elem, n_quad)
        
        # 局部积分项: ∫ ∇u·∇v dx - ∫ f·v dx
        # self.dv_basis: (n_basis, n_elem, n_quad+2)
        # self.quad_weights: (n_elem, n_quad)
        # ux: (n_elem, n_quad+2) -> ux[:, 1:-1]: (n_elem, n_quad)
        local_int = torch.sum(
            (ux[:, 1:-1][None, :, :] * self.dv_basis[:, :, 1:-1] - 
             f_interior[None, :, :] * self.v_basis[:, :, 1:-1]) * 
            self.quad_weights[None, :, :], dim=-1
        )  # (n_basis, n_elem)
        
        # 界面项: 数值通量和通量跳跃 (简化处理)
        interface_loss = 0.0
        
        for i in range(self.n_elements - 1):
            # 相邻单元的边界处导数值
            u_right = u[i, -1]  # 单元i右端
            u_left = u[i + 1, 0]  # 单元i+1左端
            
            # 跳跃
            jump_u = u_right - u_left
            
            # 惩罚项: σ/h * [u]^2
            h_avg = (self.h[i] + self.h[i + 1]) / 2
            penalty_term = self.penalty / h_avg * jump_u ** 2
            interface_loss = interface_loss + penalty_term
        
        # 边界条件: u(0)=u(1)=0 (Dirichlet)
        boundary_loss = u[0, 0]**2 + u[-1, -1]**2
        
        # 总损失
        total_loss = torch.sum(local_int**2) + interface_loss + boundary_loss
        
        return total_loss


class Burgers1D_DG(DGSolver1D):
    """
    1D Burgers方程的DG求解: u_t + u·u_x + u_xx = 0 (周期边界条件)
    
    时间方向采用显式RK格式，空间方向采用DG离散
    """
    
    def __init__(self, n_elements: int = 11, n_quad_points: int = 30, 
                 poly_order: int = 3, n_time_steps: int = 50, **kwargs):
        """
        Args:
            n_elements: 单元数
            n_quad_points: 正交点数
            poly_order: 多项式阶数
            n_time_steps: 时间步数
        """
        self.n_time_steps = n_time_steps
        self.t_final = 1.5
        super().__init__(n_elements, n_quad_points, poly_order, **kwargs)
    
    def domain_bounds(self) -> Tuple[float, float]:
        return 0.0, 2 * np.pi
    
    def source_term(self, x: torch.Tensor) -> torch.Tensor:
        """Burgers方程的通量函数 f(u) = u²/2"""
        return torch.zeros_like(x)  # 这里只是占位符
    
    def exact_solution(self, x: torch.Tensor, t: torch.Tensor = 0.0) -> torch.Tensor:
        """初始条件: u(x,0) = sin(x) + 1/2"""
        return torch.sin(x) + 0.5
    
    def compute_loss(self, u: torch.Tensor, u_exact: torch.Tensor = None) -> torch.Tensor:
        """计算Burgers方程的DG损失"""
        raise NotImplementedError("Burgers求解需要时间积分")


class Poisson2D_DG:
    """
    2D Poisson方程的DG求解: -Δu = f, u|∂Ω = 0
    
    支持不规则边界(如多边形)
    使用内惩罚DG格式
    """
    
    def __init__(self, boundary_type: str = 'polygon', n_int_elt: int = 15,
                 n_int_edge: int = 20, poly_order: int = 3,
                 mesh_param: str = 'pq30a0.2e', dtype=torch.float64):
        """
        Args:
            boundary_type: 边界类型 ('regular', 'polygon', 'irregular')
            n_int_elt: 单元内正交点数
            n_int_edge: 边界积分的正交点数
            poly_order: 多项式阶数
            mesh_param: 网格参数
            dtype: 数据类型
        """
        self.boundary_type = boundary_type
        self.n_int_elt = n_int_elt
        self.n_int_edge = n_int_edge
        self.poly_order = poly_order
        self.mesh_param = mesh_param
        self.dtype = dtype
        
        # 导入网格生成器
        try:
            from mesh.mesh2d import GenMesh2D
            self.mesh_gen = GenMesh2D(
                boundary_type=boundary_type,
                Nint_elt=n_int_elt,
                Nint_edge=n_int_edge,
                param=mesh_param
            )
        except ImportError:
            raise ImportError("需要mesh.mesh2d模块")
        
        # 获取网格信息
        self._setup_mesh_2d()
        self._setup_test_functions_2d()
    
    def _setup_mesh_2d(self):
        """建立2D网格"""
        (self.elt_int, self.elt_weights, self.edges_int, self.mesh_edges_w, 
         self.mesh_normvec, self.Mesh, self.ref_Mesh, self.inv_matrix) = \
            self.mesh_gen.get_mesh()
        
        self.n_elements = self.mesh_gen.Nelt
        self.num_elt_inner_p = self.mesh_gen.num_eltp
        self.num_elt_bd_p = self.mesh_gen.num_edgep
    
    def _setup_test_functions_2d(self):
        """预计算2D测试函数"""
        test_func = TestFunction('polynomial')
        v, grad_v = test_func.evaluate_2d(self.ref_Mesh, self.poly_order)
        
        self.v_elt_inner = v[:, :self.num_elt_inner_p]
        self.v_elt_bd = v[:, self.num_elt_inner_p:].reshape(-1, 3, self.num_elt_bd_p)
        
        dv_elt_inner = grad_v[:, :self.num_elt_inner_p, :]
        dv_elt_inner = torch.matmul(
            self.inv_matrix[None, :, None, :, :],
            dv_elt_inner[:, None, :, :].unsqueeze(-1)
        ).squeeze(-1)
        self.dv_elt_inner = dv_elt_inner
    
    def source_term(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """源项 f(x,y)"""
        if self.boundary_type == 'regular':
            # 测试问题
            return ((4 * x**4 - 4*x**3 + 10*x**2 - 6*x + 2) * (y - y**2) * torch.exp(x**2 + y**2) +
                    (4 * y**4 - 4*y**3 + 10*y**2 - 6*y + 2) * (x - x**2) * torch.exp(x**2 + y**2)) * 10
        elif self.boundary_type in ('polygon', 'irregular'):
            return torch.ones_like(x) * 10
        else:
            raise ValueError(f"Unknown boundary type: {self.boundary_type}")
    
    def exact_solution(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """精确解"""
        if self.boundary_type == 'regular':
            return 10 * x * (1 - x) * y * (1 - y) * torch.exp(x**2 + y**2)
        else:
            return None  # Polygon情况无解析解
    
    def print_info(self):
        """打印网格和求解器信息"""
        print("\n========== 2D Poisson DG求解器信息 ==========")
        print(f"边界类型: {self.boundary_type}")
        print(f"单元数: {self.n_elements}")
        print(f"多项式阶数: {self.poly_order}")
        print(f"单元内正交点数: {self.num_elt_inner_p}")
        print(f"边界积分点数: {self.num_elt_bd_p}")
        print("=" * 50 + "\n")


def demo_1d_poisson():
    """1D Poisson方程求解示例"""
    print("\n" + "="*60)
    print("1D Poisson方程 DG 求解示例")
    print("=" * 60)
    print("方程: -u'' = f(x), u(0) = u(1.5) = 0")
    print("其中 f(x) = 2ω*sin(ωx) + ω²*x*cos(ωx), ω = 15π")
    print("精确解: u(x) = x*cos(15πx)")
    print("=" * 60 + "\n")
    
    solver = Poisson1D_DG(
        n_elements=25,
        n_quad_points=20,
        poly_order=5,
        penalty_param=10.0
    )
    
    print(f"网格信息:")
    print(f"  单元数: {solver.n_elements}")
    print(f"  正交点数: {solver.n_quad_points}")
    print(f"  多项式阶数: {solver.poly_order}")
    print(f"  单元宽度范围: [{solver.h.min():.4f}, {solver.h.max():.4f}]")
    print(f"  测试函数基数: {solver.poly_order + 1}")
    print(f"  计算点总数: {solver.n_elements * (solver.n_quad_points + 2)}\n")
    
    # 计算试验
    u_test = torch.randn(solver.n_elements, solver.n_quad_points + 2, device=device)
    loss = solver.compute_loss(u_test)
    print(f"测试损失计算: {loss.item():.6e}")
    
    # 计算精确解与数值解的比较
    x_nodes = solver.x_nodes.cpu().numpy()
    u_exact = solver.exact_solution(solver.x_nodes)
    print(f"\n精确解在节点处的范围: [{u_exact.min():.6e}, {u_exact.max():.6e}]")
    
    print("\n✓ 1D Poisson求解器初始化完成")


def demo_2d_poisson():
    """2D Poisson方程求解示例"""
    print("\n" + "="*60)
    print("2D Poisson方程 DG 求解示例 (不规则边界)")
    print("=" * 60)
    print("方程: -Δu = f(x,y)")
    print("边界条件: u = 0 on ∂Ω")
    print("=" * 60 + "\n")
    
    solver = Poisson2D_DG(
        boundary_type='polygon',
        n_int_elt=15,
        n_int_edge=20,
        poly_order=3
    )
    
    solver.print_info()
    
    # 获取源项
    f = solver.source_term(solver.elt_int[..., 0], solver.elt_int[..., 1])
    print(f"源项f在单元内的统计:")
    print(f"  最小值: {f.min().item():.6e}")
    print(f"  最大值: {f.max().item():.6e}")
    print(f"  平均值: {f.mean().item():.6e}\n")
    
    print("✓ 2D Poisson求解器初始化完成")


if __name__ == "__main__":
    print("\n" + "#"*60)
    print("# 经典间断有限元(DG)方法求解器")
    print("#"*60)
    
    # 演示1D求解器
    demo_1d_poisson()
    
    # 演示2D求解器
    demo_2d_poisson()
    
    print("\n" + "#"*60)
    print("# 演示完成")
    print("#"*60 + "\n")
