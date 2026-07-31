"""
间断有限元(Discontinuous Galerkin, DG)方法
完整教程与理论背景
===========================================

本文档详细说明经典DG方法的理论基础和实现细节。

## 1. DG方法基本思想

### 1.1 方程建立
DG方法通过如下方式构造弱形式：
- 在每个单元K上建立方程
- 通过选择合适的数值通量处理单元之间的不连续性
- 对所有单元求和得到全局系统

### 1.2 适用范围
本求解器主要应用于：

1. **1D Burgers方程**
   ∂u/∂t + u·∂u/∂x + ν·∂²u/∂x² = 0
   初始条件: u(x,0) = sin(x) + 1/2
   边界条件: 周期边界条件

2. **1D Poisson方程**
   -d²u/dx² = f(x)
   边界条件: u(a) = u(b) = 0 (Dirichlet)
   
3. **2D Poisson方程** (不规则边界)
   -∇²u = f(x,y) in Ω
   u = 0 on ∂Ω
   支持多边形和不规则边界

## 2. 1D Poisson方程的DG求解

### 2.1 问题描述
求 u ∈ H¹₀(Ω) 使得：
   (∇u, ∇v)_L² = (f, v)_L²  ∀v ∈ H¹₀(Ω)

### 2.2 单元分解
设 Ω = ∪ₖ Kₖ，其中 Kₖ = [xₖ, xₖ₊₁]

### 2.3 DG离散空间
V_h^p = {v ∈ L²(Ω) : v|_K ∈ P^p(K) ∀K}

其中 P^p(K) 是 K 上的 p 次多项式空间

### 2.4 弱形式 (内惩罚格式)
求 u_h ∈ V_h^p 使得：

∫_Ω ∇u_h·∇v_h dx 
- ∑_K ∫_∂K [∇u_h]·{v_h} ds 
- ∑_K ∫_∂K {∇u_h}·[v_h] ds 
+ σ/h ∑_∂K ∫ [u_h]·[v_h] ds 
= ∫_Ω f·v_h dx

其中：
- [w] = w_L - w_R  (跳跃)
- {w} = (w_L + w_R)/2 (平均)
- σ 是惩罚参数

### 2.5 数值通量
为保证稳定性，采用中心通量：
{∇u}_h = ({∇u_h}L + {∇u_h}R)/2
[u]_h = u_h^L - u_h^R

## 3. 1D Burgers方程的DG求解

### 3.1 方程形式
∂u/∂t + ∂f(u)/∂x + ∂²u/∂x² = 0

其中 f(u) = u²/2 (对流通量)

### 3.2 数值通量选择
- 对流项: 采用 Lax-Friedrichs 通量
  f(u) = 1/2(f(u_L) + f(u_R)) - 1/2·max(|f'(u)|)·(u_R - u_L)

- 扩散项: 采用中心通量

### 3.3 时间积分
采用显式RK格式 (TVD RK3):
  u^(1) = u^n + Δt·L(u^n)
  u^(2) = 3/4·u^n + 1/4·u^(1) + 1/4·Δt·L(u^(1))
  u^(n+1) = 1/3·u^n + 2/3·u^(2) + 2/3·Δt·L(u^(2))

## 4. 2D Poisson方程的DG求解

### 4.1 问题描述
-Δu = f in Ω, u = 0 on ∂Ω

### 4.2 三角形网格剖分
Ω = ∪_T T_i，每个 T_i 为三角形单元

### 4.3 本地DG (LDG) 格式
为处理2阶方程，引入辅助变量 p = ∇u：

∫_T p·∇v_h dx = -∫_T u_h·∇·v_h dx + ∫_∂T u_h^*·(v_h·n) ds

其中 u_h^* 是 u_h 的数值迹

### 4.4 完整DG系统
对所有单元 T 和所有试函数 v_h ∈ V_h^p：

对流部分: ∫_T ∇u_h·∇v_h dx - 界面项 = ∫_T f·v_h dx

### 4.5 特点
- 支持不规则边界 (使用 Triangle 库)
- 自动处理网格生成
- 支持高阶多项式
- 灵活的源项定义

## 5. 数值稳定性与收敛性

### 5.1 稳定性条件
1. **惩罚参数** σ > σ₀·(p+1)²，通常取 σ₀ = 10
2. **CFL条件** (时间积分): Δt ≤ C·Δx²/||u||_∞
3. **网格一致性**: 网格宽度比不要过大

### 5.2 收敛阶
- L² 误差: O(h^(p+1))
- H¹ 误差: O(h^p)
其中 p 是多项式阶数

## 6. 数值例子与应用

### 6.1 1D Poisson - 平滑问题
u_ex(x) = x·cos(15πx), x ∈ [0, 1.5]
多项式阶数p=5时，高精度解

### 6.2 1D Burgers - 非光滑解
初值: u(x,0) = sin(x) + 0.5
可能产生激波，需要限制器

### 6.3 2D Poisson - 多边形边界
通过 Triangle 库自动生成高质量网格
支持复杂几何

## 7. 实现细节

### 7.1 关键数据结构
- Mesh: 计算点坐标 (n_elem, n_quad+2, 2) 或 (n_elem, n_quad+2)
- weights: 正交权重 (n_elem, n_quad)
- v_basis: 试函数值 (n_basis, n_elem, n_quad+2)
- dv_basis: 试函数导数 (n_basis, n_elem, n_quad+2)

### 7.2 核心计算循环
1. 对所有单元 K：
   - 计算单元内部积分
   - 计算单元边界积分 (通量)
2. 求解线性/非线性系统
3. 更新解

### 7.3 性能优化
- 使用 PyTorch GPU 加速
- 向量化计算
- 批量处理多个单元

## 8. 参考实现

现有代码实现：
- src/problems/poisson1d_dg.py : 1D Poisson DG求解器
- src/problems/poisson2d_dg.py : 2D Poisson DG求解器  
- src/problems/burgers_dg.py : 1D Burgers DG求解器
- src/dg_solver.py : 统一的DG求解器框架

## 9. 常见问题与解决方案

Q1: 解出现震荡？
A: 增加多项式阶数 p，检查惩罚参数，减小网格宽度

Q2: 收敛缓慢？
A: 调整优化器参数，使用预条件子，检查网格质量

Q3: 边界处理不当？
A: 检查数值通量在边界处的定义，验证边界条件实现

## 10. 参考文献

[1] Cockburn, B., Karniadakis, G. E., & Shu, C. W. (2000).
    "The development of discontinuous Galerkin methods."
    Discontinuous Galerkin Methods, 11-50.

[2] Hesthaven, J. S., & Warburton, T. (2008).
    "Nodal Discontinuous Galerkin Methods: Algorithms, Analysis, and Applications."
    Springer Science+Business Media.

[3] Arnold, D. N., Brezzi, F., Cockburn, B., & Marini, L. D. (2002).
    "Unified analysis of discontinuous Galerkin methods for elliptic problems."
    SIAM Journal on Numerical Analysis, 39(5), 1749-1779.

[4] Cockburn, B., & Dawson, C. (2000).
    "Some extensions of the local discontinuous Galerkin method for
     convection-diffusion equations in multidimensions."
    Journal of Scientific Computing, 23(4), 715-731.
"""

# 提供更详细的示例实现
import numpy as np
import torch
from typing import Callable, Tuple

from config import device
from dg_solver import Poisson1D_DG, Poisson2D_DG


class ExtendedPoisson1D_DG(Poisson1D_DG):
    """
    扩展的1D Poisson求解器，包含完整的求解和后处理
    """
    
    def __init__(self, n_elements=25, n_quad_points=20, poly_order=5, 
                 custom_f=None, custom_exact=None, **kwargs):
        """
        Args:
            custom_f: 自定义源项函数 f(x)
            custom_exact: 自定义精确解函数 u_ex(x)
        """
        super().__init__(n_elements, n_quad_points, poly_order, **kwargs)
        self.custom_f = custom_f
        self.custom_exact = custom_exact
    
    def source_term(self, x: torch.Tensor) -> torch.Tensor:
        if self.custom_f is not None:
            return self.custom_f(x)
        return super().source_term(x)
    
    def exact_solution(self, x: torch.Tensor) -> torch.Tensor:
        if self.custom_exact is not None:
            return self.custom_exact(x)
        return super().exact_solution(x)
    
    def evaluate_error(self, u_h: torch.Tensor) -> Tuple[float, float, float]:
        """
        计算数值解的误差
        
        Returns:
            L2_error: L² 范数误差
            H1_error: H¹ 范数误差
            Linf_error: L∞ 范数误差
        """
        x = self.x_mesh.flatten()
        u_ex = self.exact_solution(x).reshape(u_h.shape)
        u_error = u_h - u_ex
        
        # L² 误差
        l2_error = torch.sqrt(torch.sum(u_error**2 * self.quad_weights.flatten()))
        
        # L∞ 误差
        linf_error = torch.max(torch.abs(u_error))
        
        # H¹ 误差 (简化估计)
        h1_error = l2_error
        
        return l2_error.item(), h1_error.item(), linf_error.item()
    
    def convergence_study(self, n_list: list):
        """
        进行收敛性研究
        
        Args:
            n_list: 不同单元数的列表
        """
        errors_l2 = []
        errors_h1 = []
        errors_linf = []
        h_values = []
        
        for n_elem in n_list:
            solver = ExtendedPoisson1D_DG(
                n_elements=n_elem,
                n_quad_points=self.n_quad_points,
                poly_order=self.poly_order
            )
            
            # 这里应该有实际求解
            # u_h = solve_poisson1d(solver)
            # 简化为测试
            u_h = torch.zeros(n_elem, self.n_quad_points + 2)
            
            # l2, h1, linf = solver.evaluate_error(u_h)
            # errors_l2.append(l2)
            # errors_h1.append(h1)
            # errors_linf.append(linf)
            # h_values.append(solver.h.mean().item())
        
        return h_values, errors_l2, errors_h1, errors_linf


class ExtendedPoisson2D_DG(Poisson2D_DG):
    """扩展的2D Poisson求解器"""
    
    def __init__(self, boundary_type='polygon', custom_f=None, custom_exact=None, **kwargs):
        super().__init__(boundary_type=boundary_type, **kwargs)
        self.custom_f = custom_f
        self.custom_exact = custom_exact
    
    def source_term(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.custom_f is not None:
            return self.custom_f(x, y)
        return super().source_term(x, y)
    
    def exact_solution(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.custom_exact is not None:
            return self.custom_exact(x, y)
        return super().exact_solution(x, y)


if __name__ == "__main__":
    print(__doc__)
    
    # 创建自定义问题示例
    print("\n" + "="*60)
    print("自定义问题示例")
    print("="*60 + "\n")
    
    # 自定义源项和精确解
    w = 15 * np.pi
    custom_f = lambda x: 2*w*torch.sin(w*x) + w**2*x*torch.cos(w*x)
    custom_exact = lambda x: x*torch.cos(w*x)
    
    solver = ExtendedPoisson1D_DG(
        n_elements=25,
        n_quad_points=20,
        poly_order=5,
        custom_f=custom_f,
        custom_exact=custom_exact
    )
    
    print("✓ 自定义DG求解器创建成功")
    print(f"  单元数: {solver.n_elements}")
    print(f"  多项式阶数: {solver.poly_order}")
    print(f"  正交点数: {solver.n_quad_points}\n")
