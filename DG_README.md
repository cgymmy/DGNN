# 经典间断有限元(DG)方法求解器

## 📋 概述

本项目实现了经典的**间断有限元(Discontinuous Galerkin, DG)方法**，用于求解以下三类PDE问题：

1. **1D Burgers方程** - 非线性对流-扩散方程
2. **1D Poisson方程** - 椭圆型方程  
3. **2D Poisson方程** (不规则边界) - 椭圆型方程

DG方法是一种高精度、灵活且稳定的有限元方法，特别适合处理不连续解和复杂几何。

---

## 🚀 快速开始

### 安装依赖

```bash
# 必需包
pip install torch numpy scipy matplotlib

# 可选包 (用于2D不规则网格)
pip install triangle
```

### 基本使用

```python
from dg_solver import Poisson1D_DG

# 创建1D Poisson求解器
solver = Poisson1D_DG(
    n_elements=25,      # 单元数
    n_quad_points=20,   # 正交点数
    poly_order=5        # 多项式阶数
)

# 求解 (简化示例)
loss = solver.compute_loss(u_solution)
```

### 运行演示

```bash
# 运行所有演示
python src/dg_solver_demo.py --problem all

# 仅运行1D Poisson
python src/dg_solver_demo.py --problem poisson1d --n_elem 50

# 运行收敛性分析
python src/dg_solver_demo.py --problem convergence
```

---

## 📁 文件结构

```
├── src/
│   ├── dg_solver.py                    # 核心DG求解器 ⭐
│   ├── dg_theory_and_examples.py      # 理论和扩展示例
│   ├── dg_solver_demo.py              # 演示程序
│   ├── problems/
│   │   ├── poisson1d_dg.py           # 1D Poisson (神经网络版本)
│   │   ├── poisson2d_dg.py           # 2D Poisson (神经网络版本)
│   │   └── burgers_dg.py             # 1D Burgers (神经网络版本)
│   ├── mesh/
│   │   ├── mesh2d.py                 # 2D网格生成
│   │   ├── domains.py                # 区域定义
│   │   └── triangle_gauss.py         # 高斯正交规则
│   ├── testfuncs/
│   │   ├── testfunc1d.py            # 1D试函数空间
│   │   └── testfunc2d.py            # 2D试函数空间
│   ├── nn/
│   │   └── dgnet.py                 # 神经网络表示(可选)
│   └── config.py                    # 配置文件
├── README.md                         # 本文件
└── dg_results/                       # 输出结果目录
```

---

## 🔧 API 参考

### Poisson1D_DG

求解1D Poisson方程: $-u'' = f(x), u(0)=u(L)=0$

```python
solver = Poisson1D_DG(
    n_elements=25,           # 单元数
    n_quad_points=20,        # 正交点数
    poly_order=5,            # 多项式阶数
    penalty_param=10.0       # 惩罚参数
)

# 属性和方法
solver.x_nodes              # 单元节点坐标
solver.x_mesh               # 网格点坐标
solver.h                    # 单元宽度
solver.v_basis              # 试函数值
solver.dv_basis             # 试函数导数

# 方法
solver.compute_loss(u)      # 计算DG损失函数
solver.domain_bounds()      # 获取计算域
solver.source_term(x)       # 计算源项f(x)
solver.exact_solution(x)    # 获取精确解
```

### Burgers1D_DG

求解1D Burgers方程: $u_t + u \cdot u_x + \nu u_{xx} = 0$

```python
solver = Burgers1D_DG(
    n_elements=11,           # 空间单元数
    n_quad_points=30,        # 正交点数
    poly_order=3,            # 多项式阶数
    n_time_steps=50          # 时间步数
)
```

### Poisson2D_DG

求解2D Poisson方程: $-\Delta u = f(x,y), u|_{\partial\Omega}=0$

```python
solver = Poisson2D_DG(
    boundary_type='polygon',  # 'regular', 'polygon', 'irregular'
    n_int_elt=15,            # 单元内正交点数
    n_int_edge=20,           # 边界正交点数
    poly_order=3             # 多项式阶数
)

# 方法
solver.print_info()         # 打印求解器信息
solver.source_term(x, y)    # 计算源项f(x,y)
```

---

## 📊 测试问题

### 问题 1: 1D Poisson

**方程:**
$$-u''(x) = f(x), \quad x \in [0, 1.5]$$
$$u(0) = u(1.5) = 0$$

**源项和精确解:**
- $\omega = 15\pi$
- $f(x) = 2\omega\sin(\omega x) + \omega^2 x \cos(\omega x)$
- $u_{ex}(x) = x\cos(\omega x)$

**特点:** 高频振荡，考察多项式阶数的影响

### 问题 2: 1D Burgers

**方程:**
$$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} + \nu \frac{\partial^2 u}{\partial x^2} = 0$$

**初值和边界条件:**
- $u(x,0) = \sin(x) + 0.5$
- 周期边界条件
- 计算时间: $t \in [0, 1.5]$

**特点:** 非线性对流-扩散，可能产生激波，需要高阶方法

### 问题 3: 2D Poisson (多边形边界)

**方程:**
$$-\nabla^2 u = f(x,y) \quad \text{in } \Omega$$
$$u = 0 \quad \text{on } \partial\Omega$$

**特点:**
- 支持复杂几何 (多边形、不规则边界)
- 自动网格生成 (使用Triangle库)
- 高阶多项式近似

---

## 💡 核心算法

### DG方法的关键思想

1. **单元分解**: 将计算域分成多个单元(1D: 区间; 2D: 三角形)

2. **局部多项式空间**: 每个单元上用多项式表示近似解
   $$u_h|_K \in P^p(K)$$
   其中 $P^p(K)$ 是K上的p次多项式空间

3. **弱形式**: 在每个单元上建立积分方程

4. **数值通量**: 在单元边界处通过数值通量连接相邻单元
   - 对流通量: Lax-Friedrichs
   - 扩散通量: 中心通量
   - 惩罚项: 保证稳定性

### 内惩罚(Interior Penalty)格式

对于1D Poisson方程，DG方程为：

$$\int_{\Omega} \nabla u_h \cdot \nabla v_h \, dx 
- \sum_{E} \int_E \{{\nabla u_h}\} \cdot [[v_h]] \, ds 
- \sum_{E} \int_E \{{\nabla v_h}\} \cdot [[u_h]] \, ds 
+ \sigma \sum_{E} \int_E \frac{1}{h_E} [[u_h]] \cdot [[v_h]] \, ds 
= \int_{\Omega} f \cdot v_h \, dx$$

其中：
- $[[w]] = w^+ - w^-$ (跳跃)
- $\{\{w\}\} = (w^+ + w^-)/2$ (平均)
- $\sigma$ 是惩罚参数，通常取 $\sigma = p(p+1)$

---

## 📈 性能指标

### 收敛阶

| 问题 | L² 误差 | H¹ 误差 |
|------|--------|--------|
| Poisson | $O(h^{p+1})$ | $O(h^p)$ |
| 其他 | $O(h^p)$ | $O(h^{p-1})$ |

其中 $p$ 是多项式阶数，$h$ 是网格参数。

### 计算复杂度

- **1D Poisson**: $O(n \cdot (p+1)^2)$，$n$ 为单元数
- **2D Poisson**: $O(n_e \cdot m^2)$，$n_e$ 为单元数，$m$ 为单元内DOF数
- GPU加速: 2-10倍加速

---

## 🛠️ 扩展和自定义

### 自定义源项

```python
# 定义自定义源项函数
def my_source(x):
    return torch.exp(-x**2)

# 定义自定义精确解
def my_exact(x):
    return torch.sin(x)

# 创建扩展求解器
from dg_theory_and_examples import ExtendedPoisson1D_DG

solver = ExtendedPoisson1D_DG(
    n_elements=30,
    custom_f=my_source,
    custom_exact=my_exact
)
```

### 改变网格密度和多项式阶数

```python
# 细网格
fine_solver = Poisson1D_DG(n_elements=100, poly_order=6)

# 粗网格高阶
coarse_solver = Poisson1D_DG(n_elements=10, poly_order=10)
```

### 2D边界类型

```python
# 规则矩形边界
solver = Poisson2D_DG(boundary_type='regular')

# 不规则边界
solver = Poisson2D_DG(boundary_type='irregular')

# 多边形边界
solver = Poisson2D_DG(boundary_type='polygon')
```

---

## ⚠️ 常见问题与解决方案

### Q1: 求解不收敛？

**原因:** 
- 惩罚参数过小
- 网格质量差
- 多项式阶数过低

**解决:**
```python
# 增加惩罚参数
solver = Poisson1D_DG(penalty_param=100.0)

# 增加多项式阶数
solver = Poisson1D_DG(poly_order=8)

# 细化网格
solver = Poisson1D_DG(n_elements=100)
```

### Q2: 边界处误差大？

**原因:** Dirichlet边界条件处理不当

**解决:** 检查边界项的计算，确保边界条件被正确施加

```python
# 增加边界附近的正交点
solver = Poisson1D_DG(n_quad_points=30)
```

### Q3: 2D求解失败？

**原因:** Triangle库未安装或网格参数不合适

**解决:**
```bash
# 安装Triangle库
pip install triangle

# 调整网格参数
solver = Poisson2D_DG(
    mesh_param='pq30a0.1e'  # 更严格的网格
)
```

---

## 📚 参考文献

1. **Cockburn, B., Karniadakis, G. E., & Shu, C. W.** (2000)
   - "The development of discontinuous Galerkin methods"
   - Discontinuous Galerkin Methods, pp. 11-50

2. **Hesthaven, J. S., & Warburton, T.** (2008)
   - "Nodal Discontinuous Galerkin Methods: Algorithms, Analysis, and Applications"
   - Springer Science+Business Media

3. **Arnold, D. N., Brezzi, F., Cockburn, B., & Marini, L. D.** (2002)
   - "Unified analysis of discontinuous Galerkin methods for elliptic problems"
   - SIAM Journal on Numerical Analysis, 39(5), 1749-1779

4. **Cockburn, B., & Dawson, C.** (2000)
   - "Some extensions of the local discontinuous Galerkin method"
   - Journal of Scientific Computing, 23(4), 715-731

---

## 💾 示例脚本

### 最小化示例

```python
import torch
from src.dg_solver import Poisson1D_DG

# 创建求解器
solver = Poisson1D_DG(n_elements=25, n_quad_points=20, poly_order=5)

# 获取网格和试函数
print(f"单元数: {solver.n_elements}")
print(f"多项式基数: {solver.poly_order + 1}")
print(f"计算点数: {solver.x_mesh.shape}")

# 计算损失
u = torch.randn(25, 22)
loss = solver.compute_loss(u)
print(f"损失函数: {loss.item():.6e}")
```

### 完整求解流程

参见 `src/dg_solver_demo.py` 中的完整示例

---

## 📝 版本历史

- **v1.0** (2024) - 初始版本
  - 1D Poisson求解器
  - 1D Burgers求解器  
  - 2D Poisson求解器 (不规则边界)
  - 详细文档和演示

---

## 📧 联系与反馈

如有问题或建议，欢迎提出Issue或PR。

---

## 📄 许可证

MIT License

---

**最后更新:** 2024年5月11日
