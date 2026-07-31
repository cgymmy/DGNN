# DG求解器 - 快速参考卡片 (Cheatsheet)

## 🚀 安装与导入

```python
# 安装依赖
pip install torch numpy scipy matplotlib triangle

# 导入求解器
import sys
sys.path.insert(0, 'src')
from dg_solver import Poisson1D_DG, Burgers1D_DG, Poisson2D_DG
from config import device
```

---

## 📝 基础操作

### 创建求解器

```python
# 1D Poisson
solver1d = Poisson1D_DG(
    n_elements=25,      # 单元数
    n_quad_points=20,   # 正交点数
    poly_order=5,       # 多项式阶数
    penalty_param=10.0  # 惩罚参数
)

# 1D Burgers
solverb = Burgers1D_DG(
    n_elements=11,
    n_quad_points=30,
    poly_order=3,
    n_time_steps=50
)

# 2D Poisson
solver2d = Poisson2D_DG(
    boundary_type='polygon',  # 'regular', 'polygon', 'irregular'
    n_int_elt=15,
    n_int_edge=20,
    poly_order=3
)
```

### 访问属性

```python
# 网格信息
solver.x_nodes          # 单元节点 (n_elem+1,)
solver.x_centers        # 单元中心 (n_elem,)
solver.h                # 单元宽度 (n_elem,)
solver.x_mesh           # 网格点坐标 (n_elem, n_quad+2)

# 试函数
solver.v_basis          # 试函数值 (n_basis, n_elem, n_quad+2)
solver.dv_basis         # 试函数导数 (n_basis, n_elem, n_quad+2)
solver.quad_weights     # 正交权重 (n_elem, n_quad)

# 问题参数
solver.poly_order       # 多项式阶数
solver.n_elements       # 单元数
solver.n_quad_points    # 正交点数
```

---

## 🎯 常用方法

### 计算损失函数

```python
u = torch.randn(n_elem, n_quad+2)
loss = solver.compute_loss(u)
```

### 获取精确解

```python
x = solver.x_nodes
u_exact = solver.exact_solution(x)
```

### 计算源项

```python
x = solver.x_mesh
f = solver.source_term(x)
```

### 2D求解器特定方法

```python
solver2d.print_info()              # 打印信息
solver2d.source_term(x, y)         # 计算f(x,y)
```

---

## 📊 数据形状速查表

### 1D 求解器

| 变量 | 形状 | 说明 |
|------|------|------|
| x_nodes | (n_elem+1,) | 单元边界节点 |
| x_centers | (n_elem,) | 单元中心 |
| h | (n_elem,) | 单元宽度 |
| x_mesh | (n_elem, n_quad+2) | 所有计算点 |
| u | (n_elem, n_quad+2) | 解向量 |
| v_basis | (n_basis, n_elem, n_quad+2) | 试函数 |
| dv_basis | (n_basis, n_elem, n_quad+2) | 导数 |
| quad_weights | (n_elem, n_quad) | 正交权重 |

其中: `n_basis = poly_order + 1`

### 2D 求解器

| 变量 | 形状 | 说明 |
|------|------|------|
| elt_int | (n_elem, n_int_elt, 2) | 单元内正交点 |
| Mesh | (n_elem, n_points, 2) | 网格点坐标 |
| u | (n_elem, n_points) | 解向量 |
| elt_weights | (n_elem, n_int_elt) | 单元内正交权重 |

---

## 🔨 常见操作代码片段

### 1️⃣ 网格剖分调整

```python
# 粗网格
coarse = Poisson1D_DG(n_elements=10, poly_order=5)

# 细网格
fine = Poisson1D_DG(n_elements=100, poly_order=5)

# 高阶多项式
high_order = Poisson1D_DG(n_elements=10, poly_order=10)
```

### 2️⃣ 精度评估

```python
def compute_error(solver, u_numerical):
    u_exact = solver.exact_solution(solver.x_mesh)
    error = u_numerical - u_exact
    
    l2_error = torch.sqrt(torch.mean(error**2))
    linf_error = torch.max(torch.abs(error))
    
    return l2_error.item(), linf_error.item()

l2, linf = compute_error(solver, u_h)
print(f"L2 error: {l2:.6e}, L∞ error: {linf:.6e}")
```

### 3️⃣ 收敛性研究

```python
def convergence_study(n_list, poly_order):
    errors = []
    h_values = []
    
    for n_elem in n_list:
        solver = Poisson1D_DG(n_elements=n_elem, poly_order=poly_order)
        h = 1.5 / n_elem  # 网格参数
        
        # 求解并计算误差
        u_h = solve(solver)  # 需要定义求解函数
        l2, _ = compute_error(solver, u_h)
        
        errors.append(l2)
        h_values.append(h)
    
    return h_values, errors

# 运行收敛性研究
h_vals, err_vals = convergence_study([10, 20, 40, 80], poly_order=3)
```

### 4️⃣ 自定义问题

```python
from dg_theory_and_examples import ExtendedPoisson1D_DG

def my_source(x):
    return torch.exp(-x**2)

def my_exact(x):
    return torch.sin(x)

solver = ExtendedPoisson1D_DG(
    custom_f=my_source,
    custom_exact=my_exact
)
```

### 5️⃣ 可视化

```python
import matplotlib.pyplot as plt

def plot_solution(solver, u):
    x = solver.x_mesh.cpu().numpy().flatten()
    u_plot = u.cpu().numpy()
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, u_plot, 'b-', label='数值解')
    
    u_ex = solver.exact_solution(solver.x_mesh).cpu().numpy()
    plt.plot(x, u_ex, 'r--', label='精确解')
    
    plt.legend()
    plt.grid(True)
    plt.show()

plot_solution(solver, u_h)
```

---

## 🎓 参数建议

### 选择多项式阶数

| 问题类型 | 推荐阶数 | 理由 |
|---------|--------|------|
| 光滑解 | p ≥ 5 | 高精度 |
| 一般问题 | p = 3-4 | 平衡精度和效率 |
| 有激波 | p = 1-2 | 需要限制器 |
| 快速原型 | p = 2 | 计算快 |

### 选择网格密度

| 问题 | 单元数 | 说明 |
|------|--------|------|
| 测试 | 10-20 | 快速验证 |
| 一般 | 25-50 | 日常使用 |
| 精细 | 100+ | 高精度要求 |

### 选择正交点数

| 精度要求 | 正交点数 |
|---------|---------|
| 低 | 5-10 |
| 中 | 15-20 |
| 高 | 25-30 |

### 惩罚参数

```python
# 推荐设置
penalty = (poly_order + 1)**2  # 通用设置
penalty = 10.0                  # 较弱的稳定性
penalty = 100.0                 # 强稳定性
```

---

## ⚡ 性能优化技巧

### 1️⃣ GPU加速

```python
# 检查GPU
print(device)  # 输出: cuda:0 或 cpu

# 强制使用GPU
solver = Poisson1D_DG(...)  # 自动使用device配置中的GPU
```

### 2️⃣ 批量计算

```python
# 一次计算多个解
batch_size = 32
u_batch = torch.randn(batch_size, n_elem, n_quad+2)

for u in u_batch:
    loss = solver.compute_loss(u)
```

### 3️⃣ 内存优化

```python
# 避免一次性加载所有正交点
# 改用迭代方式处理

# 数据类型优化
solver_fp32 = Poisson1D_DG(..., dtype=torch.float32)  # 更快，内存少
solver_fp64 = Poisson1D_DG(..., dtype=torch.float64)  # 更精确
```

---

## 🐛 常见问题快速排查

| 问题 | 症状 | 解决方案 |
|------|------|--------|
| 数值不稳定 | NaN/Inf | 增加 penalty_param |
| 收敛慢 | 损失下降缓慢 | 增加 poly_order |
| 内存溢出 | OOM 错误 | 减少 n_elements 或 n_quad_points |
| 精度不足 | 误差大 | 增加网格密度或多项式阶数 |
| 边界条件错误 | 边界处解不对 | 检查边界项实现 |

---

## 📚 配置项完整列表

### Poisson1D_DG

```python
Poisson1D_DG(
    n_elements=25,          # int: 单元数
    n_quad_points=20,       # int: 正交点数
    poly_order=5,           # int: 多项式阶数
    penalty_param=10.0,     # float: 惩罚参数
    dtype=torch.float64     # torch.dtype: 数据类型
)
```

### Burgers1D_DG

```python
Burgers1D_DG(
    n_elements=11,          # int: 单元数
    n_quad_points=30,       # int: 正交点数
    poly_order=3,           # int: 多项式阶数
    n_time_steps=50,        # int: 时间步数
    dtype=torch.float64     # torch.dtype: 数据类型
)
```

### Poisson2D_DG

```python
Poisson2D_DG(
    boundary_type='polygon',    # str: 'regular'/'polygon'/'irregular'
    n_int_elt=15,              # int: 单元内正交点数
    n_int_edge=20,             # int: 边界正交点数
    poly_order=3,              # int: 多项式阶数
    mesh_param='pq30a0.2e',    # str: Triangle网格参数
    dtype=torch.float64        # torch.dtype: 数据类型
)
```

---

## 🎯 一键命令

```bash
# 运行所有演示
python src/dg_solver_demo.py --problem all

# 运行特定问题
python src/dg_solver_demo.py --problem poisson1d
python src/dg_solver_demo.py --problem burgers
python src/dg_solver_demo.py --problem poisson2d

# 收敛性分析
python src/dg_solver_demo.py --problem convergence

# 单元测试
python test_dg_solver.py

# 自定义参数运行
python src/dg_solver_demo.py \
    --problem poisson1d \
    --n_elem 50 \
    --poly_order 6 \
    --output_dir ./results
```

---

## 📖 文档导航

| 文档 | 内容 | 用途 |
|------|------|------|
| DG_README.md | 完整文档 | 系统学习 |
| PROJECT_SUMMARY.md | 项目总结 | 快速了解 |
| src/dg_solver.py | 源代码 | 深入理解 |
| src/dg_theory_and_examples.py | 理论 | 学习原理 |
| test_dg_solver.py | 测试 | 验证功能 |
| 本文件 | 快速参考 | 日常查阅 |

---

## 💡 最佳实践

✅ **推荐**
- 从 `n_elements=25, poly_order=3` 开始
- 逐步增加精度参数验证收敛性
- 使用 GPU 加速长期计算
- 保存模型和结果用于后续分析

❌ **避免**
- 过度增加网格密度导致内存溢出
- 使用过高的多项式阶数且网格过粗
- 忽视边界条件实现
- 在 CPU 上运行大规模问题

---

## 🔗 快速链接

- **GitHub/仓库**: [本地路径] d:\cgy\projects\DGNN-main
- **主文档**: DG_README.md
- **源代码**: src/dg_solver.py
- **演示**: src/dg_solver_demo.py

---

**更新日期:** 2024年5月11日  
**版本:** 1.0.0
