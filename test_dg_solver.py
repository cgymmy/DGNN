"""
DG求解器测试脚本
==================

验证各个求解器的正确性和功能完整性
"""

import sys
import os
import torch
import numpy as np

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config import device
from dg_solver import (
    QuadratureRule, TestFunction,
    Poisson1D_DG, Burgers1D_DG
)


class TestDGSolver:
    """DG求解器单元测试"""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.test_results = []
    
    def test(self, test_name: str, test_func):
        """运行单个测试"""
        try:
            test_func()
            self.passed += 1
            status = "✓ 通过"
            self.test_results.append((test_name, True, None))
            print(f"{status} - {test_name}")
        except Exception as e:
            self.failed += 1
            status = "✗ 失败"
            self.test_results.append((test_name, False, str(e)))
            print(f"{status} - {test_name}")
            print(f"        错误: {e}\n")
    
    def run_all(self):
        """运行所有测试"""
        print("\n" + "="*70)
        print("DG求解器单元测试")
        print("="*70 + "\n")
        
        # 正交规则测试
        print("[1] 正交规则测试")
        print("-" * 70)
        self.test("Legendre-Gauss正交", self.test_quadrature)
        
        # 试函数测试
        print("\n[2] 试函数空间测试")
        print("-" * 70)
        self.test("1D多项式基函数", self.test_testfunc_1d)
        self.test("2D多项式基函数", self.test_testfunc_2d)
        
        # 网格生成测试
        print("\n[3] 网格生成测试")
        print("-" * 70)
        self.test("1D网格初始化", self.test_mesh_1d)
        
        # 损失函数测试
        print("\n[4] 损失函数计算测试")
        print("-" * 70)
        self.test("Poisson1D损失计算", self.test_loss_poisson1d)
        self.test("Burgers1D初始化", self.test_init_burgers1d)
        
        # 2D求解器测试
        print("\n[5] 2D求解器测试")
        print("-" * 70)
        self.test_2d_solver()
        
        # 精度测试
        print("\n[6] 精度验证测试")
        print("-" * 70)
        self.test("精确解评估", self.test_exact_solutions)
        
        # 打印总结
        self.print_summary()
    
    def test_quadrature(self):
        """测试正交规则"""
        quad = QuadratureRule()
        
        # Legendre-Gauss正交
        nodes, weights = quad.legendre_gauss_1d(10)
        
        assert len(nodes) == 10, "节点数不匹配"
        assert len(weights) == 10, "权重数不匹配"
        assert np.allclose(np.sum(weights), 2.0), "权重和应为2"
        
        # 验证多项式正交性
        f = lambda x: x**4  # 4次多项式
        integral_quad = np.sum(f(nodes) * weights)
        integral_exact = 2.0 / 5.0  # ∫_{-1}^1 x^4 dx
        assert np.allclose(integral_quad, integral_exact, atol=1e-10), \
            f"正交精度不足: {integral_quad} vs {integral_exact}"
    
    def test_testfunc_1d(self):
        """测试1D试函数空间"""
        testfunc = TestFunction('polynomial')
        
        # 参考元上的点
        x = torch.linspace(-1, 1, 11, dtype=torch.float64)
        x_mid = torch.tensor(0.0, dtype=torch.float64)
        h = torch.tensor(2.0, dtype=torch.float64)
        
        # 计算不同阶数的基函数
        for order in range(5):
            v, dv = testfunc.evaluate_1d(x, order, x_mid, h)
            
            assert v.shape == x.shape, f"基函数形状错误 (阶数{order})"
            assert dv.shape == x.shape, f"导数形状错误 (阶数{order})"
            assert torch.isfinite(v).all(), f"基函数包含非有限值 (阶数{order})"
            assert torch.isfinite(dv).all(), f"导数包含非有限值 (阶数{order})"
    
    def test_testfunc_2d(self):
        """测试2D试函数空间"""
        testfunc = TestFunction('polynomial')
        
        # 参考三角形上的点
        points = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.25]
        ], dtype=torch.float64)
        
        # 计算不同阶数的基函数
        for order in range(1, 4):
            v, grad_v = testfunc.evaluate_2d(points, order)
            
            n_basis = (order + 1) * (order + 2) // 2
            assert v.shape[0] == n_basis, f"2D基函数个数错误 (阶数{order})"
            assert grad_v.shape[0] == n_basis, f"2D梯度基函数个数错误"
            assert grad_v.shape[-1] == 2, f"2D梯度维数应为2"
    
    def test_mesh_1d(self):
        """测试1D网格生成"""
        solver = Poisson1D_DG(
            n_elements=10,
            n_quad_points=8,
            poly_order=3
        )
        
        # 检查网格属性
        assert solver.x_nodes.shape[0] == 11, "节点数应为n_elem+1"
        assert solver.x_centers.shape[0] == 10, "单元中点数应为n_elem"
        assert solver.h.shape[0] == 10, "单元宽度数应为n_elem"
        assert solver.x_mesh.shape == (10, 10), "网格点数应为(n_elem, n_quad+2)"
        
        # 检查网格值的合理性
        assert torch.all(solver.x_nodes[:-1] < solver.x_nodes[1:]), "节点应单调递增"
        assert torch.all(solver.h > 0), "单元宽度应为正"
        assert torch.allclose(
            torch.diff(solver.x_nodes),
            solver.h,
            rtol=1e-10
        ), "单元宽度计算错误"
    
    def test_loss_poisson1d(self):
        """测试Poisson1D损失函数"""
        solver = Poisson1D_DG(
            n_elements=15,
            n_quad_points=10,
            poly_order=3
        )
        
        # 创建测试解
        u = torch.randn(15, 12, device=device, dtype=solver.dtype)
        
        # 计算损失
        loss = solver.compute_loss(u)
        
        assert torch.isfinite(loss), "损失包含非有限值"
        assert loss.item() > 0, "损失应为正值"
        
        # 测试损失对输入的连续性
        u1 = torch.randn(15, 12, device=device, dtype=solver.dtype)
        u2 = u1 + 1e-6 * torch.randn_like(u1)
        
        loss1 = solver.compute_loss(u1)
        loss2 = solver.compute_loss(u2)
        
        assert torch.isclose(loss1, loss2, rtol=1e-3), "损失对输入应连续"
    
    def test_init_burgers1d(self):
        """测试Burgers1D初始化"""
        solver = Burgers1D_DG(
            n_elements=8,
            n_quad_points=15,
            poly_order=2,
            n_time_steps=20
        )
        
        # 检查属性
        assert solver.n_time_steps == 20, "时间步数不匹配"
        assert solver.t_final == 1.5, "最终时间不匹配"
        assert solver.n_elements == 8, "单元数不匹配"
        assert solver.poly_order == 2, "多项式阶数不匹配"
        
        # 检查初值
        u0 = solver.exact_solution(solver.x_mesh)
        assert torch.isfinite(u0).all(), "初值包含非有限值"
        # u(x,0) = sin(x) + 0.5, 范围应该是 [0.5-1, 0.5+1] = [-0.5, 1.5]
        assert u0.min() > -1.0, "初值最小值不合理"
        assert u0.max() < 2.0, "初值最大值不合理"
    
    def test_2d_solver(self):
        """测试2D求解器"""
        try:
            from dg_solver import Poisson2D_DG
            
            solver = Poisson2D_DG(
                boundary_type='regular',
                n_int_elt=10,
                n_int_edge=15,
                poly_order=2
            )
            
            print("✓ 通过 - 2D Poisson求解器初始化")
            self.passed += 1
            
            # 检查网格属性
            assert solver.n_elements > 0, "单元数应大于0"
            assert solver.num_elt_inner_p > 0, "单元内点数应大于0"
            
            print("✓ 通过 - 2D网格属性检查")
            self.passed += 1
            
        except ImportError as e:
            print(f"⚠ 跳过 - 2D求解器测试 (缺失依赖: {e})")
        except Exception as e:
            print(f"✗ 失败 - 2D求解器: {e}")
            self.failed += 1
    
    def test_exact_solutions(self):
        """测试精确解"""
        solver_1d = Poisson1D_DG(n_elements=20, n_quad_points=15, poly_order=4)
        
        # 获取精确解
        x_test = solver_1d.x_nodes
        u_exact = solver_1d.exact_solution(x_test)
        
        # 边界条件检查
        assert torch.abs(u_exact[0]) < 1e-6, "左边界条件不满足"
        assert torch.abs(u_exact[-1]) < 1e-6, "右边界条件不满足"
        
        # 解的合理性
        assert torch.isfinite(u_exact).all(), "精确解包含非有限值"
        
        # 源项检查
        f = solver_1d.source_term(x_test)
        assert torch.isfinite(f).all(), "源项包含非有限值"
        assert f.abs().max() > 0, "源项不能全为0"
    
    def print_summary(self):
        """打印测试总结"""
        total = self.passed + self.failed
        
        print("\n" + "="*70)
        print("测试总结")
        print("="*70)
        print(f"总测试数: {total}")
        print(f"通过: {self.passed} ✓")
        print(f"失败: {self.failed} ✗")
        print(f"成功率: {100*self.passed/total:.1f}%")
        
        if self.failed > 0:
            print("\n失败的测试:")
            for name, passed, error in self.test_results:
                if not passed:
                    print(f"  - {name}")
                    if error:
                        print(f"    {error}")
        
        print("="*70 + "\n")
        
        return self.failed == 0


def main():
    """主函数"""
    print("\n" + "#"*70)
    print("# DG求解器测试套件")
    print("#"*70)
    print(f"\n计算设备: {device}")
    print(f"PyTorch版本: {torch.__version__}\n")
    
    # 运行测试
    tester = TestDGSolver()
    success = tester.run_all()
    
    # 返回状态码
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
