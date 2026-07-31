"""
DGNN 2D 训练入口
================

支持问题:
1. 2D Poisson 方程: -Δu = f(x,y), Dirichlet BC (规则/多边形/不规则区域)

原理:
- 2D 问题需要在三角形单元上积分，边界是二维区域的闭合曲线
- 网格剖分 (Delaunay三角化) 是区别于1D的关键步骤
- 模型输入: (x,y)，每个单元有独立的 MLP
"""

import argparse

from problems.poisson2d_dg import Poisson2d_dg


def main():
    parser = argparse.ArgumentParser(
        description='DGNN 2D 训练入口',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            示例用法:
            python run_2d.py --problem poisson2d
            python run_2d.py --problem poisson2d --boundary irregular --partition 0.1 --num_layers 3 --hidden_size 50
            python run_2d.py --problem poisson2d --boundary regular --order 4
        """
    )
    parser.add_argument('--problem', type=str, default='poisson2d',
                        choices=['poisson2d'],
                        help='问题类型')
    # 网格/区域参数
    parser.add_argument('--boundary', type=str, default='irregular',
                        choices=['regular', 'polygon', 'irregular'],
                        help='边界类型')
    parser.add_argument('--partition', type=float, default=0.05,
                        help='网格最小单元尺寸')
    parser.add_argument('--Nint_elt', type=int, default=15, help='单元内正交点数')
    parser.add_argument('--Nint_edge', type=int, default=20, help='边界积分点数')
    parser.add_argument('--order', type=int, default=3, help='测试函数多项式阶数')
    # 网络参数
    parser.add_argument('--num_layers', type=int, default=2, help='网络层数')
    parser.add_argument('--hidden_size', type=int, default=20, help='隐藏层大小')
    parser.add_argument('--act', type=str, default='tanh', help='激活函数')
    # 损失权重
    parser.add_argument('--sigma_eq', type=int, default=1, help='方程残差损失权重')
    parser.add_argument('--sigma_flux', type=int, default=1, help='通量损失权重')
    parser.add_argument('--sigma_bd', type=int, default=1, help='边界损失权重')

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print(f"2D Poisson 方程 DGNN 训练 (边界类型: {args.boundary})")
    print("=" * 60)

    P = Poisson2d_dg(
        boundary_type=args.boundary,
        Nint_elt=args.Nint_elt,
        Nint_edge=args.Nint_edge,
        order=args.order,
        partition=args.partition,
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        act=args.act,
        sigma_eq=args.sigma_eq,
        sigma_bd=args.sigma_bd,
        sigma_flux=args.sigma_flux,
    )
    P.load()
    P.train()
    P.load()


if __name__ == '__main__':
    main()
