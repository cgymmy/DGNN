"""
2D Poisson DGNN Visualization
==============================
Load a trained model and plot:
1. Mesh
2. DGNN prediction u(x,y)
3. Exact solution u_exact(x,y) — if available
4. Error |u - u_exact| — if available
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from config import device
from problems.poisson2d_dg import Poisson2d_dg


def main():
    P = Poisson2d_dg(
        boundary_type='polygon',
        Nint_elt=5, Nint_edge=5, order=2,
        partition=0.003,
        num_layers=2, hidden_size=20, act='tanh',
    )

    if not P.load():
        print("Model not found. Please run: run_2d.py --boundary regular")
        return

    _, mse, mae = P.loss()
    print(f"Test MSE: {mse.item():.6e}, MAE: {mae.item():.6e}")

    # Predict
    with torch.no_grad():
        Mesh = P.Mesh.clone().detach().requires_grad_(False).to(device)
        u_pred = P.model(Mesh).squeeze(-1).cpu().numpy()

    points = P.genmesh.points
    triangles = P.genmesh.Mesh_pinx
    has_exact = P.u_exact is not None

    mesh_np = Mesh.cpu().numpy()
    x_flat = mesh_np[..., 0].flatten()
    y_flat = mesh_np[..., 1].flatten()
    u_pred_flat = u_pred.flatten()

    if has_exact:
        u_exact_flat = P.u_exact.squeeze(-1).cpu().numpy().flatten()
        err_flat = np.abs(u_pred_flat - u_exact_flat)

    tri = mtri.Triangulation(x_flat, y_flat)

    os.makedirs('./plots', exist_ok=True)

    if has_exact:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes = axes.flatten()

    # 1. Mesh
    ax = axes[0]
    ax.triplot(points[:, 0], points[:, 1], triangles, 'k-', linewidth=0.6)
    ax.plot(points[:, 0], points[:, 1], 'ro', markersize=2)
    ax.set_title('Mesh', fontsize=13)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)

    # 2. Prediction
    ax = axes[1]
    tcf = ax.tricontourf(tri, u_pred_flat, levels=50, cmap='viridis')
    fig.colorbar(tcf, ax=ax)
    ax.set_title('DGNN Prediction u(x,y)', fontsize=13)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')

    if has_exact:
        # 3. Exact solution
        ax = axes[2]
        tcf = ax.tricontourf(tri, u_exact_flat, levels=50, cmap='viridis')
        fig.colorbar(tcf, ax=ax)
        ax.set_title('Exact Solution u_exact(x,y)', fontsize=13)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')

        # 4. Error
        ax = axes[3]
        tcf = ax.tricontourf(tri, err_flat, levels=50, cmap='hot')
        fig.colorbar(tcf, ax=ax)
        ax.set_title(f'|Error| (MSE={mse.item():.2e}, MAE={mae.item():.2e})', fontsize=13)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')

    plt.suptitle(f'2D Poisson DGNN Results ({P.name})', fontsize=15)
    plt.tight_layout()
    savepath = f'./plots/{P.name}.png'
    plt.savefig(savepath, dpi=200, bbox_inches='tight')
    print(f"Saved: {savepath}")
    plt.show()


if __name__ == '__main__':
    main()
