import torch
from utilities import *
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from problems import *
from matplotlib.path import Path



# P_dgnet = Possion2d_dg(boundary_type='polygon', Nint_elt=15, Nint_edge=15, order=3, partition=0.01, num_layers=2, hidden_size=20, act='tanh')
# print(P_dgnet.name)
# P_dgnet.genmesh.print_grid_info()
# P_dgnet.genmesh.plot_mesh()
# P_dgnet.load()



# vtx = np.array(P_dgnet.genmesh.vertices)
# path = Path(vtx)
# xmin, ymin = vtx.min(axis=0)
# xmax, ymax = vtx.max(axis=0)
# x = np.linspace(-1, 1, 2000)
# y = np.linspace(-0.9, 1.1, 2000)
# X, Y = np.meshgrid(x, y)
# mask = path.contains_points(np.vstack([X.ravel(), Y.ravel()]).T).reshape(X.shape)
# m = np.where(mask, np.nan, 0)
# cmap2 = plt.get_cmap('jet').copy()
# cmap2.set_bad(alpha=0) 
# cmap2.set_under('white') 
# # 第一层
# Mesh = P_dgnet.Mesh
# u = P_dgnet.model(Mesh)
# u_exact = P_dgnet.u_exact
# Mesh = Mesh.detach().cpu().numpy().reshape(-1, 2)
# udg_pred = u.detach().cpu().numpy().flatten()
# u_exact = u_exact.detach().cpu().numpy().flatten()


# plt.figure(figsize=(8, 6))
# plt.tricontourf(Mesh[:, 0], Mesh[:, 1], abs(udg_pred-u_exact), levels=100, cmap='jet')
# # plt.tricontourf(Mesh[:, 0], Mesh[:, 1], udg_pred, levels=100, cmap='jet')
# plt.colorbar()
# plt.pcolormesh(X, Y, m, cmap=cmap2, vmin=0.2, vmax=1)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.axis('equal')
# plt.show()
# plt.savefig("./pics/" + P_dgnet.name + '.png')


# N_partitions = [0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
# mse_list, mae_list = [], []
# for partition in N_partitions:
#     P_dgnet = Possion2d_dg(boundary_type='polygon', Nint_elt=15, Nint_edge=20, order=2, partition=partition, num_layers=2, hidden_size=20, act='tanh')
#     P_dgnet.load()
#     _, mse, mae = P_dgnet.loss()
#     mse_list.append(mse)
#     mae_list.append(mae)

# # 绘制损失曲线
# plt.figure(figsize=(10, 5))
# plt.plot(N_partitions, mse_list, marker='o', linestyle='-', color='b')
# plt.plot(N_partitions, mae_list, marker='o', linestyle='-', color='r')
# plt.xlabel('Partition')
# plt.ylabel('Loss')
# plt.title('Loss of Possion2d_dg')
# plt.legend()
# plt.show()
# plt.savefig("./pics/loss.png")

P_dgnet = Possion2d_dg(boundary_type='polygon', Nint_elt=15, Nint_edge=20, order=3, partition=0.01, num_layers=2, hidden_size=20, act='tanh')
# P_dgnet.genmesh.print_grid_info()
P_dgnet.load()
_, mse, mae = P_dgnet.loss()
print(mse, mae)