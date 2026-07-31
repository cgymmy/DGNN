import numpy as np
import torch
import matplotlib.pyplot as plt
from triangle import triangulate

from config import device
from mesh.domains import genpolygon
from mesh.triangle_gauss import rule


class GenMesh2D:
    def __init__(self, boundary_type: str = 'regular', param: str = 'pq30a0.2e', Nint_edge: int = 20, Nint_elt: int = 10) -> None:
        self.para = param
        self.Nint_edge = Nint_edge
        self.Nint_elt = Nint_elt
        self.np_dtype = np.float64
        self.torch_dtype = torch.float64

        self.vertices, self.segments = self.set_grid_data(boundary_type)
        if self.vertices is None:
            raise NotImplementedError(f"Grid name '{boundary_type}' is not supported.")

        t = triangulate({"vertices": self.vertices, 'segments': self.segments}, self.para)
        self.points = np.array(t["vertices"], dtype=self.np_dtype)
        self.edges = np.array(t["edges"])
        self.Mesh_pinx = np.array(t["triangles"])

        self.Nv = len(self.points)
        self.Nelt = len(self.Mesh_pinx)
        self.Nedge = len(self.edges)
        self.process_edges()

    def set_grid_data(self, boundary_type: str):
        v, e = None, None
        if boundary_type == 'polygon':
            v, e = genpolygon()
        grid_data = {
            'regular': (
                [[0, 0], [0, 1], [1, 1], [1, 0]],
                [[0, 1], [1, 2], [2, 3], [3, 0]],
            ),
            'irregular': (
                [[0, 0], [1, 0], [1, 0.5], [0.5, 0.5], [0.5, 1], [0, 1]],
                [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]],
            ),
            'polygon': (v, e),
        }
        return grid_data.get(boundary_type, (None, None))

    def process_edges(self):
        tria_inx = []
        label = [[] for _ in range(self.Nedge)]
        for i in range(self.Nelt):
            v1, v2, v3 = self.Mesh_pinx[i, :]
            tria = [(v1, v2), (v2, v3), (v3, v1)]
            edge_inx = []
            for no, ei in enumerate(tria):
                e1 = ei
                e2 = (ei[1], ei[0])
                for j in range(self.Nedge):
                    if np.array_equal(e1, self.edges[j]) or np.array_equal(e2, self.edges[j]):
                        inx = j
                        label[j].append((i, no))
                edge_inx.append(inx)
            tria_inx.append(edge_inx)
        self.Mesh_einx = np.array(tria_inx)
        self.label = label
        self.inner_label = []
        self.bd_label = []
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
        ref_p, ref_w = rule(self.Nint_elt)
        ref_p = ref_p.T
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
        nodes = (nodes + 1) / 2
        edges_weights = weights[None, :] * edges_l[:, None] / 2
        edges_int = nodes[None, :, None] * edges_vec[:, None, :] + edges_p[:, None, 0, :]

        # compute mesh normvec
        mesh_normvec = np.zeros((self.Nelt, 3, 2))
        mesh_normvec[:, 0, :] = (self.points[self.Mesh_pinx[:, 1]] - self.points[self.Mesh_pinx[:, 0]]) / edges_l[self.Mesh_einx[:, 0, None]]
        mesh_normvec[:, 1, :] = (self.points[self.Mesh_pinx[:, 2]] - self.points[self.Mesh_pinx[:, 1]]) / edges_l[self.Mesh_einx[:, 1, None]]
        mesh_normvec[:, 2, :] = (self.points[self.Mesh_pinx[:, 0]] - self.points[self.Mesh_pinx[:, 2]]) / edges_l[self.Mesh_einx[:, 2, None]]
        mesh_normvec = np.stack((mesh_normvec[:, :, 1], -mesh_normvec[:, :, 0]), axis=-1)

        mesh_edges_w = edges_weights[self.Mesh_einx]

        e1 = nodes[:, None] * np.array([1, 0])[None, :]
        e2 = nodes[:, None] * np.array([-1, 1])[None, :] + np.array([1, 0])[None, :]
        e3 = nodes[:, None] * np.array([0, -1])[None, :] + np.array([0, 1])[None, :]
        ref_Mesh = np.concatenate((ref_p, e1, e2, e3), axis=0)
        Mesh = np.sum(matrix[:, None, :, :] * ref_Mesh[None, :, :, None], axis=-2) + tri_p[:, None, 0, :]

        return torch.tensor(elt_int, dtype=self.torch_dtype).to(device), \
            torch.tensor(elt_weights, dtype=self.torch_dtype).to(device), \
            torch.tensor(edges_int, dtype=self.torch_dtype).to(device), \
            torch.tensor(mesh_edges_w, dtype=self.torch_dtype).to(device), \
            torch.tensor(mesh_normvec, dtype=self.torch_dtype).to(device), \
            torch.tensor(Mesh, dtype=self.torch_dtype).to(device), \
            torch.tensor(ref_Mesh, dtype=self.torch_dtype).to(device), \
            torch.tensor(inv_matrix, dtype=self.torch_dtype).to(device)

    def print_grid_info(self):
        print(f'In the whole domain: ')
        print(f'{self.Nv} points')
        print(f'{self.Nelt} elements')
        print(f'{self.Nedge} faces/edges')

    def plot_mesh(self):
        print("Plot the mesh:")
        plt.figure(figsize=(6, 6))
        plt.triplot(self.points[:, 0], self.points[:, 1], self.Mesh_pinx)
        plt.plot(self.points[:, 0], self.points[:, 1], 'o', markersize=1)
        plt.axis('equal')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.savefig('mesh.png')
        plt.show()
