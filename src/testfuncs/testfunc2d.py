import torch


class TestFunction2D:
    def __init__(self, func_type: str = 'Polynomial') -> None:
        self.type = func_type

    def get_value(self, mesh: torch.tensor, order: int = 0):
        if self.type == 'Polynomial':
            return self.Poly(mesh=mesh, order=order)
        else:
            raise ValueError(f"Unsupported function type: {self.type}")

    def Poly(self, mesh: torch.tensor, order: int = 0):
        x = mesh[..., 0]
        y = mesh[..., 1]
        v, dv = [], []
        for i in range(order + 1):
            for j in range(order + 1 - i):
                v.append(x ** i * y ** j)
                dv_x = i * x ** (i - 1) * y ** j if i > 0 else torch.zeros_like(x)
                dv_y = j * x ** i * y ** (j - 1) if j > 0 else torch.zeros_like(y)
                dv.append(torch.stack([dv_x, dv_y], dim=-1))
        v = torch.stack(v, dim=0)
        dv = torch.stack(dv, dim=0)
        return v, dv
