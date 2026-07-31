import numpy as np
import torch
from scipy.special import legendre


class TestFunction1D:
    def __init__(self, func_type='Polynomial'):
        self.type = func_type

    def get_value(self, x: torch.tensor, order: int = 0, x_mid=None, h=None):
        if self.type == 'Polynomial':
            return self.Poly(x=x, order=order, x_mid=x_mid, h=h)
        elif self.type == 'Legendre':
            return self.Legendre(x=x, x_mid=x_mid, order=order, h=h)
        else:
            raise ValueError(f"Unsupported function type: {self.type}")

    def Poly(self, x: torch.tensor, order: int, x_mid, h):
        x = 2 * (x - x_mid) / h
        if order == 0:
            v = torch.ones_like(x)
            dv = torch.zeros_like(x)
        else:
            v = x ** order
            dv = order * x ** (order - 1) * 2 / h
        return v, dv

    def Legendre(self, x: torch.tensor, x_mid: torch.tensor, order: int, h: torch.tensor):
        original_shape = x.shape
        x = 2 * (x - x_mid) / h
        x_np = x.cpu().numpy().flatten()
        hp = h.cpu()
        legendre_poly = legendre(order)
        v = legendre_poly(x_np)
        dv = np.gradient(v, x_np)
        v = torch.tensor(v).reshape(*original_shape)
        dv = torch.tensor(dv).reshape(*original_shape) * 2 / hp
        return v, dv
