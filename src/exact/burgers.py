import numpy as np
from scipy.optimize import fsolve


def solve_w(x, t, w0):
    func = lambda w: w - np.sin(x - (w + 0.5) * t)
    w_solution = fsolve(func, w0)
    return w_solution[0]


def burgers_exact_scalar(x, t):
    left = solve_w(x, t, 0.8) + 0.5
    right = solve_w(x, t, -0.8) + 0.5
    if x <= 0.5 * t + np.pi:
        return left
    elif x >= 0.5 * t + np.pi:
        return right
    else:
        return (left + right) / 2


burgers_exact = np.vectorize(burgers_exact_scalar)
