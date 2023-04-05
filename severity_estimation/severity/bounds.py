import numpy as np


def upper_bound(u, Finv_x, Fy, n, alpha):
    eps = np.sqrt(np.log(2 / alpha) / (2 * n))
    xlow = Finv_x(np.clip(u - eps, 0.0, 1.0))
    vlow = np.clip(Fy(xlow) - eps, 0.0, 1.0)
    ub = 1 - max([u + vlow - 1, 0]) / u
    return ub


def lower_bound(u, Finv_x, Fy, n, alpha):
    eps = np.sqrt(np.log(2 / alpha) / (2 * n))
    xup = Finv_x(np.clip(u + eps, 0.0, 1.0))
    vup = np.clip(Fy(xup) + eps, 0.0, 1.0)
    lb = 1 - min([u, vup]) / u
    return lb
