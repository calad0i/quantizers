import numpy as np


def binary_quantize_np(x):
    return np.where(x > 0, 1.0, -1.0)


def ternary_quantize_np(x):
    return np.where(x > 0.5, 1.0, np.where(x < -0.5, -1.0, 0.0))
