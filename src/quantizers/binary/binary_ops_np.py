import numpy as np


def binary_quantize_np(x):
    return np.where(x > 0, 1., -1.)


def ternary_quantize_np(x):
    return np.where(x > 0.5, 1., np.where(x < -0.5, -1., 0.))
