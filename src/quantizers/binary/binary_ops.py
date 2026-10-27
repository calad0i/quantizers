from .._backends import get_backend


def binary_quantize(x):
    """Quantize to {-1, 1}, 0 mapped to -1. Gradient of tanh(x)."""
    bk = get_backend(x)
    one = bk.asarray(1.0, like=x)
    r = bk.where(x > 0, one, -one)
    return bk.surrogate(r, (x, 1 - bk.tanh(x) ** 2))


def ternary_quantize(x):
    """Quantize to {-1, 0, 1}, ties at +-0.5 mapped to 0. Gradient of tanh(x)."""
    bk = get_backend(x)
    one = bk.asarray(1.0, like=x)
    r = bk.where(x > 0.5, one, bk.where(x < -0.5, -one, 0.0))
    return bk.surrogate(r, (x, 1 - bk.tanh(x) ** 2))
