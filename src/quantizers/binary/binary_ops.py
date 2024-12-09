def binary_quantize(x):
    from ._binary_ops import _binary_quantize

    return _binary_quantize(x)


def ternary_quantize(x):
    from ._binary_ops import _ternary_quantize

    return _ternary_quantize(x)
