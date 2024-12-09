def float_quantize(x, m, e, e0=0.0):
    """Quantize an array to floatlet (m mantissa bits, excl. sign bit, e exponent bits) format. Tentative gradient impl."""
    from ._float_point_ops import _float_quantize

    return _float_quantize(x, m, e, e0)


def float_decompose(x, m, e, e0=0):
    """Quantize an array to floatlet (m mantissa bits, excl. sign bit, e exponent bits) format. Tentative gradient impl."""
    from ._float_point_ops import _float_decompose

    return _float_decompose(x, m, e, e0)
