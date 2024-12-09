from keras import ops


@ops.custom_gradient
def _float_quantize(x, m, e, e0=0.0):
    """Quantize an array to floatlet (m mantissa bits, excl. sign bit, e exponent bits) format. Tentative gradient impl."""
    m = m + 1
    eps = 1e-7

    e_req = ops.floor(ops.log2(ops.abs(x) + eps))  # type: ignore
    _e_high = ops.maximum(2.0 ** (e - 1), 1.0)
    e_low, e_high = -_e_high + e0 + 1, _e_high + e0 - 1  # type: ignore
    e_act = ops.clip(e_req, e_low, e_high)
    scale = 2.0 ** (e_act - m + 1)
    sig = x / scale
    qsig = ops.round(sig)
    _sig_high = 2.0**m
    _sig_high = ops.where(e_high != e_act, _sig_high, _sig_high - 1)
    clip_sig = ops.clip(qsig, -_sig_high, _sig_high)  # type: ignore
    qx = clip_sig * scale

    def grad(*args, upstream=None):
        if upstream is None:
            (upstream,) = args
        dy = upstream
        mask_e = e_req != e_act
        mask_m = e_req <= e_high  # type: ignore
        dm = ops.where(mask_m, scale * (sig - qsig) * dy * ops.log(2.0), 0.0)
        d_exp = ops.where(mask_e, (x - qx) * dy * ops.log(2.0), 0.0)
        de = d_exp * ops.log(2.0) * _e_high  # type: ignore
        de0 = ops.where(e_req > e_high, d_exp, -d_exp)  # type: ignore
        return dy, dm, de, de0

    return ops.stop_gradient(qx), grad


def _float_decompose(x, m, e, e0=0):
    """Quantize an array to floatlet (m mantissa bits, excl. sign bit, e exponent bits) format. Tentative gradient impl."""

    m = m + 1
    eps = 1e-7

    e_req = ops.floor(ops.log2(ops.abs(x) + eps))  # type: ignore
    _e_high = 2.0 ** (e - 1)
    e_low, e_high = -_e_high + e0, _e_high + e0 - 1
    e_act = ops.clip(e_req, e_low + 1, e_high)
    scale = 2.0 ** (e_act - m + 1)
    sig = x / scale
    qsig = ops.round(sig)
    cond = (e_act != e_high) & (ops.abs(qsig) == 2.0**m)
    e_act = ops.where(cond, e_act + 1, e_act)  # type: ignore
    qsig = ops.where(cond, qsig / 2.0, qsig)  # type: ignore
    r_mantissa_bound = 2 - 2.0 ** (-m + 1)
    r_mantissa = ops.clip(qsig * 2.0 ** (-m + 1), -r_mantissa_bound, r_mantissa_bound)
    r_exponent = e_act
    return r_mantissa, r_exponent
