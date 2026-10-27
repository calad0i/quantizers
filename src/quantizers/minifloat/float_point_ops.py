import math
from typing import Any

from .._backends import get_backend


def float_quantize(x, m, e, e0: Any = 0.0):
    """Quantize an array to floatlet (m mantissa bits, excl. sign bit, e exponent bits) format. Tentative gradient impl."""
    bk = get_backend(x)
    m, e, e0 = (bk.asarray(v, like=x) for v in (m, e, e0))
    m = m + 1
    eps = 1e-7

    e_req = bk.floor(bk.log2(bk.abs(x) + eps))
    _e_high = 2.0 ** (e - 1)
    _e_high = bk.where(_e_high > 1.0, _e_high, 1.0)
    e_low, e_high = -_e_high + e0 + 1, _e_high + e0 - 1
    e_act = bk.clip(e_req, e_low, e_high)
    scale = 2.0 ** (e_act - m + 1)
    sig = x / scale
    qsig = bk.round_conv(sig)
    _sig_high = 2.0**m
    _sig_high = bk.where(e_high != e_act, _sig_high, _sig_high - 1)
    clip_sig = bk.clip(qsig, -_sig_high, _sig_high)
    qx = clip_sig * scale

    if not bk.differentiable:
        return qx

    # Gradient passthrough on x, surrogate gradients on m, e, e0
    ln2 = math.log(2.0)
    mask_e = e_req != e_act
    mask_m = e_req <= e_high
    dm = bk.where(mask_m, scale * (sig - qsig) * ln2, 0.0)
    d_exp = bk.where(mask_e, (x - qx) * ln2, 0.0)
    de = d_exp * ln2 * _e_high
    de0 = bk.where(e_req > e_high, d_exp, -d_exp)
    return bk.surrogate(qx, (x, None), (m, dm), (e, de), (e0, de0))


def float_decompose(x, m, e, e0: Any = 0.0):
    """Decompose an array into (mantissa, exponent) of floatlet (m mantissa bits, excl. sign bit, e exponent bits) format."""
    bk = get_backend(x)
    m, e, e0 = (bk.asarray(v, like=x) for v in (m, e, e0))
    m = m + 1
    eps = 1e-7

    e_req = bk.floor(bk.log2(bk.abs(x) + eps))
    _e_high = 2.0 ** (e - 1)
    e_low, e_high = -_e_high + e0, _e_high + e0 - 1
    e_act = bk.clip(e_req, e_low + 1, e_high)
    scale = 2.0 ** (e_act - m + 1)
    sig = x / scale
    qsig = bk.round_conv(sig)
    cond = (e_act != e_high) & (bk.abs(qsig) == 2.0**m)
    e_act = bk.where(cond, e_act + 1, e_act)
    qsig = bk.where(cond, qsig / 2.0, qsig)
    r_mantissa_bound = 2 - 2.0 ** (-m + 1)
    r_mantissa = bk.clip(qsig * 2.0 ** (-m + 1), -r_mantissa_bound, r_mantissa_bound)
    r_exponent = e_act
    return r_mantissa, r_exponent
