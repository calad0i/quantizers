import numpy as np


def float_quantize_np(x, m, e, e0=0.0):
    """Quantize an array to floatlet (m mantissa bits, excl. sign bit, e exponent bits) format. Tentative gradient impl."""
    m = m + 1
    eps = 1e-7

    e_req = np.floor(np.log2(np.abs(x) + eps))  # type: ignore
    _e_high = np.maximum(2.0 ** (e - 1), 1.0)
    e_low, e_high = -_e_high + e0 + 1, _e_high + e0 - 1  # type: ignore
    e_act = np.clip(e_req, e_low, e_high)
    scale = 2.0 ** (e_act - m + 1)
    sig = x / scale
    qsig = np.round(sig)
    _sig_high = 2.0**m
    _sig_high = np.where(e_high != e_act, _sig_high, _sig_high - 1)
    clip_sig = np.clip(qsig, -_sig_high, _sig_high)  # type: ignore
    qx = clip_sig * scale

    return qx


def float_decompose_np(x, m, e, e0=0):
    """Quantize an array to floatlet (m mantissa bits, excl. sign bit, e exponent bits) format. Tentative gradient impl."""
    m = m + 1
    eps = 1e-7

    e_req = np.floor(np.log2(np.abs(x) + eps))  # type: ignore
    _e_high = 2.0 ** (e - 1)
    e_low, e_high = -_e_high + e0, _e_high + e0 - 1
    e_act = np.clip(e_req, e_low + 1, e_high)
    scale = 2.0 ** (e_act - m + 1)
    sig = x / scale
    qsig = np.round(sig)
    cond = (e_act != e_high) & (np.abs(qsig) == 2.0**m)
    e_act = np.where(cond, e_act + 1, e_act)  # type: ignore
    qsig = np.where(cond, qsig / 2.0, qsig)  # type: ignore
    r_mantissa_bound = 2 - 2.0 ** (-m + 1)
    r_mantissa = np.clip(qsig * 2.0 ** (-m + 1), -r_mantissa_bound, r_mantissa_bound)
    r_exponent = e_act
    return r_mantissa, r_exponent
