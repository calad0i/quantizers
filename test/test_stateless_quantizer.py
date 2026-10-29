import numpy as np
import pytest
from conftest import FWDict

from quantizers import (
    FixedPointQuantizer,
    binary_quantize,
    float_decompose,
    float_quantize,
    ternary_quantize,
)


@pytest.fixture(scope='module')
def x_kif():
    N = (100000,)
    k = np.random.rand(N[0]) > 0.5
    i = np.random.randint(-4, 8, N).astype(np.float32)
    f = np.random.randint(-4, 8, N).astype(np.float32)
    x = np.random.normal(size=N).astype(np.float32)
    f = np.maximum(f, np.where(k, -i, -i + 1))
    return x, k, i, f


@pytest.fixture(scope='module')
def xxx_mee0():
    N = (100000,)
    M = np.random.randint(1, 9, N).astype(np.float32)
    E = np.random.randint(1, 4, N).astype(np.float32)
    E0 = np.random.randint(-8, 8, N).astype(np.float32)
    x = np.random.uniform(-1.0, 1.0, N).astype(np.float32)

    sign = np.sign(x)
    x = np.abs(x)
    m_eps = 2.0**-M
    _max = (2 - m_eps) * 2.0 ** (2.0 ** (E - 1) - 1 + E0)
    _min_pos_normal = 2.0 ** (-(2.0 ** (E - 1)) + E0 + 1)
    _min_pos_subnormal = m_eps * 2.0 ** (-(2.0 ** (E - 1)) + E0)
    log_subnormal = np.log2(_min_pos_subnormal)
    log_normal = np.log2(_min_pos_normal)
    log_overflow = np.log2(_max)

    x_normal = 2.0 ** (x * (log_overflow - log_normal) + log_normal) * sign
    x_subnormal = 2.0 ** (x * (log_normal - log_subnormal) + log_subnormal - 1) * sign
    x_overflow = 2.0 ** (x * 2 + log_overflow + 1) * sign
    xxx = (x_normal, x_subnormal, x_overflow)
    return xxx, M, E, E0


@pytest.mark.parametrize('round_mode', ['TRN', 'TRN_ZERO', 'RND', 'RND_CONV', 'RND_ZERO', 'RND_INF', 'RND_MIN_INF'])
@pytest.mark.parametrize('overflow_mode', ['WRAP', 'SAT', 'SAT_SYM', 'SAT_ZERO'])
def test_fixed_quantizer_grad(round_mode, overflow_mode, x_kif, fw: FWDict):
    if fw['grad'] is None:
        pytest.skip('No gradient support')

    quantizer = FixedPointQuantizer(round_mode, overflow_mode)
    assert callable(quantizer)

    x, k, i, f = x_kif

    xq = quantizer(x, k, i, f)  # type: ignore

    def abs_quantization_err(x, k, i, f):
        xq = quantizer(x, k, i, f)  # type: ignore
        return fw['abs'](x - xq)

    dx, di, df = fw['grad'](abs_quantization_err, (0, 2, 3))(x, 1, i, f)
    dx, di, df = map(fw['to_np'], (dx, di, df))

    if overflow_mode in ('WRAP', 'SAT_ZERO'):
        xr = quantizer.round(x, f)  # wrapping and zeroing act on the rounded values
        overflow = (xr < -(2**i)) | (xr > 2**i - 2**-f)
        dqx = fw['to_np'](fw['grad'](lambda x, i, f: quantizer(x, 1, i, f))(x, i, f))  # type: ignore
        assert np.all((dqx == 1) ^ overflow), f'X grad Error in {overflow_mode} op'
        assert np.all(dx[~overflow] == 0), 'X grad Error'
        assert np.all(di == 0), 'I grad Error'
        assert np.all((df < 0) | overflow | (x == quantizer(x, 1, i, f))), 'F grad Error'
    elif overflow_mode in ('SAT', 'SAT_SYM'):
        for _dx in np.unique(dx):
            assert _dx in (-1, 0, 1), 'X grad Error'
        assert np.all(df <= 0), 'F grad Error'
        assert np.all(di <= 0), 'I grad Error'
        assert np.all((df < 0) | (di < 0) | (x == xq) | ((k == 0) & (x < 0))), 'Grad Error for sat mode'


@pytest.mark.parametrize('round_mode', ['TRN', 'TRN_ZERO', 'RND', 'RND_CONV', 'RND_ZERO', 'RND_INF', 'RND_MIN_INF'])
@pytest.mark.parametrize('overflow_mode', ['WRAP', 'SAT', 'SAT_SYM', 'SAT_ZERO'])
@pytest.mark.parametrize('null_xgrad_on_zero_width', [False, True])
def test_fixed_quantizer_no_bits(round_mode, overflow_mode, x_kif, null_xgrad_on_zero_width, fw: FWDict):
    """k + i + f <= 0: qx=0, with the gradients left as they are"""
    quantizer = FixedPointQuantizer(round_mode, overflow_mode, null_xgrad_on_zero_width)
    rng = np.random.default_rng(0)
    x = np.concatenate([x_kif[0], np.zeros(64, np.float32)])  # exact 0: where clip ties would pass a gradient
    k = (rng.uniform(size=len(x)) > 0.5).astype(np.float32)
    f = rng.integers(-4, 8, len(x)).astype(np.float32)
    i = -f - k - rng.integers(0, 3, len(x)).astype(np.float32)
    i_clamped = np.maximum(i, -f)
    x, k, i, f, i_clamped = map(fw['make'], (x, k, i, f, i_clamped))

    np.testing.assert_array_equal(fw['to_np'](quantizer(x, k, i, f)), 0)
    if fw['grad'] is None:
        return
    dx, di, df = fw['grad'](lambda x, i, f: quantizer(x, k, i, f), (0, 1, 2))(x, i, f)  # type: ignore
    dx, di, df = map(fw['to_np'], (dx, di, df))
    if not null_xgrad_on_zero_width:
        dx_ref = fw['to_np'](fw['grad'](lambda x: quantizer.forward(x, k, i_clamped, f))(x))  # type: ignore
    else:
        dx_ref = np.zeros_like(dx)
    np.testing.assert_array_equal(dx, dx_ref)
    assert np.any(di) or np.any(df), 'no gradient on i or f'


def test_float_quantizer_grad(xxx_mee0, fw: FWDict):
    if fw['grad'] is None:
        pytest.skip('No gradient support')
    xxx, M, E, E0 = xxx_mee0
    x_normal, x_subnormal, x_overflow = xxx

    def quantization_err(x, M, E, E0):
        xq = float_quantize(x, M, E, E0)
        return fw['abs'](x - xq)

    dx, dm, de, de0 = fw['grad'](quantization_err, range(4))(x_normal, M, E, E0)
    dx, dm, de, de0 = map(fw['to_np'], (dx, dm, de, de0))
    xq = float_quantize(x_normal, M, E, E0)
    mask = x_normal != xq
    assert np.all(dx == 0), 'Normal Number X grad Error'
    assert np.all(dm[mask] < 0), f'Normal Number M grad Error: max={np.max(dm[mask])}>0'
    assert np.all(de == 0), 'Normal Number E grad Error'
    assert np.all(de0 == 0), 'Normal Number E0 grad Error'

    dx, dm, de, de0 = fw['grad'](quantization_err, range(4))(x_subnormal, M, E, E0)
    dx, dm, de, de0 = map(fw['to_np'], (dx, dm, de, de0))
    xq = float_quantize(x_subnormal, M, E, E0)
    mask = x_subnormal != xq
    assert np.all(dx == 0), 'Subnormal Number X grad Error'
    assert np.all(dm[mask] < 0), f'Subnormal Number M grad Error: max={np.max(dm[mask])}>0'
    assert np.all(de[mask] < 0), f'Subnormal Number E grad Error: max={np.max(de[mask])}>0'
    assert np.all(de0[mask] > 0), f'Subnormal Number E0 grad Error: max={np.max(de0[mask])}>0'

    dx, dm, de, de0 = fw['grad'](quantization_err, range(4))(x_overflow, M, E, E0)
    dx, dm, de, de0 = map(fw['to_np'], (dx, dm, de, de0))
    assert np.all(dx == 0), 'Overflow Number X grad Error'
    assert np.all(dm == 0), 'Overflow Number M grad Error'
    assert np.all(de < 0), f'Overflow Number E grad Error: max={np.max(de)}>0'
    assert np.all(de0 < 0), f'Overflow Number E0 grad Error: max={np.max(de0)}>0'


def float_decompose_np(*args):
    return float_decompose(*(np.asarray(a) for a in args))


def test_float_decompose(xxx_mee0):
    xxx, M, E, E0 = xxx_mee0
    x_normal, x_subnormal, x_overflow = xxx

    mm, ee = float_decompose(x_normal, M, E, E0)
    mm_, ee_ = float_decompose_np(x_normal, M, E, E0)
    assert np.all(mm == mm_) and np.all(ee == ee_), 'Float Decompose Error @ Normal Number'

    xq_ = mm * 2.0**ee  # type: ignore
    xq = float_quantize(x_normal, M, E, E0)
    assert np.all(xq == xq_), 'Float Decompose Error @ Normal Number'
    assert np.all(np.abs(mm) < 2), 'Mantissa Error @ Normal Number'  # type: ignore
    assert np.all(np.abs(mm) >= 1), 'Mantissa Error @ Normal Number'  # type: ignore

    mm, ee = float_decompose(x_subnormal, M, E, E0)
    mm_, ee_ = float_decompose_np(x_subnormal, M, E, E0)
    assert np.all(mm == mm_) and np.all(ee == ee_), 'Float Decompose Error @ Subnormal Number'

    xq_ = mm * 2.0**ee  # type: ignore
    xq = float_quantize(x_subnormal, M, E, E0)
    assert np.all(xq == xq_), 'Float Decompose Error @ Subnormal Number'
    assert np.all(np.abs(mm) < 1), 'Mantissa Error @ Subnormal Number'  # type: ignore

    mm, ee = float_decompose(x_overflow, M, E, E0)
    mm_, ee_ = float_decompose_np(x_overflow, M, E, E0)
    assert np.all(mm == mm_) and np.all(ee == ee_), 'Float Decompose Error'

    xq_ = mm * 2.0**ee  # type: ignore
    xq = float_quantize(x_overflow, M, E, E0)
    assert np.all(xq == xq_), 'Float Decompose Error'
    assert np.all(np.abs(mm) < 2), 'Mantissa Error @ Overflow Number'  # type: ignore
    assert np.all(np.abs(mm) >= 1), 'Mantissa Error @ Overflow Number'  # type: ignore


def test_binary_quantizer_grad(x_kif, fw: FWDict):
    if fw['grad'] is None:
        pytest.skip('No gradient support')
    x, *_ = x_kif
    xq = binary_quantize(x)
    assert np.all((xq == -1) | (xq == 1)), 'Binary Quantizer Error'
    grad = fw['grad'](lambda x: binary_quantize(x))(x)
    grad = fw['to_np'](grad)
    assert np.all(grad > 0), 'Gradient Error'


def test_ternary_quantizer_grad(x_kif, fw: FWDict):
    if fw['grad'] is None:
        pytest.skip('No gradient support')
    x, *_ = x_kif
    xq = ternary_quantize(x)
    assert np.all((xq == -1) | (xq == 0) | (xq == 1)), 'Ternary Quantizer Error'
    grad = fw['grad'](lambda x: ternary_quantize(x))(x)
    grad = fw['to_np'](grad)
    assert np.all(grad > 0), 'Gradient Error'
