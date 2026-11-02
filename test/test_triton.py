import os
import subprocess
import sys

import numpy as np
import pytest

from quantizers import FixedPointQuantizer, binary_quantize, float_decompose, float_quantize, ternary_quantize
from quantizers.fixed_point.fixed_point_ops import round_mode_registry, saturation_mode_registry

try:
    import jax
    import triton
except ImportError:
    pytest.skip('triton and jax are required for this test', allow_module_level=True)

import jax.numpy as jnp
import triton.language as tl
from triton.runtime.interpreter import TensorHandle

from quantizers import triton as qt

if not triton.knobs.runtime.interpret:
    pytest.skip('kernels on numpy arrays: needs TRITON_INTERPRET=1', allow_module_level=True)
pytestmark = pytest.mark.filterwarnings('ignore:invalid value encountered:RuntimeWarning')

N = 4096
BLOCK = tl.constexpr(N)


@triton.jit
def load(ptrs):
    vals = ()
    for j in tl.static_range(len(ptrs)):
        vals = vals + (tl.load(ptrs[j] + tl.arange(0, BLOCK)),)
    return vals


@triton.jit
def store(ptrs, vals):
    for j in tl.static_range(len(ptrs)):
        tl.store(ptrs[j] + tl.arange(0, BLOCK), vals[j])


@triton.jit
def fixed_kernel(ins, outs, R: tl.constexpr, O: tl.constexpr, DX0: tl.constexpr):
    x, k, i, f, g = load(ins)
    store(outs, (qt.fixed_quantize(x, k, i, f, R, O, DX0),) + qt.fixed_quantize_vjp(x, k, i, f, g, R, O, DX0))


@triton.jit
def round_kernel(ins, outs, R: tl.constexpr):
    x, f, g = load(ins)
    store(outs, (qt.fixed_round(x, f, R),) + qt.fixed_round_vjp(x, f, g, R))


@triton.jit
def float_kernel(ins, outs):
    x, m, e, e0, g = load(ins)
    store(outs, (qt.float_quantize(x, m, e, e0),) + qt.float_quantize_vjp(x, m, e, e0, g) + qt.float_decompose(x, m, e, e0))


@triton.jit
def sign_kernel(ins, outs):
    x, g = load(ins)
    store(outs, (qt.binary_quantize(x), qt.binary_quantize_vjp(x, g), qt.ternary_quantize(x), qt.ternary_quantize_vjp(x, g)))


@triton.jit
def broadcast_kernel(ins, outs, i):
    # k a literal, i a scalar argument, f a constant, x float16: broadcast to x's block, computed in float32
    x, g = load(ins)
    x = x.to(tl.float16)
    store(outs, (qt.fixed_quantize(x, 1, i, 3.0, 'RND', 'SAT'),) + qt.fixed_quantize_vjp(x, 1, i, 3.0, g, 'RND', 'SAT'))


def pointer(a: np.ndarray):
    assert a.dtype == np.float32 and a.shape == (N,) and a.flags.c_contiguous
    ty = tl.pointer_type(tl.float32)
    return tl.tensor(TensorHandle(np.array([a.ctypes.data], np.uint64), ty), ty)


def scalar(v: float):
    return tl.tensor(TensorHandle(np.array([v], np.float32), tl.float32), tl.float32)


def launch(kernel, ins, n_out, *args, **constexprs):
    outs = [np.empty(N, np.float32) for _ in range(n_out)]
    kernel[(1,)](tuple(map(pointer, ins)), tuple(map(pointer, outs)), *args, **constexprs)
    return outs


def reference(fn, args, g):
    args = [jnp.broadcast_to(jnp.asarray(a, jnp.float32), g.shape) for a in args]
    grads = jax.grad(lambda *a: jnp.sum(fn(*a) * g), argnums=range(len(args)))(*args)
    return [np.asarray(v) for v in (fn(*args), *grads)]


def assert_match(got, want, names, x, g, rtol=1e-6):
    finite = np.isfinite(x)
    xg = np.abs(x[finite] * g[finite])
    for name, a, b in zip(names, got, want, strict=True):
        np.testing.assert_array_equal(np.isnan(a[~finite]), np.isnan(b[~finite]), err_msg=f'{name}, non-finite operands')
        a, b = a[finite], b[finite]
        if not name.startswith('d'):
            np.testing.assert_array_equal(a, b, err_msg=name)
            continue
        bad = ~np.isclose(a, b, rtol=rtol, atol=rtol * xg, equal_nan=True)
        assert not bad.any(), f'{name}: {bad.sum()} mismatches, e.g. {a[bad][:4]} for {b[bad][:4]} at x={x[finite][bad][:4]}'


def fixed_operands(wide: bool):
    """x, k, i, f and cotangent g"""
    rng = np.random.default_rng(0)
    k = rng.integers(0, 2, N)
    if wide:
        i, f = rng.integers(-24, 41, N), rng.integers(-24, 41, N)
        x = rng.choice([-1, 1], N) * (1 + rng.random(N)) * 2.0 ** rng.integers(-20, 41, N)
    else:
        i, f = rng.integers(-4, 9, N), rng.integers(-4, 13, N)
        x = (np.floor((2 * rng.random(N) - 1) * 2.0 ** (np.maximum(i + f, 0) + 1)) + rng.choice([0, 0.5, -0.5], N)) * 2.0**-f
        x = x.astype(np.float32)
        step = (rng.random(N) < 0.3) & (x != 0)  # to float32 neighbour, excl. for 0
        x = np.where(step, np.nextafter(x, rng.choice(np.float32([-np.inf, np.inf]), N)), x)
    x = np.where(rng.random(N) < 0.05, rng.choice([0.0, -0.0, np.inf, -np.inf, np.nan], N), x)
    return [v.astype(np.float32) for v in (x, k, i, f, rng.standard_normal(N))]


@pytest.mark.parametrize('wide', [False, True])
@pytest.mark.parametrize('null_xgrad', [False, True])
@pytest.mark.parametrize('overflow_mode', list(saturation_mode_registry))
@pytest.mark.parametrize('round_mode', list(round_mode_registry))
def test_fixed_quantize(round_mode, overflow_mode, null_xgrad, wide):
    x, k, i, f, g = fixed_operands(wide)
    want = reference(FixedPointQuantizer(round_mode, overflow_mode, null_xgrad), (x, k, i, f), g)
    got = launch(fixed_kernel, (x, k, i, f, g), len(want), R=round_mode, O=overflow_mode, DX0=null_xgrad)
    assert_match(got, want, ('y', 'dx', 'dk', 'di', 'df'), x, g)


@pytest.mark.parametrize('round_mode', list(round_mode_registry))
def test_fixed_round(round_mode):
    x, _, _, f, g = fixed_operands(False)
    want = reference(FixedPointQuantizer(round_mode).round, (x, f), g)
    got = launch(round_kernel, (x, f, g), len(want), R=round_mode)
    assert_match(got, want, ('y', 'dx', 'df'), x, g)


def test_float_quantize():
    rng = np.random.default_rng(1)
    x = rng.choice([-1, 1], N) * (1 + rng.random(N)) * 2.0 ** rng.integers(-20, 20, N)
    x[:80] = np.outer([1, -1], 2.0 ** np.arange(-20, 20)).ravel()  # exact pow2
    m, e, e0 = rng.integers(0, 8, N), rng.integers(1, 6, N), rng.integers(-3, 4, N)
    x, m, e, e0, g = (v.astype(np.float32) for v in (x, m, e, e0, rng.standard_normal(N)))
    want = reference(float_quantize, (x, m, e, e0), g)
    want += [np.asarray(v) for v in float_decompose(*map(jnp.asarray, (x, m, e, e0)))]
    got = launch(float_kernel, (x, m, e, e0, g), len(want))
    assert_match(got, want, ('y', 'dx', 'dm', 'de', 'de0', 'mantissa', 'exponent'), x, g)


def test_binary_ternary_quantize():
    rng = np.random.default_rng(2)
    x = 2 * rng.standard_normal(N).astype(np.float32)
    g = rng.standard_normal(N).astype(np.float32)
    x[:6] = [0.0, -0.0, 0.5, -0.5, 1.0, -1.0]  # at and around the thresholds
    want = reference(binary_quantize, (x,), g) + reference(ternary_quantize, (x,), g)
    got = launch(sign_kernel, (x, g), len(want))
    # grad of tanh
    assert_match(got, want, ('binary', 'dx binary', 'ternary', 'dx ternary'), x, g, rtol=1e-5)


def test_broadcast_operands():
    x, _, _, _, g = fixed_operands(False)
    x16 = x.astype(np.float16).astype(np.float32)
    want = reference(FixedPointQuantizer('RND', 'SAT'), (x16, 1, 2.0, 3.0), g)
    got = launch(broadcast_kernel, (x, g), len(want), scalar(2.0))
    assert_match(got, want, ('y', 'dx', 'dk', 'di', 'df'), x16, g)


def test_source_key_in_cache_key():
    """cache invalidation by any change in SOURCE_KEY"""
    code = (
        'import sys, triton.language as tl, quantizers.triton as qt; '
        'qt.SOURCE_KEY = tl.constexpr(sys.argv[1]) if sys.argv[1] else qt.SOURCE_KEY; '
        'print(qt.fixed_quantize.cache_key, qt.fixed_quantize_vjp.cache_key)'
    )
    env = {k: v for k, v in os.environ.items() if k != 'TRITON_INTERPRET'}  # jit functions

    def keys(key):
        return subprocess.run([sys.executable, '-c', code, key], env=env, capture_output=True, text=True, check=True).stdout

    assert keys('') == keys('') != keys('another source')
