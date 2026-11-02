"""Triton specifics: forms and operands the shared tests do not cover, generic expressions, errors, compilation, caching.

The quantizers themselves run the shared tests as the `triton` framework (conftest, through triton_quantizers)."""

import os
import subprocess
import sys

import numpy as np
import pytest

import quantizers
from quantizers._backends import get_backend
from quantizers.fixed_point.fixed_point_ops import round_mode_registry

try:
    import jax
    import triton
except ImportError:
    pytest.skip('triton and jax are required for this test', allow_module_level=True)

import jax.numpy as jnp
import triton.language as tl
from triton.runtime.interpreter import TensorHandle

from quantizers import triton as qt
from quantizers.triton import _lowering

if not triton.knobs.runtime.interpret:
    pytest.skip('kernels on numpy arrays: needs TRITON_INTERPRET=1', allow_module_level=True)
pytestmark = pytest.mark.filterwarnings('ignore:invalid value encountered:RuntimeWarning')

import triton_quantizers as tq  # noqa: E402  kernels: the interpreter's, as checked above

N = 1024


def operands():
    """x on and around the grid of f (ties and float32 neighbours), f, and a cotangent g"""
    rng = np.random.default_rng(0)
    f = rng.integers(-4, 13, N).astype(np.float32)
    x = ((np.floor(rng.standard_normal(N) * 64) + rng.choice([0, 0.5, -0.5], N)) * 2.0**-f).astype(np.float32)
    x = np.where(rng.random(N) < 0.3, np.nextafter(x, np.float32(np.inf)), x)
    return x, f, rng.standard_normal(N).astype(np.float32)


def assert_value_and_grad(fn, ref, args, g):
    """Values equal; gradients of sum(fn * g) to float32 rounding"""
    np.testing.assert_array_equal(fn(*args), ref(*args))
    argnums = tuple(range(len(args)))
    grads = [jax.grad(lambda *a, fn=fn: jnp.sum(fn(*a) * g), argnums)(*map(jnp.asarray, args)) for fn in (fn, ref)]
    for a, b in zip(*grads):
        np.testing.assert_allclose(a, b, rtol=1e-6)


def elementwise_reference(fn, args, g):
    """Value, and the per-element cotangents of every operand, as a device vjp returns them"""
    args = [jnp.broadcast_to(jnp.asarray(a, jnp.float32), g.shape) for a in args]
    grads = jax.grad(lambda *a: jnp.sum(fn(*a) * g), tuple(range(len(args))))(*args)
    return [np.asarray(v) for v in (fn(*args), *grads)]


@pytest.mark.parametrize('round_mode', list(round_mode_registry))
def test_fixed_round(round_mode):
    x, f, g = operands()
    assert_value_and_grad(tq.FixedPointQuantizer(round_mode).round, quantizers.FixedPointQuantizer(round_mode).round, (x, f), g)


def test_float_decompose():
    rng = np.random.default_rng(1)
    x = (rng.choice([-1, 1], N) * (1 + rng.random(N)) * 2.0 ** rng.integers(-20, 20, N)).astype(np.float32)
    m, e, e0 = (rng.integers(lo, hi, N).astype(np.float32) for lo, hi in ((0, 8), (1, 6), (-3, 4)))
    for a, b in zip(tq.float_decompose(x, m, e, e0), quantizers.float_decompose(*map(jnp.asarray, (x, m, e, e0)))):
        np.testing.assert_array_equal(a, b)


@triton.jit
def broadcast_kernel(x_ptr, g_ptr, out_ptrs, i, BLOCK: tl.constexpr):
    # k a literal, i a scalar argument, f a constant, x float16: broadcast to x's block, computed in float32
    o = tl.arange(0, BLOCK)
    x, g = tl.load(x_ptr + o).to(tl.float16), tl.load(g_ptr + o)
    res = (qt.fixed_quantize(x, 1, i, 3.0, 'RND', 'SAT'),) + qt.fixed_quantize_vjp(x, 1, i, 3.0, g, 'RND', 'SAT')
    for j in tl.static_range(len(out_ptrs)):
        tl.store(out_ptrs[j] + o, res[j])


@triton.jit
def scalar_kernel(x_ptr, g_ptr, out_ptrs):
    # one element per program: every operand a scalar
    o = tl.program_id(0)
    x, g = tl.load(x_ptr + o), tl.load(g_ptr + o)
    res = (qt.fixed_quantize(x, 1.0, 2.0, 3.0, 'RND', 'SAT'),) + qt.fixed_quantize_vjp(x, 1.0, 2.0, 3.0, g, 'RND', 'SAT')
    for j in tl.static_range(len(out_ptrs)):
        tl.store(out_ptrs[j] + o, res[j])


def test_broadcast_operands():
    x, _, g = operands()
    outs = [np.empty(N, np.float32) for _ in range(5)]
    i = tl.tensor(TensorHandle(np.array([2.0], np.float32), tl.float32), tl.float32)  # a float32 scalar, as compiled
    broadcast_kernel[(1,)](tq.pointer(x), tq.pointer(g), tuple(map(tq.pointer, outs)), i, N)
    x16 = x.astype(np.float16).astype(np.float32)
    for a, b in zip(outs, elementwise_reference(quantizers.FixedPointQuantizer('RND', 'SAT'), (x16, 1, 2, 3), g)):
        np.testing.assert_allclose(a, b, rtol=1e-6)


def test_scalar_operands():
    x, _, g = (v[:64] for v in operands())
    outs = [np.empty(64, np.float32) for _ in range(5)]
    scalar_kernel[(64,)](tq.pointer(x), tq.pointer(g), tuple(map(tq.pointer, outs)))
    for a, b in zip(outs, elementwise_reference(quantizers.FixedPointQuantizer('RND', 'SAT'), (x, 1, 2, 3), g)):
        np.testing.assert_allclose(a, b, rtol=1e-6)


def expression(x, y):
    """reflected operators and |, which the quantizers do not use"""
    bk = get_backend(x)
    return bk.where((x > 0) | (y < 0), (1 + x) * (2 * y), 3 / (x * x + 1))


@_lowering.builtin
def _expression(x, y, _semantic=None):
    return _lowering.value(_semantic, expression, x, y)


@_lowering.builtin
def _expression_vjp(x, y, g, _semantic=None):
    return _lowering.vjp(_semantic, expression, (x, y), g)


@_lowering.builtin
def _branch(x, _semantic=None):
    return _lowering.value(_semantic, lambda x: x if x > 0 else -x, x)


def test_expression():
    x, y, g = np.random.default_rng(3).standard_normal((3, N)).astype(np.float32)
    assert_value_and_grad(tq.differentiable(_expression, _expression_vjp, 2), expression, (x, y), g)


def test_no_data_dependent_control_flow():
    with pytest.raises(Exception, match='no truth value'):
        tq.launch(_branch, (np.ones(N, np.float32),), 1)


def test_device_function_outside_kernel(monkeypatch):
    monkeypatch.setenv('TRITON_INTERPRET', '0')
    with pytest.raises(ValueError, match='call it from a @triton.jit function'):
        qt._fixed_quantize(1.0, 1.0, 1.0, 1.0, 'RND', 'SAT', False, qt.SOURCE_KEY)


COMPILED_KERNEL = """
import triton
import triton.language as tl

from quantizers import triton as qt


@triton.jit
def kernel(x_ptr, y_ptr, BLOCK: tl.constexpr):
    o = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + o)
    y = qt.fixed_quantize(x, 1.0, 2.0, 3.0, 'RND', 'WRAP') + qt.float_quantize(x, 3.0, 4.0) + qt.binary_quantize_vjp(x, x)
    tl.store(y_ptr + o, y)
"""


def test_compiled_lowering(tmp_path):
    """libdevice in place of numpy (rint, fmod, log2, tanh), compiled to PTX: no GPU needed"""
    (tmp_path / 'compiled_kernel.py').write_text(COMPILED_KERNEL)
    code = (
        'import sys, triton; sys.path.insert(0, sys.argv[1]); import compiled_kernel; '
        'from triton.backends.compiler import GPUTarget; from triton.compiler import ASTSource; '
        "src = ASTSource(compiled_kernel.kernel, {'x_ptr': '*fp32', 'y_ptr': '*fp32', 'BLOCK': 'constexpr'}, {'BLOCK': 128}); "
        "print(len(triton.compile(src, target=GPUTarget('cuda', 80, 32)).asm['ptx']))"
    )
    env = {k: v for k, v in os.environ.items() if k != 'TRITON_INTERPRET'}  # jit functions
    env['TRITON_CACHE_DIR'] = str(tmp_path / 'cache')  # a lowering, not a cached kernel
    r = subprocess.run([sys.executable, '-c', code, str(tmp_path)], env=env, capture_output=True, text=True)
    assert r.returncode == 0 and int(r.stdout) > 0, r.stderr


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
