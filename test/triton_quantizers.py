"""The quantizers through their Triton device functions, as array functions for the shared tests (`fw['q']`).

jax arrays in and out, differentiable: each is a jax.custom_vjp whose forward and backward run the device function and its
`_vjp` on the numpy buffers of a host callback, in Triton's CPU interpreter."""

import jax
import jax.numpy as jnp
import numpy as np
import triton
import triton.language as tl
from triton.runtime.interpreter import TensorHandle

import quantizers
from quantizers import triton as qt


@triton.jit
def kernel(ins, outs, FN: tl.constexpr, MODES: tl.constexpr, BLOCK: tl.constexpr):
    # FN(*ins, *MODES), a device function, on one block
    o = tl.arange(0, BLOCK)
    vals = ()
    for j in tl.static_range(len(ins)):
        vals = vals + (tl.load(ins[j] + o),)
    res = FN(*vals, *MODES)
    if len(outs) == 1:
        tl.store(outs[0] + o, res)
    else:
        for j in tl.static_range(len(outs)):
            tl.store(outs[j] + o, res[j])


def pointer(a: np.ndarray):
    """An interpreter pointer to the float32 buffer of `a`"""
    assert a.dtype == np.float32 and a.flags.c_contiguous
    ty = tl.pointer_type(tl.float32)
    return tl.tensor(TensorHandle(np.array([a.ctypes.data], np.uint64), ty), ty)


def launch(fn, args, n_out: int, modes=()):
    """`fn(*args, *modes)`, a device function, on numpy arrays broadcast together: `n_out` numpy arrays"""
    arrays = np.broadcast_arrays(*(np.asarray(a, np.float32) for a in args))
    shape, n = arrays[0].shape, arrays[0].size
    block = 1 << max(n - 1, 0).bit_length()  # one program
    ins = [np.ascontiguousarray(np.pad(a.ravel(), (0, block - n))) for a in arrays]
    outs = [np.empty(block, np.float32) for _ in range(n_out)]
    kernel[(1,)](tuple(map(pointer, ins)), tuple(map(pointer, outs)), fn, tuple(modes), block)
    return tuple(o[:n].reshape(shape) for o in outs)


def _sum_to(g, shape):
    """`g` summed over the dimensions it was broadcast along from `shape`"""
    g = jnp.sum(g, axis=tuple(range(g.ndim - len(shape))))
    axes = tuple(d for d, s in enumerate(shape) if s == 1 and g.shape[d] != 1)
    return jnp.sum(g, axis=axes, keepdims=True) if axes else g


def differentiable(fn, fn_vjp, n_args: int, modes=()):
    """The device function `fn` as a jax function of `n_args` arrays, with the device function `fn_vjp` as its vjp"""

    def call(device_fn, n_out, *args):
        out = jax.ShapeDtypeStruct(jnp.broadcast_shapes(*(jnp.shape(a) for a in args)), jnp.float32)
        return jax.pure_callback(lambda *a: launch(device_fn, a, n_out, modes), (out,) * n_out, *args)

    @jax.custom_vjp
    def f(*args):
        return call(fn, 1, *args)[0]

    def fwd(*args):
        return f(*args), args

    def bwd(args, g):
        return tuple(_sum_to(d, jnp.shape(a)) for d, a in zip(call(fn_vjp, n_args, *args, g), args))

    f.defvjp(fwd, bwd)
    return lambda *args: f(*(jnp.asarray(a, jnp.float32) for a in args))


class FixedPointQuantizer:
    def __init__(self, round_mode: str = 'TRN', overflow_mode: str = 'WRAP', null_xgrad_on_zero_width: bool = False):
        self._call = differentiable(
            qt.fixed_quantize, qt.fixed_quantize_vjp, 4, (round_mode, overflow_mode, null_xgrad_on_zero_width)
        )
        self.round = differentiable(qt.fixed_round, qt.fixed_round_vjp, 2, (round_mode,))

    def __call__(self, x, k, i, f):
        return self._call(x, k, i, f)


def fixed_quantize(x, k, i, f, round_mode: str = 'TRN', overflow_mode: str = 'WRAP'):
    return FixedPointQuantizer(round_mode, overflow_mode)(x, k, i, f)


float_quantize = differentiable(qt.float_quantize, qt.float_quantize_vjp, 4)
binary_quantize = differentiable(qt.binary_quantize, qt.binary_quantize_vjp, 1)
ternary_quantize = differentiable(qt.ternary_quantize, qt.ternary_quantize_vjp, 1)


def float_decompose(x, m, e, e0=1.0):
    return launch(qt.float_decompose, (x, m, e, e0), 2)


class FixedQ(quantizers.FixedQ):
    def __call__(self, x):
        k, i, f = self.signed, self.integer - self.signed, self.width - self.integer
        return fixed_quantize(x, k, i, f, self.rounding, self.overflow)


class MinifloatQ(quantizers.MinifloatQ):
    def __call__(self, x):
        return float_quantize(x, self.significant, self.exponent, self.exponent_bias)


class BinaryQ(quantizers.BinaryQ):
    def __call__(self, x):
        return binary_quantize(x)


class TernaryQ(quantizers.TernaryQ):
    def __call__(self, x):
        return ternary_quantize(x)
