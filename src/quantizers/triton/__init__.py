"""Triton device functions of the quantizers, for kernels.

    import triton
    import triton.language as tl
    from quantizers import triton as qt

    @triton.jit
    def kernel(x_ptr, ...):
        ...
        y = qt.fixed_quantize(x, k, i, f, 'RND', 'SAT')
        dx, dk, di, df = qt.fixed_quantize_vjp(x, k, i, f, g, 'RND', 'SAT')

Each is the quantizer itself, run on Triton values while the kernel compiles (the `triton` backend): elementwise in
float32 over its operands broadcast together (tensors of any dtype, or scalars), with the modes as constexpr strings. A
`_vjp` takes the cotangent `g` of the value and returns the cotangent of every operand, as JAX's vjp, of the broadcast
shape: reduce those of broadcast operands in the kernel.

Values are the JAX backend's, bit for bit on finite floats of the normal range; gradients up to float32 rounding (FMA
contraction, libdevice's tanh in the binary and ternary gradients).

Kernels using these recompile when the quantizers sources change: `SOURCE_KEY` enters their Triton cache key. Under
TRITON_INTERPRET=1 they run in Triton's CPU interpreter, with numpy in place of libdevice."""

import hashlib
import pathlib

import triton
import triton.language as tl

from ..binary import binary_quantize as _binary_quantize
from ..binary import ternary_quantize as _ternary_quantize
from ..fixed_point import FixedPointQuantizer
from ..minifloat import float_decompose as _float_decompose
from ..minifloat import float_quantize as _float_quantize
from . import _lowering as _ops

__all__ = [
    'SOURCE_KEY',
    'binary_quantize',
    'binary_quantize_vjp',
    'fixed_quantize',
    'fixed_quantize_vjp',
    'fixed_round',
    'fixed_round_vjp',
    'float_decompose',
    'float_quantize',
    'float_quantize_vjp',
    'ternary_quantize',
    'ternary_quantize_vjp',
]


def _source_key() -> str:
    root = pathlib.Path(__file__).parent.parent  # the quantizers package
    h = hashlib.sha256()
    for path in sorted(root.rglob('*.py')):
        h.update(path.relative_to(root).as_posix().encode() + b'\0' + path.read_bytes())
    return h.hexdigest()


SOURCE_KEY = tl.constexpr(_source_key())
_c = _ops.constexpr_value


def _fixed(round_mode, overflow_mode, null_xgrad_on_zero_width):
    return FixedPointQuantizer(_c(round_mode), _c(overflow_mode), bool(_c(null_xgrad_on_zero_width)))


def _fixed_round(round_mode):
    q = FixedPointQuantizer(_c(round_mode))
    return lambda x, f: q.round(x, f)


# Builtins: the lowerings, called by the jit functions below (which carry SOURCE_KEY)


@_ops.builtin
def _fixed_quantize(x, k, i, f, round_mode, overflow_mode, null_xgrad_on_zero_width, key, _semantic=None):
    return _ops.value(_semantic, _fixed(round_mode, overflow_mode, null_xgrad_on_zero_width), x, k, i, f)


@_ops.builtin
def _fixed_quantize_vjp(x, k, i, f, g, round_mode, overflow_mode, null_xgrad_on_zero_width, key, _semantic=None):
    return _ops.vjp(_semantic, _fixed(round_mode, overflow_mode, null_xgrad_on_zero_width), (x, k, i, f), g)


@_ops.builtin
def _fixed_round_value(x, f, round_mode, key, _semantic=None):
    return _ops.value(_semantic, _fixed_round(round_mode), x, f)


@_ops.builtin
def _fixed_round_vjp(x, f, g, round_mode, key, _semantic=None):
    return _ops.vjp(_semantic, _fixed_round(round_mode), (x, f), g)


@_ops.builtin
def _float_quantize_value(x, m, e, e0, key, _semantic=None):
    return _ops.value(_semantic, _float_quantize, x, m, e, e0)


@_ops.builtin
def _float_quantize_vjp(x, m, e, e0, g, key, _semantic=None):
    return _ops.vjp(_semantic, _float_quantize, (x, m, e, e0), g)


@_ops.builtin
def _float_decompose_value(x, m, e, e0, key, _semantic=None):
    return _ops.value(_semantic, _float_decompose, x, m, e, e0)


@_ops.builtin
def _binary_quantize_value(x, key, _semantic=None):
    return _ops.value(_semantic, _binary_quantize, x)


@_ops.builtin
def _binary_quantize_vjp(x, g, key, _semantic=None):
    return _ops.vjp(_semantic, _binary_quantize, (x,), g)[0]


@_ops.builtin
def _ternary_quantize_value(x, key, _semantic=None):
    return _ops.value(_semantic, _ternary_quantize, x)


@_ops.builtin
def _ternary_quantize_vjp(x, g, key, _semantic=None):
    return _ops.vjp(_semantic, _ternary_quantize, (x,), g)[0]


# Device functions


@triton.jit
def fixed_quantize(
    x,
    k,
    i,
    f,
    round_mode: tl.constexpr = 'TRN',
    overflow_mode: tl.constexpr = 'WRAP',
    null_xgrad_on_zero_width: tl.constexpr = False,
):
    """`FixedPointQuantizer(round_mode, overflow_mode, null_xgrad_on_zero_width)(x, k, i, f)`."""
    return _fixed_quantize(x, k, i, f, round_mode, overflow_mode, null_xgrad_on_zero_width, SOURCE_KEY)


@triton.jit
def fixed_quantize_vjp(
    x,
    k,
    i,
    f,
    g,
    round_mode: tl.constexpr = 'TRN',
    overflow_mode: tl.constexpr = 'WRAP',
    null_xgrad_on_zero_width: tl.constexpr = False,
):
    """(dx, dk, di, df) of `fixed_quantize` for the cotangent `g`."""
    return _fixed_quantize_vjp(x, k, i, f, g, round_mode, overflow_mode, null_xgrad_on_zero_width, SOURCE_KEY)


@triton.jit
def fixed_round(x, f, round_mode: tl.constexpr = 'TRN'):
    """`FixedPointQuantizer(round_mode).round(x, f)`: rounding to f fractional bits, without saturation."""
    return _fixed_round_value(x, f, round_mode, SOURCE_KEY)


@triton.jit
def fixed_round_vjp(x, f, g, round_mode: tl.constexpr = 'TRN'):
    """(dx, df) of `fixed_round` for the cotangent `g`."""
    return _fixed_round_vjp(x, f, g, round_mode, SOURCE_KEY)


@triton.jit
def float_quantize(x, m, e, e0=1.0):
    """`float_quantize(x, m, e, e0)`."""
    return _float_quantize_value(x, m, e, e0, SOURCE_KEY)


@triton.jit
def float_quantize_vjp(x, m, e, e0, g):
    """(dx, dm, de, de0) of `float_quantize` for the cotangent `g`."""
    return _float_quantize_vjp(x, m, e, e0, g, SOURCE_KEY)


@triton.jit
def float_decompose(x, m, e, e0=1.0):
    """(mantissa, exponent) of `float_decompose(x, m, e, e0)`."""
    return _float_decompose_value(x, m, e, e0, SOURCE_KEY)


@triton.jit
def binary_quantize(x):
    """`binary_quantize(x)`."""
    return _binary_quantize_value(x, SOURCE_KEY)


@triton.jit
def binary_quantize_vjp(x, g):
    """dx of `binary_quantize` for the cotangent `g`."""
    return _binary_quantize_vjp(x, g, SOURCE_KEY)


@triton.jit
def ternary_quantize(x):
    """`ternary_quantize(x)`."""
    return _ternary_quantize_value(x, SOURCE_KEY)


@triton.jit
def ternary_quantize_vjp(x, g):
    """dx of `ternary_quantize` for the cotangent `g`."""
    return _ternary_quantize_vjp(x, g, SOURCE_KEY)
