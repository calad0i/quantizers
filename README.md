# Quantizers
[![PyPI version](https://badge.fury.io/py/quantizers.svg)](https://badge.fury.io/py/quantizers)
[![License](https://img.shields.io/badge/License-LGPL-blue)](LICENSE)
[![Tests](https://github.com/calad0i/quantizers/actions/workflows/python-test.yml/badge.svg)](https://github.com/calad0i/quantizers/actions/workflows/python-test.yml)
[![Coverage](https://img.shields.io/codecov/c/github/calad0i/quantizers)](https://app.codecov.io/gh/calad0i/quantizers)


Hardware-oriented numerical quantizers for deep learning models: same quantizers for `numpy`, `jax`, `torch`, `tensorflow`, `mlx`, `tinygrad`, plus keras 3 variables ones and triton device functions for custom kernels. Bit-exact with the fixed-point and floating-point types of Siemens HLS libraries (`ac_types`), and differentiable through each framework's autodiff when available.

## Features

- Bit-accurate fixed-point and minifloat quantization, tested against [ac_types](https://github.com/hlslibs/ac_types)' `ac_fixed` and `ac_std_float`
- `einops` like multi framework dispatched based on the tensor type and lazy import
- Differentiable, through each framework's own autodiff (and XLA/JIT compilation like `torch.compile`, `jax.jit`, `tf.function`)
  - Surrogate gradients on the bit widths as described in [HGQ [FPGA'26]](https://doi.org/10.1145/3748173.3779200)
- Triton device functions of every quantizer with vjps for use in custom kernels

## Supported Quantizers

### Fixed-Point Quantizer

Parameters:
- `k` (keep_negative): Enable negative numbers
- `i` (integer_bits): Number of bits before decimal point (excludes sign bit)
- `f` (fractional_bits): Number of bits after decimal point
- For C++: `W = k + i + f`, `I = k + i`, `S = k`

Supported modes:
- Rounding: `TRN`, `RND`, `RND_CONV`, `TRN_ZERO`, `RND_ZERO`, `RND_MIN_INF`, `RND_INF`
- Overflow: `WRAP`, `SAT`, `SAT_SYM`, `SAT_ZERO`

Gradients:
- `x`: passed through, except where the value saturates, wraps or is zeroed
- `i`, `f`: surrogate gradients; `WRAP` and `SAT_ZERO` give none on `i`
- Zero width (`k + i + f <= 0`): the value is 0, the gradients are left as they are; `null_xgrad_on_zero_width=True` drops the gradient on `x` there

Differences from `ap_fixed`: a signed 1-bit `SAT_SYM` gives 0, the only symmetric value (`ap_fixed` keeps `{-1, 0}`), and `WRAP_SM` is not supported.

### Minifloat Quantizer

Parameters:
- `m` (mantissa_bits): Mantissa width, excluding the implicit leading bit
- `e` (exponent_bits): Exponent width
- `e0` (exponent_zero): Exponent bias, default 1 (IEEE's bias)
- Normal exponents: `[e0 + 1 - 2^(e-1), e0 - 1 + 2^(e-1)]`

Features:
- Supports subnormal numbers
- No inf or NaN encodings: the top exponent holds numbers, value beyond the maximum saturates
- `RND_CONV` rounding for all

### Simplified Quantizers

- **Binary**: Maps to {-1, 1}, with 0 to 1. Gradient of tanh
- **Ternary**: Maps to {-1, 0, 1}, i.e. fixed-point `fixed<2, 2, RND_CONV, SAT_SYM>`. Gradient of tanh


## Installation

**requires python>=3.10**

```bash
pip install quantizers
```
numpy arrays and python scalars need `numpy`. Otherwise, no dependency is required beyond the particular framework you already have. For keras, install `keras>=3.0.0` is expected.

## Usage

### Stateless Quantizers
```python
from quantizers import FixedPointQuantizer, binary_quantize, fixed_quantize, float_quantize, ternary_quantize

# Fixed-point: x, k, i, f are tensors of one framework (or python scalars), broadcast together
quantizer = FixedPointQuantizer(round_mode='RND', overflow_mode='SAT')
xq = quantizer(x, k, i, f)
xq = fixed_quantize(x, k, i, f, round_mode='RND', overflow_mode='SAT') # shortcut for the above
xq = quantizer.round(x, f)  # rounding to f fractional bits only, without overflow handling

# Minifloat
xq = float_quantize(x, mantissa_bits, exponent_bits, exponent_zero)

# Simplified quantizers
xq = binary_quantize(x)
xq = ternary_quantize(x)
```

### Stateful Quantizers
```python
from quantizers import BinaryQ, FixedQ, MinifloatQ, TernaryQ

# Fixed formats, for inference and testing; for training use the stateless ones
fixed_q = FixedQ(width, integer_bits, signed, rounding='RND', overflow='SAT')  # integer_bits includes the sign bit
xq = fixed_q(x)

mfloat_q = MinifloatQ(mantissa_bits, exponent_bits, exponent_zero)
xq = mfloat_q(x)
```

### Triton Device Functions
```python
import triton
import triton.language as tl
from quantizers import triton as qt

@triton.jit
def kernel(x_ptr, i_ptr, f_ptr, g_ptr, y_ptr, dx_ptr, di_ptr, df_ptr, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x, i, f, g = tl.load(x_ptr + o), tl.load(i_ptr + o), tl.load(f_ptr + o), tl.load(g_ptr + o)
    tl.store(y_ptr + o, qt.fixed_quantize(x, 1, i, f, 'RND', 'SAT'))
    dx, dk, di, df = qt.fixed_quantize_vjp(x, 1, i, f, g, 'RND', 'SAT')
    ...
```
- `fixed_quantize`, `fixed_round`, `float_quantize`, `binary_quantize`, `ternary_quantize`, each with a `_vjp`, and `float_decompose`
- Elementwise in float32 over the operands broadcast together (tensors of any dtype, or scalars); modes are constexpr strings
- A `_vjp` takes the cotangent `g` of the value and returns the cotangent of every operand, of the broadcast shape: reduce those of broadcast operands in the kernel
- Values match the JAX backend bit for bit on finite floats of the normal range, gradients to float32 rounding
- Kernels using them recompile when quantizers changes; with `TRITON_INTERPRET=1` they run in Triton's CPU interpreter

## Development

```bash
pip install -e '.[test]'
pytest -n auto
```
The tests compare against a C++ reference built with `make` on the first run (a C++ compiler is needed) and run on the CPU only, Triton kernels in its interpreter.
