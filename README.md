# Quantizers
[![PyPI version](https://badge.fury.io/py/quantizers.svg)](https://badge.fury.io/py/quantizers)
[![License](https://img.shields.io/badge/License-LGPL-blue)](LICENSE)
[![Tests](https://github.com/calad0i/quantizers/actions/workflows/python-test.yml/badge.svg)](https://github.com/calad0i/quantizers/actions/workflows/python-test.yml)
[![Coverage](https://img.shields.io/codecov/c/github/calad0i/quantizers)](https://app.codecov.io/gh/calad0i/quantizers)


Hardware-oriented numerical quantizers for deep learning models, implemented in Keras v3 and NumPy. Provides bit-accurate precision matching with Vivado/Vitis HLS implementations.

## Features

- Bit-accurate to the HLS implementation up to 32/64-bit floating point precision
- Support for fixed-point and minifloat number formats
- Differentiable Keras v3 implementations with gradients on inputs
  - With surrogate gradients for bit-width optimization as described in *[Gradient-based Automatic Mixed Precision Quantization for Neural Networks On-Chip](https://arxiv.org/abs/2405.00645)*
- Supports stochastic rounding for training

## Supported Quantizers

### Fixed-Point Quantizer

Parameters:
- `k` (keep_negative): Enable negative numbers
- `i` (integer_bits): Number of bits before decimal point (excludes sign bit)
- `f` (fractional_bits): Number of bits after decimal point
- For C++: `W = k + i + f`, `I = k + i`, `S = k`

Supported modes:
- Rounding: `TRN`, `RND`, `RND_CONV`, `TRN_ZERO`, `RND_ZERO`, `RND_MIN_INF`, `RND_INF`
  - `S_RND` and `S_RND_CONV` for stochastic rounding; Not available in NumPy implementation as it is for training only
- Overflow: `WRAP`, `SAT`, `SAT_SYM`, `WRAP_SM`

Limitations:
- `WRAP_SM` only works with `RND` or `RND_CONV` rounding
- `WRAP*` modes don't provide surrogate gradients for integer bits
- Saturation bit forced to zero for `WRAP` and `WRAP_SM`

### Minifloat Quantizer

Parameters:
- `m` (mantissa_bits): Mantissa width
- `e` (exponent_bits): Exponent width
- `e0` (exponent_zero): Exponent bias (default: 0)
- Range: `[-2^(e-1) + e0, 2^(e-1) - 1 + e0]`

Features:
- Supports subnormal numbers
- Uses `RND_CONV` rounding and `SAT` overflow
- HLS-synthesizable implementation in `test/cpp_source/ap_types/ap_float.h`

### Simplified Quantizers

- **Binary**: Maps to {-1,1} with 0 to -1. (preliminary implementation)
- **Ternary**: Shorthand for fixed-point `fixed<2, 1, RND_CONV, SAT_SYM>`


## Installation

**requires python>=3.10**

```bash
pip install quantizers
```
`keras>=3.0` and at least one compatible backend (`pytorch`, `jax`, or `tensorflow`) is required for training.

## Usage

### Stateless Quantizers
```python
from quantizers import (
  float_quantize(_np), # add _np for NumPy implementation
  get_fixed_quantizer(_np),
  binary_quantize(_np),
  ternary_quantize(_np),
)

# Fixed-point quantizer
fixed_quantizer = get_fixed_quantizer(round_mode, overflow_mode)
fixedp_qtensor = fixed_quantizer(
    x,
    integer_bits,
    fractional_bits,
    keep_negative,
    training, # For stochastic rounding, and WRAP does not happen during training
    seed, # For stochastic rounding only
)

# Minifloat quantizer
floatp_qtensor = float_quantize(x, mantissa_bits, exponent_bits, exponent_zero)

# Simplified quantizers
binary_qtensor = binary_quantize(x)
ternary_qtensor = ternary_quantize(x)
```

### Stateful Quantizers
```python
# Can be used for, but not intended for training
fixed_q = FixedQ(
    width,
    integer_bits, # including the sign bit)
    keep_negative,
    fixed_round_mode, # No stochastic rounding
    fixed_overflow_mode
)
quantized = fixed_q(x)

mfloat_q = MinifloatQ(mantissa_bits, exponent_bits, exponent_zero)
quantized = mfloat_q(x)
```
