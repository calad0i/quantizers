from typing import Any, TypeVar, overload

import numpy as np

from .binary import *
from .fixed_point import *
from .minifloat import *

T = TypeVar('T', np.ndarray, float, int)


def is_tensor(x):
    base_module = x.__class__.__module__.split('.', 1)[0]
    return base_module in ('tensorflow', 'torch', 'jaxlib')


class BinaryQ:
    """Stateful binary quantizer."""

    @overload
    def __call__(self, x: T) -> np.ndarray: ...

    @overload
    def __call__(self, x: Any) -> Any: ...

    def __call__(self, x):
        use_numpy = not is_tensor(x)
        if use_numpy:
            return binary_quantize_np(x)
        else:
            return binary_quantize(x)

    def __repr__(self):
        return 'Binary()'


class TernaryQ:
    """Stateful ternary quantizer."""

    @overload
    def __call__(self, x: T) -> np.ndarray: ...

    @overload
    def __call__(self, x: Any) -> Any: ...

    def __call__(self, x):
        use_numpy = not is_tensor(x)
        if use_numpy:
            return ternary_quantize_np(x)
        else:
            return ternary_quantize(x)

    def __repr__(self):
        return 'Ternary()'


class FixedQ:
    """Stateful fixed-point quantizer.

    Parameters
    ----------
    width : int
        The total bit-width of the quantized number.
    integer : int
        The number of bits used for the integer part. INCLUDES the sign bit.
    signed : bool|int
        Whether the quantized number is signed.
    rounding : str, optional
        The rounding method to use, by default 'TRN'. Available options are 'TRN', 'TRN_ZERO', 'RND', 'S_RND', 'S_RND_CONV', 'RND_CONV', 'RND_ZERO', 'RND_INF', 'RND_MIN_INF'.
    overflow : str, optional
        The overflow handling method to use, by default 'WRAP'. Available options are 'WRAP', 'SAT', 'SAT_SYM', 'WRAP_SM'.
    """

    def __init__(self, width: int, integer: int, signed: bool | int = True, rounding: str = 'TRN', overflow: str = 'WRAP'):
        self.width = width
        self.integer = integer
        self.signed = signed
        self.rounding = rounding
        self.overflow = overflow

    @overload
    def __call__(self, x: T) -> np.ndarray: ...

    @overload
    def __call__(self, x: Any) -> Any: ...

    def __call__(self, x):
        use_numpy = not is_tensor(x)
        k, i, f = self.signed, self.integer - self.signed, self.width - self.integer
        k, i, f = float(k), float(i), float(f)
        if use_numpy:
            q = get_fixed_quantizer_np(self.rounding, self.overflow)
        else:
            q = get_fixed_quantizer(self.rounding, self.overflow)
        return q(x, k, i, f, False, None)  # type: ignore

    def __repr__(self):
        return f'Fixed(w={self.width}, i={self.integer}, s={self.signed}, r={self.rounding}, o={self.overflow})'


class MinifloatQ:
    """Stateful minifloat quantizer.

    Parameters
    ----------
    significant : int
        The number of significant bits.
    exponent : int
        The number of exponent bits.
    exponent_bias : int
        The bias applied to the exponent.
    """

    def __init__(self, significant: int, exponent: int, exponent_bias: int):
        self.significant = significant
        self.exponent = exponent
        self.exponent_bias = exponent_bias

    @overload
    def __call__(self, x: T) -> np.ndarray: ...

    @overload
    def __call__(self, x: Any) -> Any: ...

    def __call__(self, x):
        s, e, e0 = float(self.significant), float(self.exponent), float(self.exponent_bias)
        use_numpy = not is_tensor(x)
        if use_numpy:
            return float_quantize_np(x, s, e, e0)
        else:
            return float_quantize(x, s, e, e0)

    def __repr__(self):
        return f'Minifloat(s={self.significant}, e={self.exponent}, b={self.exponent_bias})'
