from typing import Any, TypeVar, overload

import numpy as np

from .binary import binary_quantize, ternary_quantize
from .fixed_point import FixedPointQuantizer
from .minifloat import float_quantize

T = TypeVar('T', np.ndarray, float, int)


class BinaryQ:
    """Stateful binary quantizer."""

    @overload
    def __call__(self, x: T) -> np.ndarray: ...

    @overload
    def __call__(self, x: Any) -> Any: ...

    def __call__(self, x):
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
        The rounding method to use, by default 'TRN'. Available options are 'TRN', 'TRN_ZERO', 'RND', 'RND_CONV', 'RND_ZERO', 'RND_INF', 'RND_MIN_INF'.
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
        k, i, f = self.signed, self.integer - self.signed, self.width - self.integer
        return FixedPointQuantizer(self.rounding, self.overflow)(x, k, i, f)

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
        return float_quantize(x, s, e, e0)

    def __repr__(self):
        return f'Minifloat(s={self.significant}, e={self.exponent}, b={self.exponent_bias})'
