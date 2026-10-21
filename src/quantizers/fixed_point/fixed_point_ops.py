from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

from keras import ops
from numpy.typing import ArrayLike

round_mode_registry: dict[str, Callable[[Any], Any]] = {}
saturation_mode_registry: dict[str, Callable[[Any, Any, Any, Any], Any]] = {}

T = TypeVar('T', bound=ArrayLike)


def _clip(x, min_value, max_value):
    mask_overflow = x > max_value
    mask_underflow = x < min_value
    return ops.where(mask_overflow, max_value, ops.where(mask_underflow, min_value, x))


def rnd_mode(name: str):
    def inner(func):
        assert name not in round_mode_registry, f"Round mode '{name}' already exists."

        @wraps(func)
        def wrapper(x):
            xq = func(x)
            return ops.stop_gradient(xq) + (x - ops.stop_gradient(x))

        round_mode_registry[name] = wrapper
        return wrapper

    return inner


@rnd_mode('TRN')
def floor(x):
    return ops.floor(x)


@rnd_mode('RND')
def round(x):
    # Round to nearest, ties positive infinity.
    return ops.floor(x + 0.5)


@rnd_mode('RND_CONV')
def round_conv(x):
    # Round to nearest, ties to even.
    return ops.round(x)


@rnd_mode('TRN_ZERO')
def floor_zero(x):
    # Truncate towards zero.
    sign = ops.sign(x)
    return ops.floor(ops.abs(x)) * sign  # type: ignore


@rnd_mode('RND_ZERO')
def round_zero(x):
    # Round to nearest, ties towards zero.
    sign = ops.sign(x)
    return -ops.floor(-ops.abs(x) + 0.5) * sign  # type:ignore


@rnd_mode('RND_MIN_INF')
def round_min_inf(x):
    # Round to nearest, ties towards negative infinity.
    return -ops.floor(-x + 0.5)  # type:ignore


@rnd_mode('RND_INF')
def round_inf(x):
    # Round to nearest, ties away from zero.
    sign = ops.sign(x)
    return ops.floor(ops.abs(x) + 0.5) * sign  # type: ignore


def sat_mode(name: str | list | tuple):
    names = (name,) if isinstance(name, str) else name

    def inner(func):
        for name in names:
            assert name not in saturation_mode_registry, f"Saturation mode '{name}' already exists."
            saturation_mode_registry[name] = func
        saturation_mode_registry[func.__name__.upper()] = func
        return func

    return inner


@sat_mode('WRAP')
def wrap(x, k, i, f):
    xs = x
    bk = i + k
    bias = k * 2.0 ** (bk - 1)
    return ops.stop_gradient((xs + bias) % (2.0**bk) - bias) + (x - ops.stop_gradient(x))


@sat_mode('SAT')
def sat(x, k, i, f):
    f_eps = 2.0 ** (-f)
    __max = 2.0**i
    _max = __max - f_eps
    _min = -__max * k
    r = _clip(x, _min, _max)
    return r


@sat_mode('SAT_SYM')
def sat_sym(x, k, i, f):
    f_eps = 2.0 ** (-f)
    _max = 2.0**i - f_eps
    _min = -_max * k
    r = _clip(x, _min, _max)
    return r


@sat_mode('WRAP_SM')
def wrap_sm_fn(x, k, i, f, quant_fn: Callable = lambda x: x):
    # x=ops.round(x*2.**f)
    # High and low bounds are reflective. When overflows, can be less trash than WARP but still more trash than SAT.
    eps = 2.0**-f
    high = 2.0**i - eps
    low = -(high + eps) * k
    interval = 2.0 ** (i + k)
    c1 = ((x) / interval) % 2 >= 1  # type: ignore
    c1 = c1 & (ops.abs(x) > eps / 2)
    c2 = ((x + eps / 2) / (interval / 2)) % 2 >= 1  # type: ignore
    qx = quant_fn(x)
    mapped = ((qx - high - eps) % interval) + low

    mapped = ops.where(c2, -mapped - eps, mapped)  # type: ignore
    mapped = ops.where(c1, -mapped - eps, mapped)  # type: ignore

    return mapped


class FixedPointQuantizer:
    def round(self, x, f: Any = 1.0):
        scale = 2.0**f
        x = x * scale
        xq = self.round_fn(x)
        xq = xq / scale
        return xq

    def saturate(self, x, k, i, f):
        return self.sat_fn(x, k, i, f)

    def __init__(self, round_mode: str = 'TRN', overflow_mode: str = 'WRAP'):
        round_mode = round_mode.upper()
        overflow_mode = overflow_mode.upper()

        if overflow_mode == 'WRAP_SM':
            assert round_mode in (
                'RND',
                'RND_CONV',
            ), 'WRAP_SM only supports RND and RND_CONV rounding modes in this implementation.'

        self.round_mode = round_mode
        self.overflow_mode = overflow_mode
        round_fn = round_mode_registry[round_mode]
        sat_fn = saturation_mode_registry[overflow_mode]
        self.sat_fn = sat_fn
        self.round_fn = round_fn

    def forward(self, x, k, i, f):
        # Workaround for gradient computation around 0.
        # When have values outside boundary rounded to boundary, grad on f presents despite the value will be clipped off anyway.
        # Thus have saturation before rounding, except for wrap mode, which doesn't round during training.

        if self.overflow_mode != 'WRAP':
            x = self.saturate(x, k, i, f)
        x = self.round(x, f)
        if self.overflow_mode == 'WRAP':
            x = self.saturate(x, k, i, f)
        return x

    def forward_wrap_sm(self, x, k, i, f):
        def quant_fn(x):
            return self.round(x, f)

        x = wrap_sm_fn(x, k, i, f, quant_fn)
        return x

    def __call__(self, x, k, i, f):
        i = ops.stop_gradient(ops.maximum(i, -f)) + (i - ops.stop_gradient(i))  # type: ignore
        if self.overflow_mode != 'WRAP_SM':
            return self.forward(x, k, i, f)
        else:
            return self.forward_wrap_sm(x, k, i, f)


def fixed(x, k, i, f, round_mode: str = 'TRN', overflow_mode: str = 'WRAP'):
    return FixedPointQuantizer(round_mode, overflow_mode)(x, k, i, f)
