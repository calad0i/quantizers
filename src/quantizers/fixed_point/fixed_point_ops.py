from collections.abc import Callable
from typing import Any

from .._backends import AbstractBackend, get_backend

round_mode_registry: dict[str, Callable[..., Any]] = {}
saturation_mode_registry: dict[str, Callable[..., Any]] = {}


def rnd_mode(name: str):
    def inner(func):
        assert name not in round_mode_registry, f"Round mode '{name}' already exists."
        round_mode_registry[name] = func
        return func

    return inner


@rnd_mode('TRN')
def floor(x, bk: AbstractBackend | None = None):
    bk = bk or get_backend(x)
    return bk.floor(x)


@rnd_mode('RND')
def round(x, bk: AbstractBackend | None = None):
    # Round to nearest, ties positive infinity.
    bk = bk or get_backend(x)
    return bk.floor(x + 0.5)


@rnd_mode('RND_CONV')
def round_conv(x, bk: AbstractBackend | None = None):
    # Round to nearest, ties to even.
    bk = bk or get_backend(x)
    return bk.round_conv(x)


@rnd_mode('TRN_ZERO')
def floor_zero(x, bk: AbstractBackend | None = None):
    # Truncate towards zero.
    bk = bk or get_backend(x)
    return bk.where(x < 0, -bk.floor(-x), bk.floor(x))


@rnd_mode('RND_ZERO')
def round_zero(x, bk: AbstractBackend | None = None):
    # Round to nearest, ties towards zero.
    bk = bk or get_backend(x)
    return bk.where(x < 0, bk.floor(x + 0.5), -bk.floor(-x + 0.5))


@rnd_mode('RND_MIN_INF')
def round_min_inf(x, bk: AbstractBackend | None = None):
    # Round to nearest, ties towards negative infinity.
    bk = bk or get_backend(x)
    return -bk.floor(-x + 0.5)


@rnd_mode('RND_INF')
def round_inf(x, bk: AbstractBackend | None = None):
    # Round to nearest, ties away from zero.
    bk = bk or get_backend(x)
    return bk.where(x < 0, -bk.floor(-x + 0.5), bk.floor(x + 0.5))


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
def wrap(x, k, i, f, bk: AbstractBackend | None = None):
    bk = bk or get_backend(x)
    b = i + k
    bias = k * 2.0 ** (b - 1)
    in_range = (x >= -bias) & (x <= 2.0**i - 2.0**-f)
    return bk.where(in_range, x, bk.stop_gradient((x + bias) % (2.0**b) - bias))


@sat_mode('SAT_ZERO')
def sat_zero(x, k, i, f, bk: AbstractBackend | None = None):
    bk = bk or get_backend(x)
    b = i + k
    bias = k * 2.0 ** (b - 1)
    in_range = (x >= -bias) & (x <= 2.0**i - 2.0**-f)
    return bk.where(in_range, x, 0.0)


@sat_mode('SAT')
def sat(x, k, i, f, bk: AbstractBackend | None = None):
    bk = bk or get_backend(x)
    f_eps = 2.0 ** (-f)
    __max = 2.0**i
    _max = __max - f_eps
    _min = -__max * k
    return bk.clip(x, _min, _max)


@sat_mode('SAT_SYM')
def sat_sym(x, k, i, f, bk: AbstractBackend | None = None):
    bk = bk or get_backend(x)
    f_eps = 2.0 ** (-f)
    _max = 2.0**i - f_eps
    _min = -_max * k
    return bk.clip(x, _min, _max)


class FixedPointQuantizer:
    def round(self, x, f: Any = 1.0, bk: AbstractBackend | None = None):
        bk = bk or get_backend(x)
        scale = 2.0**f
        x = x * scale
        xq = self.round_fn(x, bk=bk)
        xq = xq / scale
        return xq

    def saturate(self, x, k, i, f, bk: AbstractBackend | None = None):
        bk = bk or get_backend(x)
        return self.sat_fn(x, k, i, f, bk=bk)

    def __init__(self, round_mode: str = 'TRN', overflow_mode: str = 'WRAP', null_xgrad_on_zero_width: bool = False):
        round_mode = round_mode.upper()
        overflow_mode = overflow_mode.upper()
        self.round_mode = round_mode
        self.overflow_mode = overflow_mode
        round_fn = round_mode_registry[round_mode]
        sat_fn = saturation_mode_registry[overflow_mode]
        self.sat_fn = sat_fn
        self.round_fn = round_fn
        self.null_xgrad_on_zero_width = null_xgrad_on_zero_width

    def forward(self, x, k, i, f, bk: AbstractBackend | None = None):
        # Saturation before rounding: values out of range rounded onto the boundary would get a gradient on f, though clipped
        # off anyway. Wrapping and zeroing after rounding, which decides whether a value overflows.
        bk = bk or get_backend(x)
        if self.overflow_mode in ('WRAP', 'SAT_ZERO'):
            return self.saturate(self.round(x, f, bk=bk), k, i, f, bk=bk)
        return self.round(self.saturate(x, k, i, f, bk=bk), f, bk=bk)

    def __call__(self, x, k, i, f):
        bk = get_backend(x)
        k, i, f = (bk.asarray(v, like=x) for v in (k, i, f))
        has_bits = k + i + f > 0
        i = bk.passthrough(i, bk.where(i > -f, i, -f))  # i >= -f, gradient on i
        if self.null_xgrad_on_zero_width:
            x = bk.where(has_bits, x, bk.stop_gradient(x))
        xq = self.forward(x, k, i, f, bk=bk)
        return bk.passthrough(xq, bk.where(has_bits, xq, 0.0))


def fixed_quantize(x, k, i, f, round_mode: str = 'TRN', overflow_mode: str = 'WRAP'):
    return FixedPointQuantizer(round_mode, overflow_mode)(x, k, i, f)
