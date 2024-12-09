from collections.abc import Callable
from typing import Any, TypeVar

import numpy as np

round_mode_registry_np: dict[str, Callable[[Any, bool | None], Any]] = {}
round_mode_registry_scaled_np: dict[str, Callable[[Any, Any, bool | None], Any]] = {}
saturation_mode_registry_np: dict[str, Callable[[Any, Any, Any, Any, bool | None], Any]] = {}

T = TypeVar('T', bound=np.ndarray)


def rnd_mode_np(names: str | list[str] | tuple[str, ...]):
    names = (names,) if isinstance(names, str) else names

    def inner(func):
        def wrapper(x, f, training=None, seed_gen=None):
            scale = 2.0**f
            sx = x * scale
            sxq = func(sx)
            xq = sxq / scale
            return xq

        for name in names:
            assert name not in round_mode_registry_scaled_np, f"Rounding mode '{name}' already exists."
            round_mode_registry_scaled_np[name] = wrapper
        round_mode_registry_scaled_np[func.__name__.upper()] = wrapper

        for name in names:
            assert name not in round_mode_registry_np, f"Rounding mode '{name}' already exists."
            round_mode_registry_np[name] = func
        round_mode_registry_np[func.__name__.upper()] = func

        return func

    return inner


@rnd_mode_np('TRN')
def floor(x):
    return np.floor(x)


@rnd_mode_np('RND')
def round(x):
    return np.floor(x + 0.5)


@rnd_mode_np('RND_CONV')
def round_conv(x):
    return np.round(x)


@rnd_mode_np('TRN_ZERO')
def floor_zero(x):
    sign = np.sign(x)
    return np.floor(np.abs(x)) * sign


@rnd_mode_np('RND_ZERO')
def round_zero(x):
    sign = np.sign(x)
    return -np.floor(-np.abs(x) + 0.5) * sign


@rnd_mode_np('RND_MIN_INF')
def round_min_inf(x):
    return -np.floor(-x + 0.5)


@rnd_mode_np('RND_INF')
def round_inf(x):
    sign = np.sign(x)
    return np.floor(np.abs(x) + 0.5) * sign


def sat_mode_np(name: str | list | tuple):
    names = (name,) if isinstance(name, str) else name

    def inner(func):
        for name in names:
            assert name not in saturation_mode_registry_np, f"Saturation mode '{name}' already exists."
            saturation_mode_registry_np[name] = func
        saturation_mode_registry_np[func.__name__.upper()] = func
        return func

    return inner


@sat_mode_np('WRAP')
def wrap(x, k, i, f, training=None):
    xs = x
    bk = i + k
    bias = k * 2.0 ** (bk - 1)
    return (xs + bias) % (2.0**bk) - bias


@sat_mode_np('SAT')
def sat(x, k, i, f, training=None):
    f_eps = 2.0 ** (-f)
    __max = 2.0**i
    _max = __max - f_eps
    _min = -__max * k
    return np.clip(x, _min, _max)


@sat_mode_np('SAT_SYM')
def sat_sym(x, k, i, f, training=None):
    f_eps = 2.0 ** (-f)
    _max = 2.0**i - f_eps
    _min = -_max * k
    return np.clip(x, _min, _max)


@sat_mode_np('WRAP_SM')
def wrap_sm_fn(x, k, i, f, training=None, quant_fn: Callable = lambda x: x):
    eps = 2.0**-f
    high = 2.0**i - eps
    low = -(high + eps) * k
    interval = 2 ** (i + k)
    c1 = ((x) / interval) % 2 >= 1
    c1 = c1 & (np.abs(x) > eps / 2)
    c2 = ((x + eps / 2) / (interval / 2)) % 2 >= 1
    qx = quant_fn(x)
    mapped = ((qx - high - eps) % interval) + low

    mapped = np.where(c2, -mapped - eps, mapped)
    mapped = np.where(c1, -mapped - eps, mapped)

    return mapped


def get_fixed_quantizer_np(round_mode: str = 'TRN', overflow_mode: str = 'WRAP'):
    round_mode = round_mode.upper()
    overflow_mode = overflow_mode.upper()
    round_mode = round_mode[2:] if round_mode.startswith('S_') else round_mode

    round_fn_scaled = round_mode_registry_scaled_np.get(round_mode, None)
    if round_fn_scaled is None:
        raise ValueError(f'Unknown rounding mode: {round_mode}')
    sat_fn = saturation_mode_registry_np.get(overflow_mode, None)
    if sat_fn is None:
        raise ValueError(f'Unknown saturation mode: {overflow_mode}')

    if overflow_mode == 'WRAP_SM':
        assert round_mode in ('RND', 'RND_CONV'), 'WRAP_SM only supports RND and RND_CONV rounding modes in this implementation.'

    def quantizer(x: T, k: T, i: T, f: T, training: bool | None = False, seed_gen: None = None) -> np.ndarray:
        assert not training, 'Training mode not supported in numpy implementation.'

        _i = np.maximum(i, -f)

        if overflow_mode == 'WRAP_SM':

            def rnd_fn_wrapped(x):
                return round_fn_scaled(x, f, training)

            return wrap_sm_fn(x, k, _i, f, training, rnd_fn_wrapped)

        if overflow_mode != 'WRAP':
            x = sat_fn(x, k, _i, f, training)
        x = round_fn_scaled(x, f, training)
        if overflow_mode == 'WRAP':
            x = sat_fn(x, k, _i, f, training)
        return x

    return quantizer
