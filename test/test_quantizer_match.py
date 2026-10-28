import ctypes

import numpy as np
import pytest
from conftest import FWDict
from gen_configs import LIB_PATH, MATCH_CONFIGS_FIXED, MATCH_CONFIGS_FLOAT, fixed_symbol, float_symbol
from numpy.ctypeslib import ndpointer

from quantizers import BinaryQ, FixedQ, MinifloatQ, TernaryQ


@pytest.fixture(scope='session')
def lib():
    return ctypes.CDLL(str(LIB_PATH))  # built in conftest.pytest_sessionstart


@pytest.fixture(scope='session')
def data():
    arr = np.random.default_rng(0).integers(-(2**15), 2**15, 1000) * 2.0**-8
    return arr.astype(np.float32)


def c_quantize(lib, symbol: str, x: np.ndarray, dtype: type = np.float32) -> np.ndarray:
    fn = getattr(lib, symbol)
    fn.argtypes = [ndpointer(np.float32, flags='C'), ndpointer(dtype, flags='C'), ctypes.c_size_t]
    fn.restype = None
    out = np.empty_like(x, dtype=dtype)
    fn(x, out, x.size)
    return out


@pytest.mark.parametrize('config', MATCH_CONFIGS_FIXED, ids=lambda c: fixed_symbol(*c))
def test_fixed_quantizer_forward(config, data, lib, fw: FWDict):
    S, W, I, RND, SAT = config
    fixed_q = FixedQ(W, I, S, RND, SAT)
    print(fixed_q)

    arr_py_fixed = np.asarray(fixed_q(fw['make'](data)))
    arr_c_fixed = c_quantize(lib, fixed_symbol(*config), data)

    mismatch = np.where(arr_py_fixed != arr_c_fixed)[0]
    assert len(mismatch) == 0, f"""Fixed quantizer has inconsistent behavior with C++ implementation:
        [* {len(mismatch)} mismatches, {min(len(mismatch), 5)} shown *]
        C++: {arr_c_fixed[mismatch][:5]}
        Py: {arr_py_fixed[mismatch][:5]}
        in: {data[mismatch][:5]}
    """


@pytest.mark.parametrize('config', MATCH_CONFIGS_FLOAT, ids=lambda c: float_symbol(*c))
def test_minifloat_quantizer_forward(config, data, lib, fw: FWDict):
    q = MinifloatQ(*config)
    print(q)

    arr_py_float = np.asarray(q(fw['make'](data)))
    arr_c_float = c_quantize(lib, float_symbol(*config), data)

    mismatch = np.where(arr_py_float != arr_c_float)[0]
    assert len(mismatch) == 0, f"""Float quantizer has inconsistent behavior with C++ implementation:
        [* {len(mismatch)} mismatches, {min(len(mismatch), 5)} shown *]
        C++: {arr_c_float[mismatch][:5]}
        Py: {arr_py_float[mismatch][:5]}
        in: {data[mismatch][:5]}
    """


def test_binary(data, lib, fw: FWDict):
    q = BinaryQ()
    print(q)

    arr_py_binary = np.asarray(q(fw['make'](data)))
    arr_c_binary = c_quantize(lib, 'binary', data)

    mismatch = np.where(arr_py_binary != arr_c_binary)[0]
    assert len(mismatch) == 0, f"""Binary quantizer has inconsistent behavior with C++ implementation:
        [* {len(mismatch)} mismatches, {min(len(mismatch), 5)} shown *]
        C++: {arr_c_binary[mismatch][:5]}
        Py: {arr_py_binary[mismatch][:5]}
        in: {data[mismatch][:5]}
    """


def test_ternary(data, lib, fw: FWDict):
    q = TernaryQ()
    print(q)

    arr_py_ternary = np.asarray(q(fw['make'](data)))
    arr_c_ternary = c_quantize(lib, 'ternary', data)

    mismatch = np.where(arr_py_ternary != arr_c_ternary)[0]
    assert len(mismatch) == 0, f"""Ternary quantizer has inconsistent behavior with C++ implementation:
        [* {len(mismatch)} mismatches, {min(len(mismatch), 5)} shown *]
        C++: {arr_c_ternary[mismatch][:5]}
        Py: {arr_py_ternary[mismatch][:5]}
        in: {data[mismatch][:5]}
    """
