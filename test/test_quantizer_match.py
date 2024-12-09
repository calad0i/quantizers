from pathlib import Path

import cppyy
import numpy as np
import pytest
from keras import ops

from quantizers import BinaryQ, FixedQ, MinifloatQ, TernaryQ


@pytest.fixture(scope='session')
def register_cpp():
    path = Path(__file__).parent
    cppyy.add_include_path(f'{path}/cpp_source')
    cppyy.add_include_path(f'{path}/cpp_source/ap_types')
    cppyy.include('quantizers.h')


@pytest.fixture(scope='session')
def data():
    arr = np.random.randint(-2**15, 2**15, 1000) * 2.**-8
    return ops.array(arr.astype(np.float32))


def c_quantize_fixed(x, k, i, f, round_mode, overflow_mode):
    W, I = k + i + f, k + i
    if round_mode.startswith('S_'):
        round_mode = round_mode[2:]
    round_mode = f'AP_{round_mode}'
    overflow_mode = f'AP_{overflow_mode}'

    fn = cppyy.gbl.qkn_test.fixedq[W, I, k, round_mode, overflow_mode]

    r = fn(x)
    return r


def c_quantize_float(x, M, E, E0):
    fn = cppyy.gbl.qkn_test.floatq[M, E, E0]
    r = fn(x)
    return r


def c_quantize_binary(x):
    fn = cppyy.gbl.qkn_test.binaryq
    r = fn(x)
    return r


def c_quantize_ternary(x):
    fn = cppyy.gbl.qkn_test.ternaryq
    r = fn(x)
    return r


@pytest.mark.parametrize('fixed_round_mode', ['TRN', 'TRN_ZERO', 'RND', 'S_RND', 'S_RND_CONV', 'RND_CONV', 'RND_ZERO', 'RND_INF', 'RND_MIN_INF'])
@pytest.mark.parametrize('fixed_overflow_mode', ['WRAP', 'WRAP_SM', 'SAT', 'SAT_SYM'])
@pytest.mark.parametrize('k', [0, 1])
@pytest.mark.parametrize('b', [2, 7, 12])
@pytest.mark.parametrize('i', [-5, 0, 8])
def test_fixed_quantizer_forward(fixed_round_mode, fixed_overflow_mode, k, b, i, data, register_cpp):

    k, i, f = k, i, b - i
    W, I = k + b, k + i
    if fixed_overflow_mode == 'WRAP_SM':
        if k == 0:
            pytest.skip('WRAP_SM does not support k=0')
        if fixed_round_mode not in ('RND_CONV', 'RND'):
            pytest.skip('WRAP_SM only supports RND-like rounding')

    fixed_q = FixedQ(W, I, k, fixed_round_mode, fixed_overflow_mode)
    print(fixed_q)

    arr_py_fixed_np = np.array(fixed_q(np.array(data)))
    arr_py_fixed = np.array(fixed_q(data))
    assert np.all(arr_py_fixed_np == arr_py_fixed), 'numpy / keras implementation inconsistent'

    arr_c_fixed = c_quantize_fixed(data, k, i, f, fixed_round_mode, fixed_overflow_mode)
    arr_c_fixed_np = np.array([float(x) for x in arr_c_fixed])

    mismatch = np.where(arr_py_fixed != arr_c_fixed_np)[0]
    assert len(mismatch) == 0, f'''Fixed quantizer has inconsistent behavior with C++ implementation:
        [* {len(mismatch)} mismatches, {min(len(mismatch), 5)} shown *]
        C++: {arr_c_fixed_np[mismatch][:5]}
        Py: {arr_py_fixed[mismatch][:5]}
        in: {data[mismatch][:5]}
    '''


@pytest.mark.parametrize('M', [2, 6])
@pytest.mark.parametrize('E', [2, 6])
@pytest.mark.parametrize('E0', [0, 3, 5])
def test_minifloat_quantizer_forward(M, E, E0, data, register_cpp):

    q = MinifloatQ(M, E, E0)
    print(q)
    arr_py_float = np.array(q(data))
    arr_py_float_np = np.array(q(np.array(data)))
    assert np.all(arr_py_float_np == arr_py_float), 'numpy / keras implementation inconsistent'

    arr_c_float = c_quantize_float(data, M, E, E0)
    arr_c_float_np = np.array([float(x) for x in arr_c_float])

    mismatch = np.where(arr_py_float != arr_c_float_np)[0]
    assert len(mismatch) == 0, f'''Float quantizer has inconsistent behavior with C++ implementation:
        [* {len(mismatch)} mismatches, {min(len(mismatch), 5)} shown *]
        [* Up to 5 shown]
        C++: {arr_c_float_np[mismatch][:5]}
        Py: {arr_py_float[mismatch][:5]}
        in: {data[mismatch][:5]}
    '''


@pytest.mark.parametrize('fixed_round_mode', ['RND'])
@pytest.mark.parametrize('fixed_overflow_mode', ['SAT'])
@pytest.mark.parametrize('k', [0, 1])
@pytest.mark.parametrize('i', [3, -1])
@pytest.mark.parametrize('f', [4, 2])
@pytest.mark.parametrize('M', [2, 4])
@pytest.mark.parametrize('E', [2, 4])
@pytest.mark.parametrize('E0', [1, 8])
def test_fixed_float_mult(fixed_round_mode, fixed_overflow_mode, k, i, f, M, E, E0, data, register_cpp):

    if fixed_overflow_mode == 'WRAP_SM':
        if k == 0:
            pytest.skip('WRAP_SM does not support k=0')
        if fixed_round_mode not in ('RND_CONV', 'RND'):
            pytest.skip('WRAP_SM only supports RND-like rounding')

    arr_c_fixed = c_quantize_fixed(data, k, i, f, fixed_round_mode, fixed_overflow_mode)
    arr_c_float = c_quantize_float(data, M, E, E0)
    arr_c_fixed_np = np.array([float(x) for x in arr_c_fixed])
    arr_c_float_np = np.array([float(x) for x in arr_c_float])

    mult_c = np.array([(fp * fx).to_double() for fx, fp in zip(arr_c_fixed, arr_c_float)])
    mult_py = arr_c_fixed_np * arr_c_float_np

    mismatch = np.where(mult_py != mult_c)[0]
    print(len(mismatch))
    assert len(mismatch) == 0, f'''Multiplication has inconsistent behavior with C++ implementation:
        [* {len(mismatch)} mismatches, {min(len(mismatch), 5)} shown *]
        C++: {mult_c[mismatch][:5]}
        Py: {mult_py[mismatch][:5]}
        fx in: {arr_c_fixed_np[mismatch][:5]}
        fp in: {arr_c_float_np[mismatch][:5]}
    '''


def test_binary(data, register_cpp):
    q = BinaryQ()
    print(q)
    arr_py_binary = q(data)
    arr_py_binary_np = np.array(q(np.array(data)))
    assert np.all(arr_py_binary_np == arr_py_binary), 'numpy / keras implementation inconsistent'

    arr_c_binary = c_quantize_binary(data)
    arr_c_binary_np = np.array([float(x) for x in arr_c_binary])

    mismatch = np.where(arr_py_binary != arr_c_binary_np)[0]
    assert len(mismatch) == 0, f'''Binary quantizer has inconsistent behavior with C++ implementation:
        [* {len(mismatch)} mismatches, {min(len(mismatch), 5)} shown *]
        C++: {arr_c_binary_np[mismatch][:5]}
        Py: {arr_py_binary[mismatch][:5]}
        in: {data[mismatch][:5]}
    '''


def test_ternary(data, register_cpp):
    q = TernaryQ()
    print(q)
    arr_py_ternary = q(data)
    arr_py_ternary_np = np.array(q(np.array(data)))
    assert np.all(arr_py_ternary_np == arr_py_ternary), 'numpy / keras implementation inconsistent'

    arr_c_ternary = c_quantize_ternary(data)
    arr_c_ternary_np = np.array([float(x) for x in arr_c_ternary])
    mismatch = np.where(arr_py_ternary != arr_c_ternary_np)[0]
    assert len(mismatch) == 0, f'''Ternary quantizer has inconsistent behavior with C++ implementation:
        [* {len(mismatch)} mismatches, {min(len(mismatch), 5)} shown *]
        C++: {arr_c_ternary_np[mismatch][:5]}
        Py: {arr_py_ternary[mismatch][:5]}
        in: {data[mismatch][:5]}
    '''
