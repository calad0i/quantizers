import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from conftest import FWDict

from quantizers._backends import KerasBackend, NumpyBackend, get_backend

X = np.array([-2.5, -1.5, -0.75, -0.5, 0.0, 0.5, 0.75, 1.5, 2.5, 3.25], np.float32)


def test_dispatch(fw: FWDict):
    bk = get_backend(fw['make'](X))
    assert get_backend(fw['make'](X)) is bk
    for t in fw['extra']:
        assert get_backend(t) is bk


def test_dispatch_jax_jit():
    jax = pytest.importorskip('jax')

    @jax.jit
    def fn(x):
        bk = get_backend(x)
        assert bk.name == 'jax'
        return bk.floor(x)

    np.testing.assert_array_equal(fn(X), np.floor(X))


def test_jax_partial_grad():
    """jax grad with only some of the parameters differentiable"""
    jax = pytest.importorskip('jax')
    from quantizers import FixedPointQuantizer, float_quantize

    w = np.arange(1, len(X) + 1, dtype=np.float32)
    cases = [
        (FixedPointQuantizer('RND', 'SAT'), (1.0, 1.0, 1.0)),
        (float_quantize, (0.0, 2.0, -1.0)),
    ]
    for q, params in cases:

        def loss(*args, q=q):
            return jax.numpy.sum(q(*args) * w)

        args = [jax.numpy.asarray(X)] + [jax.numpy.full(len(X), p) for p in params]
        joint = jax.grad(loss, argnums=range(len(args)))(*args)
        assert all(np.any(g) for g in joint[1:])
        for j, g in enumerate(joint):
            np.testing.assert_array_equal(jax.grad(loss, argnums=j)(*args), g)


def test_unknown_type():
    with pytest.raises(TypeError):
        get_backend(object())
    assert isinstance(get_backend(1.0), NumpyBackend)


@pytest.mark.parametrize('op', ['floor', 'round_conv', 'abs', 'tanh', 'log2'])
def test_unary(fw, op):
    ref = {'round_conv': np.round}.get(op) or getattr(np, op)
    x_np = np.abs(X) + 1 if op == 'log2' else X
    x = fw['make'](x_np)
    np.testing.assert_allclose(fw['to_np'](getattr(get_backend(x), op)(x)), ref(x_np), rtol=1e-6)


def test_scalar_args(fw: FWDict):
    x = fw['make'](X.astype(np.float16))
    bk = get_backend(x)

    def compute(x):
        return (
            bk.where(x > 0, x, 0.0),
            bk.where(x > 0, 0.0, x),
            bk.clip(x, -0.5, 1.5),
            bk.floor(x),
            bk.round_conv(x),
            bk.stop_gradient(x),
        )

    refs = [np.where(X > 0, X, 0), np.where(X > 0, 0, X), np.clip(X, -0.5, 1.5), np.floor(X), np.round(X), X]
    for wrap in (lambda fn: fn, fw.get('jit')):
        if wrap is None:
            continue
        for r, ref in zip(wrap(compute)(x), refs):
            assert 'float16' in str(r.dtype) or 'half' in str(r.dtype), f'float16 not preserved, got {r.dtype}'
            np.testing.assert_array_equal(fw['to_np'](r), ref.astype(np.float16))
    np.testing.assert_array_equal(fw['to_np'](bk.where(x > 0, 1.0, -1.0)), np.where(X > 0, 1.0, -1.0))


def test_asarray(fw: FWDict):
    x = fw['make'](X.astype(np.float16))
    bk = get_backend(x)
    if bk.name == 'numpy':
        assert bk.asarray(0.5, like=x) == 0.5, 'python scalars are native to numpy'
    for v in ([1.0, 2.0],) if bk.name == 'numpy' else (0.5, 2, np.float64(0.5), np.array([1.0, 2.0])):
        r = bk.asarray(v, like=x)
        assert get_backend(r) is bk
        assert r.dtype == x.dtype  # type: ignore
    native = fw['make'](X)
    assert bk.asarray(native, like=x) is native, 'native tensors are returned as is'


def _check_keras_variable():
    import keras

    var = keras.Variable(X)
    bk = get_backend(var)
    assert isinstance(bk, KerasBackend), bk
    assert bk.native is get_backend(keras.ops.convert_to_tensor(X))
    assert bk.native.name == keras.backend.backend()

    for r, ref in [
        (bk.floor(var), np.floor(X)),
        (bk.round_conv(var), np.round(X)),
        (bk.abs(var), np.abs(X)),
        (bk.where(bk.asarray(X) > 0, var, 0.0), np.where(X > 0, X, 0)),
        (bk.clip(var, -1.0, 1.0), np.clip(X, -1, 1)),
        (bk.stop_gradient(var), X),
        (bk.asarray(0.5, like=var), np.float32(0.5)),
    ]:
        assert get_backend(r) is bk.native, type(r)
        np.testing.assert_array_equal(keras.ops.convert_to_numpy(r), ref)

    print(bk)
    assert bk.differentiable == bk.native.differentiable


@pytest.mark.parametrize('keras_backend', ['jax', 'torch', 'tensorflow'])
def test_keras_variable(keras_backend):
    pytest.importorskip('keras')
    if importlib.util.find_spec(keras_backend) is None:
        pytest.skip(f'{keras_backend} not installed')
    code = f'import sys; sys.path.insert(0, {str(Path(__file__).parent)!r}); import test_backends; test_backends._check_keras_variable()'
    env = {**os.environ, 'KERAS_BACKEND': keras_backend}
    r = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
