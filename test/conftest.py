import subprocess
from collections.abc import Callable
from typing import Any, TypedDict

import numpy as np
import pytest
from gen_configs import CPP_SOURCE, gen_configs


class FWDict(TypedDict):
    make: Callable
    to_np: Callable
    grad: Callable | None
    extra: list
    jit: Callable | None
    abs: Callable


def pytest_sessionstart(session):
    if hasattr(session.config, 'workerinput'):
        return  # xdist worker
    gen_configs()
    nproc = session.config.getoption('numprocesses', None) or 1
    r = subprocess.run(['make', '-C', str(CPP_SOURCE), f'-j{nproc}'], capture_output=True, text=True)
    if r.returncode:
        pytest.exit(
            f'Failed to build the C++ reference quantizers:\n{r.stdout}{r.stderr}', returncode=pytest.ExitCode.INTERNAL_ERROR
        )


def _numpy():
    return dict(make=np.asarray, to_np=np.asarray, grad=None, extra=[np.float32(1.0), 1.0, 1])


def _jax():
    jax = pytest.importorskip('jax')
    return FWDict(
        make=jax.numpy.asarray,
        to_np=np.asarray,
        grad=lambda fn, argnums=0: jax.grad(lambda *args: jax.numpy.sum(fn(*args)), argnums=argnums),
        extra=[],
        jit=jax.jit,
        abs=jax.numpy.abs,
    )


def _torch():
    torch = pytest.importorskip('torch')

    def grad(fn, argnums=0):
        _argnums = (argnums,) if isinstance(argnums, int) else tuple(argnums)

        def closure(*args) -> tuple[Any, ...]:
            xs = [torch.as_tensor(a) for a in args]
            for i, x in enumerate(xs):
                if i in _argnums:
                    x.requires_grad_(True)
                else:
                    x.requires_grad_(False)
            y = fn(*xs)
            y.sum().backward()
            r = tuple(x.grad.detach().numpy() for i, x in enumerate(xs) if i in _argnums)
            if isinstance(argnums, int):
                return r[0]
            return r

        return closure

    return FWDict(
        make=torch.as_tensor,
        to_np=lambda x: x.detach().numpy(),
        grad=grad,
        extra=[torch.nn.Parameter(torch.ones(2))],
        abs=lambda x: torch.sum(torch.abs(x)),
        jit=None,
    )


def _tensorflow():
    tf = pytest.importorskip('tensorflow')

    def grad(fn, x):
        x = tf.Variable(x)
        with tf.GradientTape() as tape:
            y = tf.reduce_sum(fn(x))
        return tape.gradient(y, x)

    return FWDict(
        make=tf.convert_to_tensor,
        to_np=lambda x: x.numpy(),
        grad=grad,
        extra=[tf.Variable(1.0)],
        jit=lambda fn: tf.function(fn, autograph=False),
        abs=lambda x: tf.abs,
    )


def _mlx():
    mx = pytest.importorskip('mlx.core')
    mx.set_default_device(mx.cpu)
    return dict(make=mx.array, to_np=np.asarray, grad=lambda fn, x: mx.grad(lambda x: fn(x).abs().sum())(x), extra=[])


def _tinygrad():
    Tensor = pytest.importorskip('tinygrad').Tensor
    return FWDict(
        make=lambda x: Tensor(np.asarray(x), device='CPU'),
        to_np=lambda x: x.numpy(),
        grad=lambda fn, x: fn(x).sum().gradient(x)[0],
        extra=[],
        jit=None,
        abs=lambda x: x.abs(),
    )


FRAMEWORKS = {
    'numpy': _numpy,
    'jax': _jax,
    'torch': _torch,
    'tensorflow': _tensorflow,
    'mlx.core': _mlx,
    'tinygrad': _tinygrad,
}


@pytest.fixture(params=list(FRAMEWORKS), scope='session')
def _fw(request) -> tuple[str, FWDict]:
    return request.param, FRAMEWORKS[request.param]()


@pytest.fixture(scope='session')
def fw(_fw):
    return _fw[1]


@pytest.fixture(scope='session')
def jax_fw():
    return FRAMEWORKS['jax']()
