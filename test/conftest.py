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

        def closure(*args):
            xs = [torch.as_tensor(a) if i in _argnums else a for i, a in enumerate(args)]
            for i, x in enumerate(xs):
                if i in _argnums:
                    x.requires_grad_(True)
            y = fn(*xs)
            y.sum().backward()
            r = tuple(x.grad.detach().numpy() for i, x in enumerate(xs) if i in _argnums)
            if isinstance(argnums, int):
                return r[0]
            return r

        return closure

    return FWDict(
        make=torch.as_tensor,
        to_np=lambda x: x.detach().numpy() if getattr(x, 'requires_grad', False) else np.asarray(x),
        grad=grad,
        extra=[torch.nn.Parameter(torch.ones(2))],
        abs=lambda x: torch.sum(torch.abs(x)),
        jit=None,
    )


def _tensorflow():
    tf = pytest.importorskip('tensorflow')

    def grad(fn, argnums=0):
        _argnums = (argnums,) if isinstance(argnums, int) else tuple(argnums)

        def closure(*args):
            xs = [tf.convert_to_tensor(a) if i in _argnums else a for i, a in enumerate(args)]
            with tf.GradientTape() as tape:
                tape.watch([xs[i] for i in _argnums])
                y = tf.reduce_sum(fn(*xs))
            r = tape.gradient(y, [xs[i] for i in _argnums], unconnected_gradients=tf.UnconnectedGradients.ZERO)
            if isinstance(argnums, int):
                return r[0]
            return r

        return closure

    return FWDict(
        make=tf.convert_to_tensor,
        to_np=lambda x: x.numpy(),
        grad=grad,
        extra=[tf.Variable(1.0)],
        jit=lambda fn: tf.function(fn, autograph=False),
        abs=tf.abs,
    )


def _mlx():
    mx = pytest.importorskip('mlx.core')
    mx.set_default_device(mx.cpu)

    def grad(fn, argnums=0):
        _argnums = (argnums,) if isinstance(argnums, int) else tuple(argnums)

        def closure(*args) -> tuple[Any, ...]:
            xs = [mx.array(a) if i in _argnums else a for i, a in enumerate(args)]
            r = mx.grad(lambda *xs: fn(*xs).sum(), argnums=list(_argnums))(*xs)
            if isinstance(argnums, int):
                return r[0]
            return r

        return closure

    return FWDict(make=mx.array, to_np=np.asarray, grad=grad, extra=[], jit=None, abs=mx.abs)


def _tinygrad():
    tinygrad = pytest.importorskip('tinygrad')
    from tinygrad.helpers import DEV

    # Default device for the tensors created internally, e.g. by Tensor.gradient. Otherwise tinygrad picks one by reading env
    # vars named after its devices, and crashes on e.g. CPU=x86_64, set by some shells.
    DEV.value = 'CPU'
    Tensor = tinygrad.Tensor

    def grad(fn, argnums=0):
        _argnums = (argnums,) if isinstance(argnums, int) else tuple(argnums)

        def closure(*args) -> tuple[Any, ...]:
            xs = [Tensor(np.asarray(a), device='CPU') if i in _argnums else a for i, a in enumerate(args)]
            r = fn(*xs).sum().gradient(*(xs[i] for i in _argnums))
            if isinstance(argnums, int):
                return r[0]
            return r

        return closure

    return FWDict(
        make=lambda x: Tensor(np.asarray(x), device='CPU'),
        to_np=lambda x: x.numpy(),
        grad=grad,
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
