import subprocess

import numpy as np
import pytest
from gen_configs import CPP_SOURCE, gen_configs


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
    return dict(
        make=jax.numpy.asarray,
        to_np=np.asarray,
        grad=lambda fn, x: jax.grad(lambda x: fn(x).sum())(x),
        extra=[],
        jit=jax.jit,
    )


def _torch():
    torch = pytest.importorskip('torch')

    def grad(fn, x):
        x = x.detach().requires_grad_()
        y = fn(x).sum()
        if not y.requires_grad:  # independent of x
            return None
        y.backward()
        return x.grad

    return dict(make=torch.as_tensor, to_np=lambda x: x.detach().numpy(), grad=grad, extra=[torch.nn.Parameter(torch.ones(2))])


def _tensorflow():
    tf = pytest.importorskip('tensorflow')

    def grad(fn, x):
        x = tf.Variable(x)
        with tf.GradientTape() as tape:
            y = tf.reduce_sum(fn(x))
        return tape.gradient(y, x)

    return dict(
        make=tf.convert_to_tensor,
        to_np=lambda x: x.numpy(),
        grad=grad,
        extra=[tf.Variable(1.0)],
        jit=lambda fn: tf.function(fn, autograph=False),
    )


def _mlx():
    mx = pytest.importorskip('mlx.core')
    mx.set_default_device(mx.cpu)
    return dict(make=mx.array, to_np=np.asarray, grad=lambda fn, x: mx.grad(lambda x: fn(x).sum())(x), extra=[])


def _tinygrad():
    Tensor = pytest.importorskip('tinygrad').Tensor
    return dict(
        make=lambda x: Tensor(np.asarray(x), device='CPU'),
        to_np=lambda x: x.numpy(),
        grad=lambda fn, x: fn(x).sum().gradient(x)[0],
        extra=[],
    )


FRAMEWORKS = {
    'numpy': _numpy,
    'jax': _jax,
    'torch': _torch,
    'tensorflow': _tensorflow,
    'mlx.core': _mlx,
    'tinygrad': _tinygrad,
}


@pytest.fixture(params=list(FRAMEWORKS), scope='module')
def fw(request):
    return request.param, FRAMEWORKS[request.param]()


@pytest.fixture(scope='module')
def jax_fw():
    return FRAMEWORKS['jax']()
