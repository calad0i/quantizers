from keras import ops
from keras.api.random import SeedGenerator


@ops.custom_gradient
def binary_quantize(x):
    r = ops.where(x > 0, 1., -1.)

    def grad(*args, upstream=None):
        if upstream is None:
            (upstream,) = args
        dy = upstream
        return dy * (1. - ops.tanh(x)**2)  # type: ignore

    return ops.stop_gradient(r), grad


@ops.custom_gradient
def ternary_quantize(x):
    r = ops.where(x > 0.5, 1., ops.where(x < -0.5, -1., 0.))

    def grad(*args, upstream=None):
        if upstream is None:
            (upstream,) = args
        dy = upstream
        return dy * (1. - ops.tanh(x)**2)  # type: ignore

    return ops.stop_gradient(r), grad
