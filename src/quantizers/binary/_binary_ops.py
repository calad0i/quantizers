from keras import ops


@ops.custom_gradient
def _binary_quantize(x):
    r = ops.where(x > 0, 1.0, -1.0)

    def grad(*args, upstream=None):
        if upstream is None:
            (upstream,) = args
        dy = upstream
        return dy * (1.0 - ops.tanh(x) ** 2)  # type: ignore

    return ops.stop_gradient(r), grad


@ops.custom_gradient
def _ternary_quantize(x):
    r = ops.where(x > 0.5, 1.0, ops.where(x < -0.5, -1.0, 0.0))

    def grad(*args, upstream=None):
        if upstream is None:
            (upstream,) = args
        dy = upstream
        return dy * (1.0 - ops.tanh(x) ** 2)  # type: ignore

    return ops.stop_gradient(r), grad
