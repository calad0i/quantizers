"""
Backend dispatch adapted from `einops` (https://github.com/arogozhnikov/einops/), MIT license
Author: "Alex Rogozhnikov"

Dispatch goes by the module a tensor's type comes from, so it never imports a framework (but numpy, for python scalars).
"""

import math
import sys
from numbers import Number
from typing import Any

_backends: dict[str, 'AbstractBackend'] = {}
_type2name: dict[type, str] = {}


def get_backend(tensor) -> 'AbstractBackend':
    """Backend for the type of `tensor`. Python and numpy scalars go to the numpy backend."""
    return _backends[_backend_name(type(tensor))]


def _backend_name(_type: type) -> str:
    name = _type2name.get(_type)
    if name is None:
        name = _type2name[_type] = _backend_class(_type)().name
    return name


_backend_name._dynamo_marked_constant = True  # type: ignore


def _backend_class(_type: type) -> type['AbstractBackend']:
    for cls in _type.__mro__:
        if (backend_cls := _module_backend(cls)) is not None:
            return backend_cls
    if issubclass(_type, Number):
        return NumpyBackend
    raise TypeError(f'No backend found for tensor type {_type}')


def _module_backend(cls: type):
    # frameworks by root pkg, quantizers provided ones using own tensor types (e.g., quantizers.triton) by subpackage
    parts = cls.__module__.split('.', 2)
    return _module2backend.get('.'.join(parts[:2])) or _module2backend.get(parts[0])


class _Singleton(type):
    def __call__(cls):
        if cls.name not in _backends:  # type: ignore
            _backends[cls.name] = super().__call__()  # type: ignore
        return _backends[cls.name]  # type: ignore


class AbstractBackend(metaclass=_Singleton):
    """Native ops used by the quantizers. Arguments other than the first may be python scalars, except for `where`."""

    name: str
    differentiable: bool = True

    @classmethod
    def is_appropriate_type(cls, tensor) -> bool:
        """Only called with the framework imported, available as `sys.modules[cls.name]`."""
        ...

    def asarray(self, x, like=None) -> Any:
        """`x` if it is native to the backend, else `x` converted to a native tensor with the dtype and device of `like`."""
        return x if self.is_appropriate_type(x) else self._asarray(x, like)

    def _asarray(self, x, like) -> Any: ...

    def _floor(self, x): ...

    def _round_conv(self, x) -> Any:
        """Round to nearest, ties to even."""
        ...

    def _clip(self, x, min_value, max_value) -> Any: ...

    def abs(self, x) -> Any: ...

    def where(self, cond, x, y) -> Any:
        """`cond` must be a native tensor; `x` and `y` may be python scalars."""
        ...

    def log2(self, x) -> Any: ...

    def tanh(self, x) -> Any: ...

    def stop_gradient(self, x) -> Any: ...

    def surrogate(self, value, *grads):
        """`value`, with gradient `c` w.r.t. `x` for each `(x, c)` in `grads`; `c=None` passes the gradient through.
        No gradient flows through the computation of `value` or `c`. Python scalar `x` are ignored."""
        grads = [(x, c) for x, c in grads if not isinstance(x, Number)]
        return self._surrogate(value, [x for x, _ in grads], [c for _, c in grads])

    def _surrogate(self, value, xs, cs) -> Any: ...
    def passthrough(self, x, r):
        """Value of `r`, gradient passed through to `x`."""
        return self.surrogate(r, (x, None))

    def floor(self, x):
        """Floor, with gradient passthrough."""
        return self.passthrough(x, self._floor(x))

    def round_conv(self, x):
        """Round to nearest, ties to even, with gradient passthrough."""
        return self.passthrough(x, self._round_conv(x))

    def clip(self, x, min_value, max_value):
        """Clip with ties passing gradient to both x and the touched boundary. Undefined if min_value > max_value."""
        return self.surrogate(
            self._clip(x, min_value, max_value),
            (x, (x >= min_value) & (x <= max_value)),
            (min_value, x <= min_value),
            (max_value, x >= max_value),
        )

    def __repr__(self):
        return f'<{type(self).__name__}>'


class _NumpyLikeBackend(AbstractBackend):
    """Ops from a numpy-compatible namespace `self.np`, shared by numpy and jax."""

    np: Any

    def _asarray(self, x, like):
        return self.np.asarray(x, dtype=getattr(like, 'dtype', None))

    def _floor(self, x):
        return self.np.floor(x)

    def _round_conv(self, x):
        return self.np.round(x)

    def _clip(self, x, min_value, max_value):
        return self.np.clip(x, min_value, max_value)

    def abs(self, x):
        return self.np.abs(x)

    def where(self, cond, x, y):
        return self.np.where(cond, x, y)

    def log2(self, x):
        return self.np.log2(x)

    def tanh(self, x):
        return self.np.tanh(x)


class NumpyBackend(_NumpyLikeBackend):
    """No gradient support, so the gradient routing is skipped."""

    name = 'numpy'
    differentiable = False

    def __init__(self):
        import numpy

        self.np = numpy

    @classmethod
    def is_appropriate_type(cls, tensor):
        np = sys.modules['numpy']
        return isinstance(tensor, (np.ndarray, np.generic, Number))

    def stop_gradient(self, x):
        return x

    def surrogate(self, value, *grads):
        return value

    def clip(self, x, min_value, max_value):
        return self.np.clip(x, min_value, max_value)


class JaxBackend(_NumpyLikeBackend):
    name = 'jax'

    def __init__(self):
        import jax
        import jax.numpy as jnp
        from jax.custom_derivatives import SymbolicZero

        self.jax = jax
        self.np = jnp

        @jax.custom_jvp
        def surrogate(value, xs, cs):
            return value

        def surrogate_jvp(primals, tangents):
            value, xs, cs = primals
            out = jnp.zeros_like(value)
            for t, c in zip(tangents[1], cs):
                if isinstance(t, SymbolicZero):
                    continue
                out = out + (t if c is None else t * c)
            return value, jnp.broadcast_to(out, value.shape).astype(value.dtype)

        surrogate.defjvp(surrogate_jvp, symbolic_zeros=True)
        self._custom_surrogate = surrogate

    @classmethod
    def is_appropriate_type(cls, tensor):
        # Tracers are also jax.Array instances
        return isinstance(tensor, sys.modules['jax'].Array)

    def stop_gradient(self, x):
        return self.jax.lax.stop_gradient(x)

    def _surrogate(self, value, xs, cs):
        return self._custom_surrogate(value, tuple(xs), tuple(cs))


class TorchBackend(AbstractBackend):
    name = 'torch'

    def __init__(self):
        import torch

        from ._torch_ops import Surrogate

        self.torch = torch
        self._Surrogate = Surrogate

    @classmethod
    def is_appropriate_type(cls, tensor):
        return isinstance(tensor, sys.modules['torch'].Tensor)

    def _asarray(self, x, like):
        return self.torch.as_tensor(x, dtype=getattr(like, 'dtype', None), device=getattr(like, 'device', None))

    def _floor(self, x):
        return self.torch.floor(x)

    def _round_conv(self, x):
        return self.torch.round(x)

    def _clip(self, x, min_value, max_value):
        # torch.clamp does not take a tensor and a python scalar bound at once
        return self.torch.minimum(self.torch.maximum(x, self.asarray(min_value, x)), self.asarray(max_value, x))

    def abs(self, x):
        return self.torch.abs(x)

    def where(self, cond, x, y):
        return self.torch.where(cond, x, y)

    def log2(self, x):
        return self.torch.log2(x)

    def tanh(self, x):
        return self.torch.tanh(x)

    def stop_gradient(self, x):
        return x.detach()

    def _surrogate(self, value, xs, cs):
        # Nothing to route without grads; dynamo also fails to trace the Function when no input requires grad
        if not self.torch.is_grad_enabled() or not any(x.requires_grad for x in xs):
            return value.detach()
        # No gradient flows into value and the cs (backward returns None for them), so they go in detached
        # or compiled autograd will crash on None grad for inputs requiring it
        cs = [c if c is None else c.detach() for c in cs]
        return self._Surrogate.apply(value.detach(), *(v for xc in zip(xs, cs) for v in xc))


class TensorflowBackend(AbstractBackend):
    """Covers tf.Tensor, tf.Variable, and symbolic tensors inside tf.function."""

    name = 'tensorflow'

    def __init__(self):
        import tensorflow

        self.tf = tensorflow

    def _pair(self, x, y):
        # tf does not promote python scalars against tensors; take the dtype of the tensor operand
        tf = self.tf
        if not tf.is_tensor(x):
            x = tf.convert_to_tensor(x, dtype=y.dtype if tf.is_tensor(y) else None)
        if not tf.is_tensor(y):
            y = tf.convert_to_tensor(y, dtype=x.dtype)
        return x, y

    def _unbroadcast(self, g, x):
        tf = self.tf
        assert x.dtype.is_floating, f'Gradient of {type(x)} is not floating: {x.dtype}'
        axes, _ = tf.raw_ops.BroadcastGradientArgs(s0=tf.shape(x), s1=tf.shape(g))
        return tf.cast(tf.reshape(tf.reduce_sum(g, axis=axes), tf.shape(x)), x.dtype)

    @classmethod
    def is_appropriate_type(cls, tensor):
        return sys.modules['tensorflow'].is_tensor(tensor)

    def _asarray(self, x, like):
        return self.tf.convert_to_tensor(x, dtype=getattr(like, 'dtype', None))

    def _floor(self, x):
        return self.tf.floor(x)

    def _round_conv(self, x):
        return self.tf.round(x)

    def _clip(self, x, min_value, max_value):
        return self.tf.minimum(self.tf.maximum(*self._pair(x, min_value)), self._pair(x, max_value)[1])

    def abs(self, x):
        return self.tf.abs(x)

    def where(self, cond, x, y):
        return self.tf.where(cond, *self._pair(x, y))

    def log2(self, x):
        # No native log2 in tf
        return self.tf.math.log(x) * (1 / math.log(2))

    def tanh(self, x):
        return self.tf.tanh(x)

    def stop_gradient(self, x):
        return self.tf.stop_gradient(x)

    def _surrogate(self, value, xs, cs):
        tf = self.tf

        # value as an input, not a closure: the gradient would leak through a captured tensor
        @tf.custom_gradient
        def surrogate(value, *xs):
            def grad(g):
                return [None] + [self._unbroadcast(g if c is None else g * tf.cast(c, g.dtype), x) for x, c in zip(xs, cs)]

            return tf.identity(value), grad

        return surrogate(value, *xs)


class MLXBackend(AbstractBackend):
    # Check for mlx.core, as other mlx.<submodules> don't need it
    name = 'mlx.core'

    def __init__(self):
        import mlx.core as mx

        self.mx = mx

        def unbroadcast(g, shape):
            g = g.sum(axis=tuple(range(g.ndim - len(shape)))) if g.ndim > len(shape) else g
            axes = tuple(a for a, n in enumerate(shape) if n == 1 and g.shape[a] != 1)
            return g.sum(axis=axes, keepdims=True) if axes else g

        @mx.custom_function
        def surrogate(value, xs, cs):
            return value

        @surrogate.vjp
        def surrogate_vjp(primals, g, output):
            value, xs, cs = primals
            gxs = [unbroadcast(g if c is None else g * c.astype(g.dtype), x.shape) for x, c in zip(xs, cs)]
            return mx.zeros_like(value), gxs, [None if c is None else mx.zeros_like(c) for c in cs]

        self._custom_surrogate = surrogate

    @classmethod
    def is_appropriate_type(cls, tensor):
        return isinstance(tensor, sys.modules['mlx.core'].array)

    def _asarray(self, x, like):
        return self.mx.array(x, dtype=getattr(like, 'dtype', None))

    def _floor(self, x):
        return self.mx.floor(x)

    def _round_conv(self, x):
        return self.mx.round(x)

    def _clip(self, x, min_value, max_value):
        return self.mx.minimum(self.mx.maximum(x, min_value), max_value)

    def abs(self, x):
        return self.mx.abs(x)

    def where(self, cond, x, y):
        return self.mx.where(cond, x, y)

    def log2(self, x):
        return self.mx.log2(x)

    def tanh(self, x):
        return self.mx.tanh(x)

    def stop_gradient(self, x):
        return self.mx.stop_gradient(x)

    def _surrogate(self, value, xs, cs):
        return self._custom_surrogate(value, xs, cs)


class TinygradBackend(AbstractBackend):
    """No custom gradients in tinygrad: `surrogate` uses the stop_gradient fallback."""

    name = 'tinygrad'

    def __init__(self):
        from tinygrad import Tensor

        self.Tensor = Tensor

    @classmethod
    def is_appropriate_type(cls, tensor):
        return isinstance(tensor, sys.modules['tinygrad'].Tensor)

    def _asarray(self, x, like):
        return self.Tensor(x, dtype=getattr(like, 'dtype', None), device=getattr(like, 'device', None))

    def _floor(self, x):
        return x.floor()

    def _round_conv(self, x):
        return (x == x.trunc()).where(x, x.round())

    def _clip(self, x, min_value, max_value):
        return x.maximum(min_value).minimum(max_value)

    def abs(self, x):
        return x.abs()

    def where(self, cond, x, y):
        return cond.where(x, y)

    def log2(self, x):
        return x.log2()

    def tanh(self, x):
        return x.tanh()

    def stop_gradient(self, x):
        # adding kernel boundary keeping values exact in wrap
        return x.detach().contiguous()

    def _surrogate(self, value, xs, cs):
        r = self.stop_gradient(value)
        for x, c in zip(xs, cs):
            d = x - x.detach()
            r = r + (d if c is None else d * self.stop_gradient(c))
        return r


class TritonBackend(AbstractBackend):
    """Values in a Triton kernel being compiled (`quantizers.triton`), differentiated by the tape of their lowering."""

    name = 'triton'

    def __init__(self):
        from .triton import _lowering

        self.ops = _lowering

    @classmethod
    def is_appropriate_type(cls, tensor):
        return isinstance(tensor, sys.modules['quantizers.triton._lowering'].Value)

    def _asarray(self, x, like):
        return self.ops.const(x, like)

    def _floor(self, x):
        return self.ops.floor(x)

    def _round_conv(self, x):
        return self.ops.round_conv(x)

    def _clip(self, x, min_value, max_value):
        return self.ops.clip(x, min_value, max_value)

    def abs(self, x):
        return self.ops.abs(x)

    def where(self, cond, x, y):
        return self.ops.where(cond, x, y)

    def log2(self, x):
        return self.ops.log2(x)

    def tanh(self, x):
        return self.ops.tanh(x)

    def stop_gradient(self, x):
        return self.ops.stop_gradient(x)

    def _surrogate(self, value, xs, cs):
        return self.ops.surrogate(value, xs, cs)


class KerasBackend(AbstractBackend):
    """Thin shim for keras 3 Variables: unwraps them to `.value` and delegates to the backend keras runs on."""

    name = 'keras'

    def __init__(self):
        import keras

        self.keras = keras

        self.Variable = keras.Variable
        native_cls = {cls.name: cls for cls in (JaxBackend, TorchBackend, TensorflowBackend, NumpyBackend)}
        assert keras.backend.backend() in native_cls, f'Keras backend {keras.backend.backend()} is not supported'
        self.native = native_cls[keras.backend.backend()]()

    def _unwrap(self, x):
        return tuple(self._unwrap(v) for v in x) if isinstance(x, tuple) else (x.value if isinstance(x, self.Variable) else x)

    @property
    def differentiable(self):  # type: ignore
        return self.native.differentiable

    def __repr__(self):
        return f'<{type(self).__name__}: {self.native!r}>'


def _delegate(name: str):
    def method(self: KerasBackend, *args, **kwargs):
        args = [self._unwrap(a) for a in args]
        kwargs = {k: self._unwrap(v) for k, v in kwargs.items()}
        return getattr(self.native, name)(*args, **kwargs)

    method.__name__ = name
    return method


for _name, _attr in list(vars(AbstractBackend).items()):
    if callable(_attr) and not _name.startswith('_'):
        setattr(KerasBackend, _name, _delegate(_name))


# Keyed on the module, or its root package, of each class in the MRO: the first match wins, e.g. keras for keras Variables
# on tensorflow
_module2backend: dict[str, type[AbstractBackend]] = {
    'keras': KerasBackend,
    'jax': JaxBackend,
    'jaxlib': JaxBackend,
    'torch': TorchBackend,
    'tensorflow': TensorflowBackend,
    'mlx': MLXBackend,
    'tinygrad': TinygradBackend,
    'numpy': NumpyBackend,
    'quantizers.triton': TritonBackend,
}
