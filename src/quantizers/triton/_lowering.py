"""Lowering of the generic quantizers into Triton kernels, for `quantizers.triton`.

Triton's tensor ops only work in code Triton compiles itself (they take its `_semantic`), and Triton has no autodiff. So
the quantizers run at kernel compile time on `Value`s: Triton tensors with the semantic, and a tape. A vjp replays the tape
backwards with JAX's transpose rules. float32 throughout.

Each op as its float32 arithmetic: division `div_rn`, `2.0 ** e` exp2 (exact for integral e), libdevice (numpy in the
interpreter) for rint, fmod, log2 and tanh."""

import functools
import inspect
import math
from typing import Any

import triton.language as tl
from triton.language.core import TRITON_BUILTIN

_LN2 = math.log(2.0)
_NAN_ALL = tl.PropagateNan.ALL


def builtin(fn):
    """`fn` as a Triton builtin: called at compile time with the compiler's semantic, or with the interpreter's under
    TRITON_INTERPRET=1, which calls builtins outside of `triton.language` without one."""

    @functools.wraps(fn)
    def wrapper(*args, _semantic=None, **kwargs):
        if _semantic is None:
            from triton import knobs

            if not knobs.runtime.interpret:
                raise ValueError(f'{fn.__name__} is a Triton device function: call it from a @triton.jit function')
            from triton.runtime.interpreter import interpreter_semantic

            _semantic = interpreter_semantic
        return fn(*args, _semantic=_semantic, **kwargs)

    setattr(wrapper, TRITON_BUILTIN, True)
    wrapper.signature = inspect.signature(fn)  # type: ignore
    return wrapper


def constexpr_value(v):
    return v.value if isinstance(v, tl.constexpr) else v


class Lowering:
    """The semantic of one lowering, and the order of its tape."""

    def __init__(self, sem):
        from triton import knobs

        self.sem = sem
        self.interp = bool(knobs.runtime.interpret)
        self.n = 0

    def tensor(self, v):
        """float32 tensor of a Value, a tensor or a python scalar (masks as 0 and 1)."""
        v = v.t if isinstance(v, Value) else constexpr_value(v)
        if isinstance(v, tl.tensor):
            return v if v.dtype == tl.float32 else tl.cast(v, tl.float32, _semantic=self.sem)
        return tl.full([], float(v), tl.float32, _semantic=self.sem)

    def where(self, c, a, b):
        return tl.where(c, a, b, _semantic=self.sem)

    def div(self, a, b):
        return tl.div_rn(self.tensor(a), self.tensor(b), _semantic=self.sem)

    def mul(self, a, b):
        return self.tensor(a).__mul__(self.tensor(b), _semantic=self.sem)

    def libdevice(self, name, *args):
        """libdevice's `name`, or numpy's in the interpreter, which has no libdevice."""
        args = tl.broadcast(*map(self.tensor, args), _semantic=self.sem) if len(args) > 1 else (self.tensor(args[0]),)
        if not self.interp:
            lib = self.sem.builder.module_map['triton.language.extra.libdevice']
            return getattr(lib, name)(*args, _semantic=self.sem)
        import numpy as np
        from triton.runtime.interpreter import TensorHandle

        with np.errstate(all='ignore'):
            data = getattr(np, name)(*(a.handle.data for a in args)).astype(np.float32)
        return tl.tensor(TensorHandle(data, tl.float32), args[0].type)

    def inputs(self, args):
        """float32 tensors of `args` broadcast together, and their shape ([] for scalars)."""
        ts = [self.tensor(a) for a in args]
        ref = ts[0]
        for t in ts[1:]:
            ref, _ = tl.broadcast(ref, t, _semantic=self.sem)
        if not isinstance(ref.type, tl.block_type):
            return ts, []
        shape = list(ref.shape)
        return [tl.broadcast_to(t, *shape, _semantic=self.sem) for t in ts], shape

    def output(self, v, shape):
        t = self.tensor(v)
        return tl.broadcast_to(t, *shape, _semantic=self.sem) if shape else t

    def vjp(self, y: 'Value', leaves: 'list[Value]', g, shape):
        """Cotangents of `leaves` for the cotangent `g` of `y`; zeros where unreached."""
        reached, stack = {}, [y]
        while stack:
            v = stack.pop()
            if id(v) not in reached:
                reached[id(v)] = v
                stack.extend(p for p, _ in v.parents)
        adj = {id(y): g}
        for v in sorted(reached.values(), key=lambda v: v.idx, reverse=True):  # reverse creation: children first
            ct = adj[id(v)]
            for p, rule in v.parents:
                c = rule(ct)
                old = adj.get(id(p))
                adj[id(p)] = c if old is None else old.__add__(c, _semantic=self.sem)
        zeros = lambda: tl.full(shape, 0.0, tl.float32, _semantic=self.sem)  # noqa: E731
        return [self.output(adj[id(v)], shape) if id(v) in adj else zeros() for v in leaves]


class Value:
    """A tensor in a lowering, with its tape: `parents` are (value, rule) pairs, a rule maps the cotangent of this value to
    its contribution to the parent's (JAX's transpose rule)."""

    __slots__ = ('t', 'lo', 'parents', 'idx', 'leaf')

    def __init__(self, t, lo: Lowering, parents=(), leaf=False):
        self.t = t
        self.lo = lo
        self.leaf = leaf
        # Only the parents a gradient reaches inputs through
        self.parents = tuple((p, r) for p, r in parents if isinstance(p, Value) and (p.leaf or p.parents))
        self.idx = lo.n
        lo.n += 1

    def __bool__(self):
        raise TypeError('a Triton value has no truth value: data-dependent python control flow cannot be lowered')

    def _new(self, t, *parents):
        return Value(t, self.lo, parents)

    def _neg(self, ct):
        return self.lo.tensor(ct).__neg__(_semantic=self.lo.sem)

    def _binary(self, op, a, b):
        return getattr(self.lo.tensor(a), op)(self.lo.tensor(b), _semantic=self.lo.sem)

    def __add__(self, other):
        return self._new(self._binary('__add__', self, other), (self, lambda ct: ct), (other, lambda ct: ct))

    def __radd__(self, other):
        return self._new(self._binary('__add__', other, self), (other, lambda ct: ct), (self, lambda ct: ct))

    def __sub__(self, other):
        return self._new(self._binary('__sub__', self, other), (self, lambda ct: ct), (other, self._neg))

    def __rsub__(self, other):
        return self._new(self._binary('__sub__', other, self), (other, lambda ct: ct), (self, self._neg))

    def __neg__(self):
        return self._new(self._neg(self), (self, self._neg))

    def _mul(self, a, b):
        lo = self.lo
        return self._new(lo.mul(a, b), (a, lambda ct: lo.mul(ct, b)), (b, lambda ct: lo.mul(ct, a)))

    def __mul__(self, other):
        return self._mul(self, other)

    def __rmul__(self, other):
        return self._mul(other, self)

    def _div(self, a, b):
        lo = self.lo
        rule_b = lambda ct: self._neg(lo.mul(lo.mul(ct, lo.div(1.0, lo.mul(b, b))), a))  # noqa: E731  -(ct * b^-2 * a)
        return self._new(lo.div(a, b), (a, lambda ct: lo.div(ct, b)), (b, rule_b))

    def __truediv__(self, other):
        return self._div(self, other)

    def __rtruediv__(self, other):
        return self._div(other, self)

    def __pow__(self, n):
        if n != 2:
            raise NotImplementedError(f'Value ** {n!r}: only ** 2 is lowered')
        lo = self.lo
        return self._new(lo.mul(self, self), (self, lambda ct: lo.mul(ct, lo.mul(2.0, self))))

    def __rpow__(self, base):
        if base != 2:
            raise NotImplementedError(f'{base!r} ** Value: only 2.0 ** e is lowered')
        lo = self.lo
        r = tl.exp2(self.t, _semantic=lo.sem)  # exact for integral e in the normal range
        return self._new(r, (self, lambda ct: lo.mul(ct, lo.mul(_LN2, r))))  # ct * (log(2) * ans)

    def __mod__(self, other):
        # jnp.remainder: fmod, shifted by the divisor where the signs differ. Gradient: d/dself 1, d/dother -floor(a / b)
        lo, sem = self.lo, self.lo.sem
        a, b = lo.tensor(self), lo.tensor(other)
        r = lo.libdevice('fmod', a, b)
        signs_differ = r.__lt__(0.0, _semantic=sem).__ne__(b.__lt__(0.0, _semantic=sem), _semantic=sem)
        shift = r.__ne__(0.0, _semantic=sem).__and__(signs_differ, _semantic=sem)
        rule = lambda ct: lo.mul(self._neg(ct), tl.floor(lo.div(a, b), _semantic=sem))  # noqa: E731
        return self._new(lo.where(shift, r.__add__(b, _semantic=sem), r), (self, lambda ct: ct), (other, rule))

    def _compare(self, op, other):
        return self._new(self._binary(op, self, other))

    def __lt__(self, other):
        return self._compare('__lt__', other)

    def __le__(self, other):
        return self._compare('__le__', other)

    def __gt__(self, other):
        return self._compare('__gt__', other)

    def __ge__(self, other):
        return self._compare('__ge__', other)

    def __eq__(self, other):  # type: ignore
        return self._compare('__eq__', other)

    def __ne__(self, other):  # type: ignore
        return self._compare('__ne__', other)

    __hash__ = None  # type: ignore

    def _logic(self, op, other):
        return self._new(getattr(self.t, op)(other.t if isinstance(other, Value) else other, _semantic=self.lo.sem))

    def __and__(self, other):
        return self._logic('__and__', other)

    def __or__(self, other):
        return self._logic('__or__', other)

    __rand__ = __and__
    __ror__ = __or__


# Backend ops (`quantizers._backends.TritonBackend`)


def const(v, like: Value) -> Value:
    return Value(like.lo.tensor(v), like.lo)


def floor(x: Value) -> Value:
    return Value(tl.floor(x.t, _semantic=x.lo.sem), x.lo)


def round_conv(x: Value) -> Value:
    return Value(x.lo.libdevice('rint', x), x.lo)


def clip(x: Value, min_value, max_value) -> Value:
    # minimum(maximum(x, min), max), NaN propagating
    lo, sem = x.lo, x.lo.sem
    t = tl.maximum(x.t, lo.tensor(min_value), propagate_nan=_NAN_ALL, _semantic=sem)
    return Value(tl.minimum(t, lo.tensor(max_value), propagate_nan=_NAN_ALL, _semantic=sem), lo)


def abs(x: Value) -> Value:  # noqa: A001
    lo, sem = x.lo, x.lo.sem
    nonneg = x.t.__ge__(0.0, _semantic=sem)
    return Value(tl.abs(x.t, _semantic=sem), lo, [(x, lambda ct: lo.where(nonneg, ct, x._neg(ct)))])  # select(x >= 0, g, -g)


def where(cond: Value, a, b) -> Value:
    lo, c = cond.lo, cond.t
    t = lo.where(c, lo.tensor(a), lo.tensor(b))
    return Value(t, lo, [(a, lambda ct: lo.where(c, ct, 0.0)), (b, lambda ct: lo.where(c, 0.0, ct))])


def log2(x: Value) -> Value:
    lo = x.lo
    return Value(lo.libdevice('log2', x), lo, [(x, lambda ct: lo.div(ct, lo.mul(x, _LN2)))])  # ct / (x * log(2))


def tanh(x: Value) -> Value:
    lo = x.lo
    sem = lo.sem
    y = lo.libdevice('tanh', x)
    rule = lambda ct: lo.mul(ct.__add__(lo.mul(ct, y), _semantic=sem), lo.tensor(1.0).__sub__(y, _semantic=sem))  # noqa: E731
    return Value(y, lo, [(x, rule)])  # (ct + ct * ans) * (1 - ans)


def stop_gradient(x: Value) -> Value:
    return Value(x.t, x.lo)


def surrogate(value: Value, xs, cs) -> Value:
    lo = value.lo
    rules = [(lambda ct: ct) if c is None else (lambda ct, c=lo.tensor(c): lo.mul(ct, c)) for c in cs]  # ct * c
    return Value(value.t, lo, list(zip(xs, rules)))


# Lowering of a generic quantizer call


def value(sem, fn, *args: Any):
    """`fn(*args)` on Triton tensors (or scalars), float32 of the broadcast shape; a tuple for a tuple result."""
    lo = Lowering(sem)
    ts, shape = lo.inputs(args)
    out = fn(*(Value(t, lo) for t in ts))
    if isinstance(out, tuple):
        return tuple(lo.output(o, shape) for o in out)
    return lo.output(out, shape)


def vjp(sem, fn, args, g):
    """Cotangents of every operand of `fn(*args)` for the cotangent `g` of its value, float32 of the broadcast shape."""
    lo = Lowering(sem)
    ts, shape = lo.inputs([*args, g])
    leaves = [Value(t, lo, leaf=True) for t in ts[:-1]]
    return tuple(lo.vjp(fn(*leaves), leaves, ts[-1], shape))
