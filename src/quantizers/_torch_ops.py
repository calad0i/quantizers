"""dynamo cannot trace class creation when the backend inited in compiled region."""

import torch


class Surrogate(torch.autograd.Function):
    """Returns `value`, with gradient `g * c` for each interleaved `x, c` input (`c=None` passes `g` through)."""

    @staticmethod
    def forward(ctx, value, *xcs):
        ctx.shapes = [x.shape for x in xcs[::2]]
        ctx.save_for_backward(*xcs[1::2])
        return value

    @staticmethod
    def backward(ctx, g):  # type: ignore
        grads = [None]
        for shape, c, need in zip(ctx.shapes, ctx.saved_tensors, ctx.needs_input_grad[1::2]):
            grads += [(g if c is None else g * c.to(g.dtype)).sum_to_size(shape) if need else None, None]
        return tuple(grads)
