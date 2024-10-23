from .binary_ops_np import binary_quantize_np, ternary_quantize_np

try:
    from .binary_ops import binary_quantize, ternary_quantize
except ImportError:  # pragma: no cover
    binary_quantize = binary_quantize_np  # pragma: no cover
    ternary_quantize = ternary_quantize_np  # pragma: no cover

__all__ = ['binary_quantize', 'ternary_quantize', 'binary_quantize_np', 'ternary_quantize_np']
