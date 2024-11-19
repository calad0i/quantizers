from .binary_ops_np import binary_quantize_np, ternary_quantize_np

try:
    from .binary_ops import binary_quantize, ternary_quantize
except ImportError:
    binary_quantize = binary_quantize_np
    ternary_quantize = ternary_quantize_np

__all__ = ['binary_quantize', 'ternary_quantize', 'binary_quantize_np', 'ternary_quantize_np']
