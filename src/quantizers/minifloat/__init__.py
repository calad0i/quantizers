from .float_point_ops_np import float_decompose_np, float_quantize_np

try:
    from .float_point_ops import float_decompose, float_quantize
except ImportError:
    float_quantize = float_quantize_np
    float_decompose = float_decompose_np

__all__ = ['float_quantize', 'float_decompose', 'float_quantize_np', 'float_decompose_np']
