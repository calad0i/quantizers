from .fixed_point_ops_np import get_fixed_quantizer_np

try:
    from .fixed_point_ops import get_fixed_quantizer
except ImportError:
    get_fixed_quantizer = get_fixed_quantizer_np

__all__ = ['get_fixed_quantizer', 'get_fixed_quantizer_np']
