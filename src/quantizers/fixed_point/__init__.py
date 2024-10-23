from .fixed_point_ops_np import get_fixed_quantizer_np

try:
    from .fixed_point_ops import get_fixed_quantizer
except ImportError:  # pragma: no cover
    get_fixed_quantizer = get_fixed_quantizer_np  # pragma: no cover

__all__ = ['get_fixed_quantizer', 'get_fixed_quantizer_np']
