"""
Debayering of Bayer pixels. RGGB --> RGB
"""
from .bilinear import BilinearDebayer

__all__ = ["BilinearDebayer"]
