"""
Debayering of Bayer pixels. RGGB --> RGB
"""
from .bilinear import BilinearDebayer
from .malvar import MalvarDebayer

__all__ = ["MalvarDebayer", "BilinearDebayer"]
