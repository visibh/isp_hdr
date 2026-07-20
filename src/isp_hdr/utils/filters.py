"""
Filter coefficients for various ISP stages
"""

import numpy as np


def gaussian_kernel_1D(sigma: float, radius: int) -> np.ndarray:
    """
    1D Gaussian kernel, normalized sum to 1
    """
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return (k / k.sum()).astype(np.float32)


def separable_gaussian(plane: np.ndarray, sigma: float) -> np.ndarray:
    """
    2D Gaussian blur via two separable 1D conv pass
    """
    return np.zeros(1)
    pass
