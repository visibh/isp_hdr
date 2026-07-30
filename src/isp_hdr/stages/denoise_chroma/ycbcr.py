"""
Chroma noise reduction in YCbCr decomposed camera RGB space. Primitive chroma filtering.

HUman vision is more sensitive to Luma than Chrominance components so we
can perform a denoising (i.e. Gaussian blur) on chrominance components without perceptible softening.
"""
import cv2
import numpy as np

from ..base import Stage
from ...color.matrices import apply3x3
from ...context import ISPContext

_M_TO = np.array([
    [0.2989,  0.5866,  0.1145], # Y
    [-0.1687, -0.3313,  0.5000], # Cb
    [0.5000, -0.4187, -0.0813], # Cr
])
_M_FROM = np.linalg.inv(_M_TO)

def ycbcr_chroma_nr(cam_rgb: np.ndarray, sigma_chroma: float = 1.5) -> np.ndarray:
    ycbcr = apply3x3(_M_TO, cam_rgb)

    ksize = max(3, int(sigma_chroma * 4) | 1)
    for ch in (1, 2): # Cb and Cr only
        ycbcr[:, :, ch] = cv2.GaussianBlur(
            ycbcr[:, :, ch].astype(np.float32), (ksize, ksize), sigma_chroma
        )

    return apply3x3(_M_FROM, ycbcr)

class ChromaDenoise(Stage):
    name = "denoise_chroma"

    def __init__(self, sigma_chroma: float = 1.5):
        self.sigma_chroma = sigma_chroma

    def process(self, cam_rgb: np.ndarray, ctx: ISPContext) -> np.ndarray:
        return ycbcr_chroma_nr(cam_rgb, self.sigma_chroma)
