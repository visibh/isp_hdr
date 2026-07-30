"""
Primitive HDR-aware adaptive unsharp mask (luma channel only).

This step sharpens BT.2100 luma and rescales RGB by the luma ratio.
It does not clip at 1.0, so highlight detail is preserved for the Ultra HDR gain map.
A noise-threshold mask suppresses sharpening of flat/noisy regions.
"""

import cv2
import numpy as np

from ..base import Stage
from ...color.matrices import REC2020_LUMA
from ...context import ISPContext


def perceptual_sharpen(
        rec2020: np.ndarray,
        sigma: float = 1.0,
        amount: float = 0.4,
        noise_threshold: float = 0.015,
) -> np.ndarray:
    lum = (rec2020 * REC2020_LUMA).sum(axis=-1).astype(np.float32)

    ksize = max(3, int(sigma * 4) | 1)
    blurred = cv2.GaussianBlur(lum, (ksize, ksize), sigma)
    hp = lum - blurred

    mask = (np.abs(hp) > noise_threshold).astype(np.float32)
    sharpened = lum + amount * mask * hp
    sharpened = np.maximum(sharpened, 0.0)

    scale = (sharpened / lum.clip(1e-6))[:, :, np.newaxis]
    return np.maximum(rec2020 * scale, 0.0)


class PerceptualSharpen(Stage):
    name = "sharpen"

    def __init__(self, sigma: float = 1.0, amount: float = 0.4, noise_threshold: float = 0.015):
        self.sigma = sigma
        self.amount = amount
        self.noise_threshold = noise_threshold

    def process(self, rec2020: np.ndarray, ctx: ISPContext) -> np.ndarray:
        return perceptual_sharpen(rec2020, self.sigma, self.amount, self.noise_threshold)
