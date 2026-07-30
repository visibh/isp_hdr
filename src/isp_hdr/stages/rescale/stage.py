"""
Rescale stage

This stage runs after tone mapping (display-referred linear Rec.2020) and before sharpening,
so sharpen sigma is tuned to the output pixel density.
"""

import numpy as np

from ..base import Stage
from ...context import ISPContext
from .lanczos import center_crop_to_16x9, polyphase_rescale


class Rescale(Stage):
    name = "rescale"

    def __init__(self, dst_h: int = 1080, dst_w: int = 1920):
        self.dst_h = dst_h
        self.dst_w = dst_w

    def process(self, rec2020: np.ndarray, ctx: ISPContext) -> np.ndarray:
        cropped = center_crop_to_16x9(rec2020.astype(np.float32))
        out = polyphase_rescale(cropped, dst_h=self.dst_h, dst_w=self.dst_w)
        return out
