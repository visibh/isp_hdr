"""
Linearization + Black level subtraction
"""

import numpy as np

from ...context import ISPContext
from ..base import Stage


class Linearize(Stage):
    name = "linearize"

    def process(self, image: np.ndarray, ctx: ISPContext) -> np.ndarray:
        meta = ctx.meta
        lut = meta.linearization_table

        if lut is not None:
            image = lut[image.astype(np.int32).clip(0, len(lut) - 1)]

        cfa, bl = meta.cfa_pattern, meta.black_level
        for r in range(2):
            for c in range(2):
                image[r::2, c::2] -= bl[cfa[r, c]]
        return np.clip(image, 0.0, None)
