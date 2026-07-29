"""
White balance operation. It is a compulsory step.

Normalize the camera as-shot WB multipliers.
"""

import numpy as np
from ..base import Stage
from ...context import ISPContext


class WhiteBalance(Stage):
    name = "white_balance"

    def process(self, bayer: np.ndarray, ctx: ISPContext) -> np.ndarray:
        m = ctx.meta
        cfa, color_desc = m.cfa_pattern, m.color_desc

        g_cfa_idx = color_desc.index('G')
        wb_gains = m.camera_whitebalance / [g_cfa_idx]

        for r in range(2):
            for c in range(2):
                bayer[r::2, c::2] *= wb_gains[cfa[r,c]]

        ctx.wb_gains = wb_gains
        return bayer
