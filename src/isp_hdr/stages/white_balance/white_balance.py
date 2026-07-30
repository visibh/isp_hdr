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

        wb_gains = m.camera_whitebalance / m.camera_whitebalance[g_cfa_idx]

        # Guard against non-positive gains -> replace with same color siblings
        # Because DJI report camera_whitebalance[3] i.e. Gb as 0.0
        for i in range(4):
            if wb_gains[i] <= 0.0:
                wb_gains[i] = wb_gains[color_desc.index(color_desc[i])]

        for r in range(2):
            for c in range(2):
                bayer[r::2, c::2] *= wb_gains[cfa[r,c]]

        ctx.wb_gains = wb_gains
        return bayer
