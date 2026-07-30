"""
Optical correction stage to apply DNG OpcodeList3
Order of application: GainMap -> WarpRectilinear
"""

import numpy as np

from ..base import Stage
from ...context import ISPContext
from .gain_map import apply_gain_map
from .opcodes import parse_opcode_list3
from .warp import apply_warp_rectilinear


class OpticalCorrection(Stage):
    name = "optical corrections"

    def process(self, cam_rgb: np.ndarray, ctx: ISPContext) -> np.ndarray:
        for op in parse_opcode_list3(ctx.meta.dng_path):
            if op["type"] == "gain_map":
                cam_rgb = apply_gain_map(cam_rgb, op)
            elif op["type"] == "warp":
                cam_rgb = apply_warp_rectilinear(cam_rgb, op)
        return cam_rgb
