"""
CCM (Color Conversion Matrix)
"""
import numpy as np
from ..base import Stage
from ...color.matrices import M_Rec2020_to_XYZ, M_XYZ_to_DG, apply3x3
from ...context import ISPContext

class CCM(Stage):
    name = "ccm"

    def process(self, cam_rgb: np.ndarray, ctx: ISPContext) -> np.ndarray:
        m = ctx.meta
        xyz_lifted = apply3x3(m.cam_to_xyz, cam_rgb)*m.total_lift
        ctx.xyz_lifted = xyz_lifted

        dgamut_linear = apply3x3(M_XYZ_to_DG, xyz_lifted)
        return np.clip(dgamut_linear, 0.0, None)
