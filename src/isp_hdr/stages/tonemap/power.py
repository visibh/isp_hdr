"""
Power curve tone map with true black anchor.
Chrominance preserving power curve
"""
import numpy as np

from .base import BaseToneMap
from ...color.matrices import M_DG_to_XYZ
from ...context import ISPContext

class PowerToneMap(BaseToneMap):
    name = "tonemap:power"
    clip_hdr = True

    def __init__(self, gamma: float = 1.1, black_anchor_percentile: float = 0.5) -> None:
        self.gamma = gamma
        self.black_anchor_percentile = black_anchor_percentile

    def _tone(self, dgamut_linear: np.ndarray, ctx: ISPContext) -> np.ndarray:
        black_anchor = np.percentile(dgamut_linear, self.black_anchor_percentile)
        dgamut_anchored = np.maximum(dgamut_linear - black_anchor, 0)

        lum = (dgamut_anchored @ M_DG_to_XYZ[1]).clip(1e-8)[:, :, np.newaxis]
        lum_toned = np.power(lum, self.gamma)
        scale = lum_toned / lum
        return dgamut_anchored * scale
