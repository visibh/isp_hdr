"""
Global ACES Narkowicz filmic tone map
"""

import numpy as np

from .base import BaseToneMap, aces_narkowicz
from ...color.matrices import M_DG_to_XYZ
from ...context import ISPContext


class AcesToneMap(BaseToneMap):
    name = "tonemap:aces"

    def __init__(self, perceptual_ev: float = 0.47):
        self.perceptual_ev = perceptual_ev

    def _tone(self, dgamut_linear: np.ndarray, ctx: ISPContext) -> np.ndarray:
        xyz_y = M_DG_to_XYZ[1]
        lum = (dgamut_linear @ xyz_y).clip(1e-8)
        p99 = np.percentile(lum, 99)
        scene = dgamut_linear / p99 * (2.0 ** self.perceptual_ev)

        lum_scene = (np.maximum(scene, 0) @ xyz_y).clip(1e-8)[:, :, np.newaxis]
        lum_toned = aces_narkowicz(lum_scene)
        scale = np.where(lum_scene > 1e-6, lum_toned / lum_scene, 0.0)
        return np.maximum(scene, 0) * scale
