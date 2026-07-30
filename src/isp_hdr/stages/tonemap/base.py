import abc
import numpy as np
from ..base import Stage
from ...color.matrices import M_DG_to_XYZ, M_XYZ_to_Rec2020, apply3x3
from ...context import ISPContext
def aces_narkowicz(x:np.ndarray) -> np.ndarray:
    """
    ACES filmic S curve
    """
    x = np.maximum(x, 0.0)
    return (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)

class BaseToneMap(Stage):
    name = "tonemap"

    clip_hdr: bool = False

    @abc.abstractmethod
    def _tone(self, dgamut_linear: np.ndarray, ctx: ISPContext) -> np.ndarray:
        ...

    def process(self, dgamut_linear: np.ndarray, ctx: ISPContext) -> np.ndarray:
        dgamut_toned = self._tone(dgamut_linear, ctx)
        rec2020 = apply3x3(M_XYZ_to_Rec2020, apply3x3(M_DG_to_XYZ, dgamut_toned))
        hi = 65504.0 if self.clip_hdr else 1.0
        return np.clip(rec2020, 0.0, hi).astype(np.float32)
