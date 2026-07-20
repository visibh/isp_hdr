"""
Pipeline context containing per-frame metadata and derived parameters that stages need beyond the image array itself
"""
from dataclasses import dataclass, field
import numpy as np
from .color.matrices import DJI_EXTRA_EV

@dataclass
class CameraMetaData:
    """Everything pipeline needs to know about the captured frame"""
    dng_path: str

    # From rawpy
    black_level: np.ndarray # (4,) per CFA channel
    cfa_pattern: np.ndarray # (2,2) raw_pattern indices
    color_desc: str # e.g. RGGB though more are possible as well
    camera_whitebalance: np.ndarray # (4,) as shot WB multipliers

    # From exiftool
    color_matrix2: np.ndarray
    as_shot_neutral: np.ndarray
    baseline_exposure_ev: float
    white_level: np.ndarray
    noise_alpha: np.ndarray
    noise_beta: np.ndarray
    linearization_table: np.ndarray

    # Derived
    @property
    def total_lift(self) -> float:
        """Scene linear exposure lift, as per DJI recommended workflow"""
        return 2.0 ** (self.baseline_exposure_ev + DJI_EXTRA_EV)

    @property
    def cam_to_xyz(self) -> np.ndarray:
        """Camera RGB to scene-linear"""
        r = self.color_desc.index('R')
        g = self.color_desc.index('G')
        b = self.color_desc.index('B')

        return np.array([
            self.white_level[r] - self.black_level[r],
            self.white_level[g] - self.black_level[g],
            self.white_level[b] - self.black_level[b],
        ], dtype=np.float32)

@dataclass
class ISPContext:
    """Mutable per run context"""
    meta: CameraMetaData

    backend: str = "numpy"

    # Handed forward between stages
    wb_gains: np.ndarray | None = None
    xyz_lifted: np.ndarray | None = None

    # For anything else
    extra: dict = field(default_factory=dict)
