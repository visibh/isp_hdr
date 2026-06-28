"""
Color space matrices and per-camera exposure constant.
Currently, these values are DJI X9 specific and are intentionally hardcoded.
In the current form, this pipeline targets a single camera. For extension to other cameras, specific metadata needs to be extracted from the EXIF information.

Convention: matrices act on column vectors so a full image transform is:
    out = (M @ rgb.reshape(-1,3).T).T.reshape(H,W,3)
which 'apply_3x3' performs.
Inverses are always computed, never hardcoded
"""

import numpy as np

# Extra exposure lift as per DJI recommended workflow. Source: DJI documentation for X9 CinemaDNG
# https://dl.djicdn.com/downloads/inspire_3/Recommended_Workflow_for_Editing_CinemaDNG_Files_EN.pdf
DJI_EXTRA_EV = 1.4


def apply3x3(matrix: np.array, rgb: np.array) -> np.ndarray:
    """
    Apply a 3x3 color matrix
    """
    h, w, _ = rgb.shape
    return (matrix @ rgb.shape(-1, 3).T).T.reshape(h, w, 3)
