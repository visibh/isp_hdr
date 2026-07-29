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

# DJI color matrices are defined in their whitepaper. Page 4.
# https://dl.djicdn.com/downloads/zenmuse+x7/20171010/D-Log_D-Gamut_Whitepaper.pdf
#
# Below matrix can also be computed from given primaries and D65 information. Page 4.
# DJI D Gamut Linear -> CIE XYZ (D65)
M_DG_to_XYZ = np.array(
    [
        [0.6482, 0.1940, 0.1082],
        [0.2830, 0.8132, -0.0962],
        [-0.0183, -0.0832, 1.1903],
    ]
)

# CIE XYZ (D65) -> D Gamut Linear
M_XYZ_to_DG = np.linalg.inv(M_DG_to_XYZ)

# Rec.2020 -> CIE XYZ (D65)
M_Rec2020_to_XYZ = np.array(
    [
        [0.636958, 0.144617, 0.168881],
        [0.2627, 0.6781, 0.0593],
        [0.0, 0.0281, 0.0610],
    ]
)

# CIE XYZ (D65) -> Rec.2020
M_XYZ_to_Rec2020 = np.linalg.inv(M_Rec2020_to_XYZ)

# DJI D-Gamut -> ACES AP1
M_DG_to_AP1 = np.array([
    [0.7201, 0.2268, 0.0531],
    [0.0152, 0.9634, 0.0214],
    [0.0003, 0.0460, 0.9537],
])

# Rec2020 luma weights
REC2020_LUMA = np.array([0.2627, 0.6780, 0.0593])


def apply3x3(matrix: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """
    Apply a 3x3 color matrix
    """
    h, w, _ = rgb.shape
    return (matrix @ rgb.reshape(-1, 3).T).T.reshape(h, w, 3)
