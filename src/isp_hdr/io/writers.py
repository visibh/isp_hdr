"""
Output writing.
Output target is Ultra HDR JPEG.
"""

import subprocess

import cv2
import imagecodecs
import numpy as np
from imagecodecs import ULTRAHDR

from ..color.matrices import REC2020_LUMA, M_DG_to_AP1, apply3x3


def write_ultrahdr(
    rec2020: np.ndarray,
    output_path: str,
    nits: int = 1600,
    level: int = 95,
) -> None:
    """
    Encode display referred linear Rec.2020 as an Ultra HDR (gain-map) JPEG
    """
    h, w = rec2020.shape[:2]
    rgba_f16 = np.zeros((h, w, 4), dtype=np.float16)
    rgba_f16[..., :3] = rec2020.astype(np.float16)
    rgba_f16[..., 3] = 1.0

    uhdr_jpeg = imagecodecs.ultrahdr_encode(
        rgba_f16,
        level=level,
        gamut=ULTRAHDR.CG.BT_2100,
        transfer=ULTRAHDR.CT.LINEAR,
        nits=nits,
    )
    with open(output_path, "wb") as f:
        f.write(uhdr_jpeg)


def write_linear_exrs(dgamut_linear: np.ndarray) -> None:
    """
    Write D-Gamut linear and ACEScg AP1 EXR. Scene-referred, linear light)
    """
    print("Saving D-Gamut Linear EXR -> output_dgamut_linear.exr")
    cv2.imwrite("output_dgamut_linear.exr", dgamut_linear.astype(np.float32)[..., ::-1])

    print("Saving ACEScg AP1 EXR -> output_aces_cg.exr")
    aces_cg = apply3x3(M_DG_to_AP1, dgamut_linear)
    cv2.imwrite("output_aces_cg.exr", aces_cg.astype(np.float32)[..., ::-1])
