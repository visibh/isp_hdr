"""
Extract ISP relevant meta from DNG using exiftool
"""

import json
import subprocess
from dataclasses import dataclass

import numpy as np

_TAGS = [
    "-ColorMatrix2",
    "-AsShotNeutral",
    "-BaselineExposure",
    "-LinearizationTable",
    "-WhiteLevel",
    "-NoiseProfile",
]

@dataclass
class ExifMetadata:
    color_matrix2: np.ndarray # (3, 3) XYZ(D50) -> camera
    as_shot_neutral: np.ndarray # (3,)
    baseline_exposure_ev: float
    white_level: np.ndarray | None # (4,) or None -> use rawpy fallback
    noise_alpha: np.ndarray # (4,)
    noise_beta: np.ndarray # (4,)
    linearization_table: np.ndarray | None

def extract_exif(dng_path: str) -> ExifMetadata:
    """
    Run exiftool and parse the tags for the pipeline
    """
    result = subprocess.run(
        ["exiftool", "-j", *_TAGS, dng_path],
        capture_output=True, text=True, check=True,
    )
    tags = json.loads(result.stdout)[0]

    asn = np.array(
        [float(v) for v in tags.get("AsShotNeutral", "1 1 1").split()], dtype=np.float64
    )
    color_matrix2 = np.array(
        [float(v) for v in tags.get("ColorMatrix2", "").split()], dtype=np.float64
    ).reshape(3, 3)
    baseline_ev = float(tags.get("BaselineExposure", 0.0))

    # NoiseProfile: (alpha, beta) pairs per CFA plane
    noise_tag = tags.get("NoiseProfile", "")
    if noise_tag:
        nvals = [float(v) for v in str(noise_tag).split()]
        n_planes = len(nvals) // 2
        noise_alpha = np.zeros(4, dtype=np.float64)
        noise_beta = np.zeros(4, dtype=np.float64)
        for i in range(min(n_planes, 4)):
            noise_alpha[i] = nvals[2 * i]
            noise_beta[i] = nvals[2 * i + 1]
        if n_planes == 3:  # duplicate G plane for Gr/Gb
            noise_alpha[3] = noise_alpha[1]
            noise_beta[3] = noise_beta[1]
    else:
        noise_alpha = np.full(4, 3e-5, dtype=np.float64)
        noise_beta = np.full(4, 1e-6, dtype=np.float64)

    # WhiteLevel
    wl_tag = tags.get("WhiteLevel", "")
    if wl_tag:
        wl_vals = [float(v) for v in str(wl_tag).split()]
        white_level = np.array(
            wl_vals if len(wl_vals) == 4 else wl_vals * 4, dtype=np.float32
        )
    else:
        white_level = None

    lin_tag = tags.get("LinearizationTable", "")
    linearization_table = (
        np.array([float(v) for v in lin_tag.split()], dtype=np.float32) if lin_tag else None
    )

    return ExifMetadata(
        color_matrix2=color_matrix2,
        as_shot_neutral=asn,
        baseline_exposure_ev=baseline_ev,
        white_level=white_level,
        noise_alpha=noise_alpha,
        noise_beta=noise_beta,
        linearization_table=linearization_table,
    )
