"""
Combine rawpy (bayer data) and exiftool (metadata) into a single load
"""
import numpy as np
from ..context import CameraMetadata
from .metadata import extract_exif # For reading metadata using exiftool
from .raw_reader import read_raw # For reading DNG using rawpy



def load_dng(dng_path: str) -> tuple[np.ndarray, CameraMetadata]:
    """
    LOad the raw bayer mosaic and assemble the full CameraMetadata
    """
    raw = read_raw(dng_path)
    exif = extract_exif(dng_path)

    white_level = exif.white_level
    if white_level is None:
        white_level = np.full(4, raw.white_level_fallback, dtype=np.float32)

    meta = CameraMetadata(
        dng_path=dng_path,
        black_level=raw.black_level,
        cfa_pattern=raw.cfa_pattern,
        color_desc=raw.color_desc,
        camera_whitebalance=raw.camera_whitebalance,
        color_matrix2=exif.color_matrix2,
        as_shot_neutral=exif.as_shot_neutral,
        baseline_exposure_ev=exif.baseline_exposure_ev,
        white_level=white_level,
        noise_alpha=exif.noise_alpha,
        noise_beta=exif.noise_beta,
        linearization_table=exif.linearization_table,
    )
    return raw.bayer, meta
