"""
Unprocessed bayer data reading from DNG using rawpy
"""

from dataclasses import dataclass

import numpy as np
import rawpy


@dataclass
class RawCapture:
    """
    Sensor data from DNG without any processing applied on it
    """

    bayer: np.ndarray  # (H, W) float32
    black_level: np.ndarray
    cfa_pattern: np.ndarray
    color_desc: str  # e.g. "RGGB" i.e. CFA pattern
    camera_whitebalance: np.ndarray  # (4,)
    white_level_fallback: float  # raw.white_level; used if no WhiteLvel tag is found


def read_raw(dng_path: str) -> RawCapture:
    """
    Load raw bayer mosaic mosaic and rawpy derived sensor metadata
    """
    with rawpy.imread(dng_path) as raw:
        return RawCapture(
                bayer=raw.raw_image_visible.astype(np.float32),
                black_level=np.array(raw.black_level_per_channel, dtype=np.float32),
                cfa_pattern=raw.raw_pattern,
                color_desc=raw.color_desc.decode(),
                camera_whitebalance=np.array(
                    raw.camera_white_level_per_channel, dtype=np.float64
                ),
                white_level_fallback=float(raw.white_level),
            )
