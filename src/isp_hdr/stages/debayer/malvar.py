"""
Malvar-He-Cutler (MHC) linear demosaic algorithm for RGGB Bayer.
Source: https://web.stanford.edu/class/ee367/reading/Demosaicing_ICASSP04.pdf
"""
import numpy as np

from .base import BaseDebayer

# Kernel definition from paper
_K_G = np.array([
    [0, 0, -1, 0, 0],
    [0, 0, 2, 0, 0],
    [-1, 2, 4, 2, -1],
    [0, 0, 2, 0, 0],
    [0, 0, -1, 0, 0],
], dtype=np.float64) / 8.0

_K_RGr = np.array([
    [0, 0, 0.5, 0, 0],
    [0, -1, 0, -1, 0],
    [-1, 4, 5, 4, -1],
    [0, -1, 0, -1, 0],
    [0, 0, 0.5, 0, 0],
], dtype=np.float64) / 8.0

_K_RGb = np.array([
    [0, 0, -1, 0, 0],
    [0, -1, 4, -1, 0],
    [0.5, 0, 5, 0, 0.5],
    [0, -1, 4, -1, 0],
    [0, 0, -1, 0, 0],
], dtype=np.float64) / 8.0

_K_RB = np.array([
    [0, 0, -1.5, 0, 0],
    [0, 2, 0, 2, 0],
    [-1.5, 0, 6, 0, -1.5],
    [0, 2, 0, 2, 0],
    [0, 0, -1.5, 0, 0],
], dtype=np.float64) / 8.0

def _conv5(pad: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Apply 5x5 kernel to the reflect-padded image
    """
    H = pad.shape[0] - 4
    W = pad.shape[1] - 4
    out = np.zeros((H, W), dtype=np.float64)
    for dy in range(5):
        for dx in range(5):
            w = kernel[dy, dx]
            if w != 0.0:
                out += w * pad[dy: dy + H, dx: dx + W]
    return out

def demosaic_malvar(bayer: np.ndarray) -> np.ndarray:
    """
    Linear demosaic using MHC 5x5 filters.
    """
    assert bayer.ndim == 2, "Input must be a 2D Bayer mosaic"
    H, W = bayer.shape
    assert H % 2 == 0 and W % 2 == 0, "H and W must be even (RGGB tiles)" # Handling easy case for now

    b64 = bayer.astype(np.float64)
    pad = np.pad(b64, 2, mode="reflect")  # 5x5 kernel never reads out of bounds

    G_interp = _conv5(pad, _K_G) # G at R and G at B positions
    R_at_Gr = _conv5(pad, _K_RGr) # R at Gr (even row, odd col)
    R_at_Gb = _conv5(pad, _K_RGb) # R at Gb (odd row, even col)
    R_at_B = _conv5(pad, _K_RB) # R at B  (odd row, odd col)
    B_at_Gb = _conv5(pad, _K_RGr) # B kernels are symmetric to R kernels
    B_at_Gr = _conv5(pad, _K_RGb)
    B_at_R = _conv5(pad, _K_RB)

    rows = np.arange(H, dtype=np.int32)[:, None]
    cols = np.arange(W, dtype=np.int32)[None, :]
    R_mask = (rows % 2 == 0) & (cols % 2 == 0)
    Gr_mask = (rows % 2 == 0) & (cols % 2 == 1)
    Gb_mask = (rows % 2 == 1) & (cols % 2 == 0)
    B_mask = (rows % 2 == 1) & (cols % 2 == 1)

    R = np.where(R_mask, b64,
                 np.where(Gr_mask, R_at_Gr,
                          np.where(Gb_mask, R_at_Gb, R_at_B)))
    G = np.where(R_mask | B_mask, G_interp, b64)
    B = np.where(B_mask, b64,
                 np.where(Gb_mask, B_at_Gb,
                          np.where(Gr_mask, B_at_Gr, B_at_R)))

    rgb = np.stack([R, G, B], axis=-1).astype(np.float32)
    return np.clip(rgb, 0.0, None)

class MalvarDebayer(BaseDebayer):
    name = "debayer:malvar"

    def _demosaic(self, bayer: np.ndarray) -> np.ndarray:
        return demosaic_malvar(bayer)
