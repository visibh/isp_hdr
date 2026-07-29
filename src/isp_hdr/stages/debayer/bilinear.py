"""
Bilinear demosaicing.

G at R/B : 4 axial neighbors -> sum // 4
R/B at G : 2 axial neighbots -> sum // 2
R/B at B/R : 4 diagonal neighbors -> sum // 4
"""
import numpy as np
from scipy.ndimage import correlate
from .base import BaseDebayer

def demosaic_bilinear(bayer: np.ndarray) -> np.ndarray:
    """
    Bilinear demosaicing
    """

    assert bayer.ndim == 2, "Input must be 2D Bayer Mosaic Image"
    H, W = bayer.shape
    assert H % 2 == 0 and W % 2 == 0

    bayer_i = bayer.astype(np.int32) # Integer math

    # Borders use mode='nearest'
    K_cross = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.int32) # 4 axial
    K_diag = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=np.int32) # 4 diagonal
    K_h = np.array([[0, 0, 0], [1, 0, 1], [0, 0, 0]], dtype=np.int32) # 2 horizontal
    K_v = np.array([[0, 1, 0], [0, 0, 0], [0, 1, 0]], dtype=np.int32) # 2 vertical

    G_cross = correlate(bayer_i, K_cross, mode="nearest")
    RB_diag = correlate(bayer_i, K_diag, mode="nearest")
    RB_h = correlate(bayer_i, K_h, mode="nearest")
    RB_v = correlate(bayer_i, K_v, mode="nearest")

    G_interp = G_cross // 4 # G at R or B
    RB_h_interp = RB_h // 2 # horizontal axial average
    RB_v_interp = RB_v // 2 # vertical axial average
    RB_d_interp = RB_diag // 4 # R at B, B at R

    r_m = np.zeros((H, W), dtype=bool); r_m[0::2, 0::2] = True # R
    gr_m = np.zeros((H, W), dtype=bool); gr_m[0::2, 1::2] = True # Gr
    gb_m = np.zeros((H, W), dtype=bool); gb_m[1::2, 0::2] = True # Gb
    b_m = np.zeros((H, W), dtype=bool); b_m[1::2, 1::2] = True # B

    R = np.where(r_m, bayer_i,
                    np.where(gr_m, RB_h_interp,
                            np.where(gb_m, RB_v_interp, RB_d_interp)))
    G = np.where(r_m | b_m, G_interp, bayer_i)
    B = np.where(b_m, bayer_i,
                    np.where(gb_m, RB_h_interp,
                            np.where(gr_m, RB_v_interp, RB_d_interp)))

    return np.stack([R, G, B], axis=-1).astype(np.float32)

class BilinearDebayer(BaseDebayer):
    name = "debayer:bilinear"

    def _demosaic(self, bayer: np.ndarray) -> np.ndarray:
        print("Debayer: Bilinear")
        return demosaic_bilinear(bayer)
