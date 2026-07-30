"""
Apply DNG WarpRectilinear opcode for lens distortion correction
"""
import numpy as np

def apply_warp_rectilinear(rgb: np.ndarray, op: dict) -> np.ndarray:
    H, W = rgb.shape[:2]
    planes = op["planes"]
    cx_n = op["cx"]
    cy_n = op["cy"]
    m_diag = (W ** 2 + H ** 2) ** 0.5

    xg = np.arange(W, dtype=np.float64)
    yg = np.arange(H, dtype=np.float64)
    xg, yg = np.meshgrid(xg, yg)

    xn = (xg - cx_n * W) / m_diag
    yn = (yg - cy_n * H) / m_diag
    r2 = xn * xn + yn * yn
    r4, r6 = r2 * r2, r2 * r2 * r2

    corrected = np.empty_like(rgb)
    for c, k in enumerate(planes):
        k0, k1, k2, k3, k4, k5 = k

        kr = k0 + k1 * r2 + k2 * r4 + k3 * r6
        dx = k4 * (2.0 * xn * yn) + k5 * (r2 + 2.0 * xn * xn)
        dy = k4 * (r2 + 2.0 * yn * yn) + k5 * (2.0 * xn * yn)
        src_x = cx_n * W + (xn * kr + dx) * m_diag
        src_y = cy_n * H + (yn * kr + dy) * m_diag

        # Important: We compute sub-pixel fractions from the UNCLAMPED floor, then
        # clamp integer indices.
        # Clamping before computing fx/fy lets out-of-bounds
        # taps produce fx >> 1 leading to corrupted bilinear lerp causing sphere/scratch artifacts.
        ix = np.floor(src_x).astype(np.int32)
        iy = np.floor(src_y).astype(np.int32)
        fx = (src_x - ix).astype(np.float32)  # always in [0, 1)
        fy = (src_y - iy).astype(np.float32)
        ix = ix.clip(0, W - 2)
        iy = iy.clip(0, H - 2)

        ch = rgb[:, :, c]
        corrected[:, :, c] = (
            ch[iy, ix] * (1.0 - fx) * (1.0 - fy)
            + ch[iy, ix + 1] * fx * (1.0 - fy)
            + ch[iy + 1, ix] * (1.0 - fx) * fy
            + ch[iy + 1, ix + 1] * fx * fy
        )
    return corrected
