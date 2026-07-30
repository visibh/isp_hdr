"""
Apply a DNG GainMap opcode for lens vignetting correction
"""
import numpy as np

def apply_gain_map(rgb: np.ndarray, op: dict) -> np.ndarray:
    H, W = rgb.shape[:2]
    gains = op["gains"]
    grid_V = op["map_pts_v"]
    grid_H = op["map_pts_h"]
    map_planes = op["map_planes"]
    top, left = op["top"], op["left"]
    bottom, right = op["bottom"], op["right"]
    spacing_v = op["spacing_v"]
    spacing_h = op["spacing_h"]
    origin_v = op["origin_v"]
    origin_h = op["origin_h"]

    h_active = float(bottom - top) if bottom > top else float(H)
    w_active = float(right - left) if right > left else float(W)

    rows = np.arange(H, dtype=np.float64)
    cols = np.arange(W, dtype=np.float64)
    vy = np.clip((rows - top) / h_active, 0.0, 1.0)
    vx = np.clip((cols - left) / w_active, 0.0, 1.0)
    gy = np.clip(((vy - origin_v) / spacing_v).astype(np.float32), 0.0, grid_V - 1)
    gx = np.clip(((vx - origin_h) / spacing_h).astype(np.float32), 0.0, grid_H - 1)

    iy = np.floor(gy).astype(np.int32).clip(0, grid_V - 2)
    ix = np.floor(gx).astype(np.int32).clip(0, grid_H - 2)
    fy = (gy - iy).astype(np.float32)[:, np.newaxis]
    fx = (gx - ix).astype(np.float32)[np.newaxis, :]

    def _interp(g: np.ndarray) -> np.ndarray:
        return (
            g[iy, :][:, ix] * (1.0 - fx) * (1.0 - fy)
            + g[iy, :][:, ix + 1] * fx * (1.0 - fy)
            + g[iy + 1, :][:, ix] * (1.0 - fx) * fy
            + g[iy + 1, :][:, ix + 1] * fx * fy
        )

    if map_planes == 1:
        return rgb * _interp(gains[:, :, 0])[:, :, np.newaxis]

    corrected = np.empty_like(rgb)
    for c in range(3):
        corrected[:, :, c] = rgb[:, :, c] * _interp(gains[:, :, c])
    return corrected
