"""
Polyphase Lanczos-3 rescaler + 16:9 center crop

For a rational downscale factor (scale), the anti-aliasing lowpass has
cutoff scale*pi, so the Lanczos-3 kernel is dilated by 1/scale for downscale and
kept unit-width for upscale.
The 2D rescale is separable (horizontal then
vertical). Weight normalisation per output pixel handles boundaries without
zero-padding leading to no edge darkening afterwards.
"""

import math

import numpy as np


def _sinc(x: np.ndarray) -> np.ndarray:
    """
    Normalised sinc.
    The 0/0 at x=0 is masked
    """
    pi_x = np.pi * x
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(np.abs(x) < 1e-8, 1.0, np.sin(pi_x) / pi_x)


def _lanczos3(x: np.ndarray) -> np.ndarray:
    """
    Lanczos-3 kernel: sinc(x)*sinc(x/3) for |x|<3 , else 0. 6 taps implementation.
    """
    return np.where(np.abs(x) < 3.0, _sinc(x) * _sinc(x / 3.0), 0.0)


def _resample_axis(src: np.ndarray, dst_len: int, axis: int) -> np.ndarray:
    """
    1D polyphase Lanczos-3 resampling along one axis of an N-D array
    """
    src_len = src.shape[axis]
    scale = dst_len / src_len

    kernel_scale = min(scale, 1.0)  # dilate for downscale, unit for upscale
    tap_radius = int(math.ceil(3.0 / kernel_scale))

    dst_indices = np.arange(dst_len, dtype=np.float64)
    # center-aligned (half-pixel) mapping: src_x = (dst_x + 0.5)/scale - 0.5
    src_centers = (dst_indices + 0.5) / scale - 0.5
    first_tap = (np.floor(src_centers) - tap_radius + 1).astype(np.int32)

    tap_count = 2 * tap_radius
    tap_offsets = np.arange(tap_count, dtype=np.int32)
    src_tap_indices = first_tap[:, np.newaxis] + tap_offsets[np.newaxis, :]

    x = (src_tap_indices - src_centers[:, np.newaxis]) * kernel_scale
    weights = _lanczos3(x).astype(np.float32)

    # Clamp source indices AFTER weight computation
    src_tap_clamped = np.clip(src_tap_indices, 0, src_len - 1)

    weight_sum = weights.sum(axis=1, keepdims=True).clip(1e-8)
    weights_normalised = weights / weight_sum

    src_moved = np.moveaxis(src.astype(np.float32), axis, 0)
    orig_shape = src_moved.shape
    src_2d = src_moved.reshape(src_len, -1)

    src_gathered = src_2d[src_tap_clamped]  # (dst_len, tap_count, rest)
    dst_2d = np.einsum("dt,dtv->dv", weights_normalised, src_gathered, optimize=True)

    dst_shape = list(orig_shape)
    dst_shape[0] = dst_len
    dst_moved = dst_2d.reshape(dst_shape)
    return np.moveaxis(dst_moved, 0, axis).astype(np.float32)


def polyphase_rescale(image: np.ndarray, dst_h: int, dst_w: int) -> np.ndarray:
    """
    Separable 2D Lanczos-3 rescale: horizontal then vertical
    Operations are performed in float32
    """
    assert image.ndim == 3, "polyphase_rescale expects (H, W, C) input"
    assert image.dtype == np.float32, "polyphase_rescale expects float32 input"

    intermediate = _resample_axis(image, dst_w, axis=1)  # (src_h, dst_w, C)
    return _resample_axis(intermediate, dst_h, axis=0)  # (dst_h, dst_w, C)


def center_crop_to_16x9(image: np.ndarray) -> np.ndarray:
    """
    Center-crop columns to 16:9 with no throwing away rows.
    Returns a view if cropped
    """
    H, W = image.shape[:2]
    crop_w = int(round(H * 16.0 / 9.0))

    if crop_w >= W:
        print(f"  [Crop] Source {W}x{H} is already <= 16:9 hence, no crop applied")
        return image

    left = (W - crop_w) // 2
    right = left + crop_w
    print(
        f" [Crop] {W}x{H} -> {crop_w}x{H} (columns [{left}:{right}], discarding {left}px each side)"
    )
    return image[:, left:right, :]
