"""
Debayer slot base class
"""
import abc
import numpy as np
from numpy.core.numeric import ndarray
from ..base import Stage

from ...context import ISPContext
from ... import ops

class BaseDebayer(Stage):
    name = "debayer"

    @abc.abstractmethod
    def _demosaic(self, bayer: np.ndarray) -> np.ndarray:
        """
        Interpolate (H,W) Bayer mosaic to (H,W,3) raw RGB
        """

    def process(self, bayer: np.ndarray, ctx: ISPContext) -> np.ndarray:
        rgb_raw = self._demosaic(bayer)
        norm_scale = ops.asarray_like(ctx.meta.norm_scale, rgb_raw)
        return rgb_raw / norm_scale[None, None, :]
