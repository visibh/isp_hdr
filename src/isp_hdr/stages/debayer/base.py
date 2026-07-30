"""
Debayer slot base class
"""
import abc
import numpy as np

from ..base import Stage

from ...context import ISPContext


class BaseDebayer(Stage):
    name = "debayer"

    @abc.abstractmethod
    def _demosaic(self, bayer: np.ndarray) -> np.ndarray:
        """
        Interpolate (H,W) Bayer mosaic to (H,W,3) raw RGB
        """

    def process(self, bayer: np.ndarray, ctx: ISPContext) -> np.ndarray:
        rgb_raw = self._demosaic(bayer)
        return rgb_raw / ctx.meta.norm_scale[None, None, :]
