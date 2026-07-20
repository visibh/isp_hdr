import abc
import numpy as np

from ..context import ISPContext

class Stage(abc.ABC):
    """One ISP step. Stages are purely transformational steps. (image, context) -> image"""
    name: str = "stage"
    @abc.abstractmethod
    def process(self, image: np.ndarray, ctx: ISPContext) -> np.ndarray:
        ...
