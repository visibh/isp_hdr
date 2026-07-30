"""
ISP pipeline.
Some of the stages are mandatory and hence are instantiated directly.
"""

import sys
import numpy as np

from .context import ISPContext
from .io import load_dng
from .io.writers import write_ultrahdr
from .stages.linearize import Linearize
from .stages.white_balance import WhiteBalance
from .stages.debayer.malvar import MalvarDebayer # Possible options: BilinearDebayer, MalvarDebayer
from .stages.ccm import CCM
from .stages.tonemap.power import PowerToneMap
from .stages.denoise_chroma import ChromaDenoise
from .stages.optical import OpticalCorrection
from .stages.rescale import Rescale
from .stages.sharpen import PerceptualSharpen


def render(dng_path: str) -> np.ndarray:
    print(f"Loading DNG: {dng_path}")
    bayer, meta = load_dng(dng_path)
    ctx = ISPContext(meta=meta)

    image = bayer
    for stage in [Linearize(), WhiteBalance(), MalvarDebayer(), ChromaDenoise(), OpticalCorrection(), CCM(), PowerToneMap(),Rescale(), PerceptualSharpen(),]:
        print(f"Running stage: {stage.name}")
        image = stage.process(image, ctx)

    return image


def run_pipeline(dng_path: str, output: str = "debug_dump.jpg") -> None:
    image = render(dng_path)

    write_ultrahdr(image, output)


if __name__ == "__main__":
    dng = sys.argv[1] if len(sys.argv) > 1 else "frames/379.DNG"
    run_pipeline(dng)
