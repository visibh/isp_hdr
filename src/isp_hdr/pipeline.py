"""
ISP pipeline.
Some of the stages are mandatory and hence are instantiated directly.
"""
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from .context import ISPContext
from .io import load_dng
from .io.writers import write_linear_exrs, write_ultrahdr
from .stages.base import Stage
from .stages.ccm import CCM
from .stages.linearize import Linearize
from .stages.white_balance import WhiteBalance

def _build(node) -> Stage | None:
    """
    Instantiate an swappable stage from its config node. None if disabled
    """
    if node is None:
        return None
    if OmegaConf.select(node, "_target_") is None:
        return None
    return instantiate(node)

def _run(stages: list[Stage | None], image, ctx: ISPContext):
    for st in stages:
        if st is None:
            continue
        image = st.process(image, ctx)
    return image

def render(cfg: DictConfig) -> np.ndarray:
    """
    Run the full ISP and return the display-referred Rec.2020 image
    """
    print(f"Loading DNG: {cfg.input_dng}")
    bayer, meta = load_dng(cfg.input_dng)
    ctx = ISPContext(meta=meta)

    # Sensor + camera RGB domain
    print("Stage group: ISP decode (sensor -> camera RGB)")
    image = _run([
        Linearize(),
        WhiteBalance(),
        _build(cfg.get("debayer")),
        _build(cfg.get("denoise_chroma")),
        _build(cfg.get("optical")),
        CCM(),
    ], bayer, ctx)

    # Scene-referred side-outputs. Tapped at the D-Gamut stage
    if cfg.save_exr:
        print("Side-output: D-Gamut linear + ACEScg EXR")
        write_linear_exrs(image)

    # Display domain. Rec.2020
    print("Stage group: tone map -> rescale -> sharpen. Display Rec.2020")
    image = _run([
        _build(cfg.get("tonemap")),
        _build(cfg.get("rescale")),
        _build(cfg.get("sharpen")),
    ], image, ctx)

    return image

def run_pipeline(cfg: DictConfig) -> None:
    """
    Render and encode the terminal Ultra HDR JPEG
    """
    image = render(cfg)
    write_ultrahdr(image, cfg.output, nits=cfg.encoder.nits, level=cfg.encoder.level)
