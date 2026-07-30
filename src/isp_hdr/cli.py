"""
CLI entrypoint
"""

import os
import sys

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from .pipeline import run_pipeline

_CONF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conf")

_USAGE = """\
isp-hdr  DNG -> Ultra HDR JPEG image signal processor (DJI X9)

Usage:
  isp-hdr input_dng=PATH [key=value ...]

Examples:
  isp-hdr input_dng=frames/379.DNG
  isp-hdr input_dng=in.DNG debayer=bilinear tonemap=power rescale=none

"""


def main() -> None:
    args = sys.argv[1:]
    if any(a in ("-h", "--help") for a in args):
        print(_USAGE)
        return

    overrides = [a for a in args if not a.startswith("-")]
    with initialize_config_dir(version_base=None, config_dir=_CONF_DIR):
        cfg = compose(config_name="config", overrides=overrides)

    if OmegaConf.is_missing(cfg, "input_dng"):
        print(_USAGE)
        raise SystemExit("Error: input_dng is required. e.g. input_dng=path/to/file.DNG")

    run_pipeline(cfg)


if __name__ == "__main__":
    main()
