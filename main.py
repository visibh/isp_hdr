def main():
    import numpy as np
    from rich import print

    # Dummy imports for testing
    from isp_hdr.io import load_dng
    from isp_hdr.context import ISPContext
    from isp_hdr.stages.linearize import Linearize
    from isp_hdr.stages.white_balance import WhiteBalance
    from isp_hdr.stages.debayer.bilinear import BilinearDebayer

    DNG = "frames/379.DNG"
    bayer, meta = load_dng(DNG)

    print("Shape:", bayer.shape, "dtype:", bayer.dtype)

    print("Hello from isp-hdr-8k!")


if __name__ == "__main__":
    main()
