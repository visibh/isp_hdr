def main():
    import numpy as np

    # Dummy imports for testing
    from isp_hdr.color.matrices import M_DG_to_XYZ, M_XYZ_to_DG
    from isp_hdr.utils.filters import separable_gaussian
    from isp_hdr.context import CameraMetadata
    from isp_hdr.io.raw_reader import read_raw
    r = read_raw("frames/379.DNG")
    print(r.bayer.shape, r.bayer.dtype, r.color_desc, r.black_level)

    print("Hello from isp-hdr-8k!")


if __name__ == "__main__":
    main()
