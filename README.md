This repo illustrates a plausible ISP pipeline consisting of several algorithms. The pipeline is capable of rendering HDR enabled JPEG output often referred as JPEG HDR. JPEG supports HDR by embedding SDR JPEG along with an embedded gain map (MPF, Multi Picture Format). The goal of this repo is to demonstrate such a modern ISP which supports JPEG HDR rendering.

## Prerequisites
- Python ≥ 3.14
- **exiftool** — required external binary for DNG metadata + OpcodeList3 parsing:
  On Mac: `brew install exiftool`

## Image Data
To develop this ISP, 16 bit DNG captured with DJI X9 (mounted on DJI Inspire 3) was used. DJI Inspire 3 is capable of recording Cinema DNG. The idea behind using the DNG was to capture the metadata required to render the Bayer mosaic to usable color output images. The resolution of sample DNG is 8192x4320.

This repo does not claims the ownership of the data and ownership remains with the original creators. Original material is 25 FPS and shot on 50 mm lens; It is located on the [DJI website](https://terra-1-g.djicdn.com/851d20f7b9f64838a34cd02351370894/630sample/nature031-Y507C0006_230216_0129.zip). The footage is 5GB in size.

For the sake of easiness, (1) Input DNG (`379.DNG`), (2) HDR JPEG (`DJI_X9_UltraHDR.jpg`) and (3) SDR JPEG (`DJI_X9_UltraHDR_SDR.jpg`) are committed to the repo but is almost never a good idea to commit large files in git.

## Algorithms
Following Algorithms have been chained in sequence:
1. Linearization
2. White balance
3. Debayering
4. Chroma denoising
5. Optical corrections
6. CCM
7. Tone mapping
8. Rescaling
9. Sharpening

Ordering of the algorithms can be altered in `pipeline.py`.

### Compute Backend
This repo is written purely in Python with numpy backend. In future, I have plans to expand it to use GPU acceleration but in the current form, it is CPU only.

### Skipped Algorithms
A commercial ISP contains several more stages however, I have intentionally skipped them owing to complexity involved in developing them. Furthermore, this repo does not contain state-of-the-art algorithms either.

The tone curve is deliberately minimal i.e. it is a chrominance-preserving power function that compresses scene luminance for display while leaving hue and saturation essentially unchanged. No creative color grading or look development has been applied therefore the output is a neutral, display-referred rendering, not a graded image.

## Example Runs
1. Render HDR JPEG in 16:9 format. So it crops the frame: 8192x4320 -> 7680x4320 
`uv run isp-hdr input_dng=frames/379.DNG`

2. Render HDR JPEG without 16:9 crop and selecting bilinear interpolation in place of MHC for debayering
`uv run isp-hdr input_dng=frames/379.DNG debayer=bilinear tonemap=power rescale=none`

### Output
Running above commands would produce a HDR JPEG `DJI_X9_UltraHDR.jpg`. 

## Viewing HDR Images
Decoding and viewing HDR JPEG on a consumer device is still a complicated topic and requires aligning several things including rendering application and display. Easiest option is to view the image with the default MacOS viewer (called, Preview) on a Retina XDR supported MacBook Pro (tested on M2 Max). Other alternative would be an iPhone to view the rendered image (tested on iPhone 16).

On a SDR display, HDR JPEG (`DJI_X9_UltraHDR.jpg`) and SDR JPEG (`DJI_X9_UltraHDR_SDR.jpg`) will look similar and it is by ensured by JPEG standard.

HDR brightness has been set to 1600 nits to match MBP display and it is configurable in `config.yaml`.

### Disclaimer
This repository contains a personal, independent project. All code, research, and opinions expressed here are entirely my own and do not represent the views, strategies, or official positions of my employer. No proprietary information, code, or assets from my employer have been used in the creation of this project.
