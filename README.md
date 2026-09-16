# SCLSR: Remote Sensing Image Super Resolution Based on Self Calibration Learning

This is the source code of the paper named "Remote Sensing Image Super Resolution Based on Self Calibration Learning". This paper has been submitted to *IEEE Transactions on Geoscience and Remote Sensing*.

## Introduction

This repository contains the implementation of **SCLSR** (Remote Sensing Image Super Resolution Based on Self Calibration Learning), a remote-sensing image super-resolution method designed to recover spatial structures and high-frequency details from low-resolution imagery. It includes the SCLSR network definition and a PyQt5 desktop demo for image inference, visual comparison, PSNR/SSIM evaluation, residual visualization, and local-detail inspection.

## Features

- **Self-calibrating feature learning:** jointly models frequency-aware and spatial information.
- **Wave-based processing:** captures long-range dependencies with efficient frequency-domain operations.
- **Multi-scale feature extraction:** aggregates local details with depthwise convolutions at different receptive fields.
- **Interactive desktop demo:** supports image loading, super-resolution, result saving, PSNR/SSIM calculation, high-frequency residual display, and region zooming.

## Repository Structure

```text
SCLSR/
├── SCL.py              # SCLSR network architecture
├── run_scl_app.py      # Model loading and inference entry point
├── ui2.py              # PyQt5 graphical interface
├── requirements.txt    # Python dependencies
└── README.md
```

## System Requirements

The code is intended for Python 3.9 or later. A CUDA-enabled NVIDIA GPU is recommended, although the inference entry point automatically falls back to CPU when CUDA is unavailable.

```bash
conda create -n sclsr python=3.9 -y
conda activate sclsr
pip install -r requirements.txt
```

Install the PyTorch build appropriate for your CUDA version if the default package selected by `pip` is not suitable for your system. See the official PyTorch installation guide for platform-specific commands.

## Usage

### 1. Setup

Clone this repository and install the required packages as described above.

### 2. Model Preparation

Place the x4 checkpoint in the repository root with the following filename:

```text
SCL_x4.pth
```

The checkpoint is not included in the source archive used to prepare this repository. The demo will display an error if `SCL_x4.pth` is missing.

### 3. Running the Demo

```bash
python run_scl_app.py
```

After the application opens:

1. Select a JPG, PNG, BMP, or TIF image.
2. Wait for SCLSR inference to complete.
3. Save the super-resolved image or inspect PSNR, SSIM, high-frequency residuals, and local regions.

## Notes

- The current demo is configured for x4 super-resolution.
- PSNR and SSIM are evaluated on the luminance channel.
- Large remote-sensing images may require substantial GPU memory.
- Training scripts and pretrained weights are not part of the supplied source package.

## Citation

The manuscript is currently under review at *IEEE Transactions on Geoscience and Remote Sensing*. Citation information will be updated after the paper becomes publicly available.

## Contact

If you have any questions, please contact us at [jianghe@cumt.edu.cn](mailto:jianghe@cumt.edu.cn).

## License

No open-source license has been specified for this release. All rights are reserved by the copyright holder unless otherwise stated.
