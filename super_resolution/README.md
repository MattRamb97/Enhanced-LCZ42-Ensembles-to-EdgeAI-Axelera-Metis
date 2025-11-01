# Super-Resolution Pipelines

Each subfolder corresponds to a different SR model used to enhance Sentinel-2 imagery.

| Model | Scale | Type | Framework | Folder | Output File |
|:------|:------|:------|:-----------|:-----------|:--------------------------|
| **VDSR** | ×2 / ×3 | CNN (shallow) | MATLAB | `VDSR/` | `training_vdsr2x.h5` |
| **EDSR** | ×2 / ×4 | CNN (deep) | PyTorch | `EDSR/` | `training_edsr2x.h5` |
| **ESRGAN** | ×2 | GAN (triplet semantic bands) | MATLAB | `ESRGAN/` | `training_esrgan2x.h5` |
| **SwinIR** | ×2 | Transformer | PyTorch | `SwinIR/` | `training_swinir2x.h5` |
| **BSRNet / BSRGAN** | ×2 | Blind SR (RRDBNet) | PyTorch | `BSRNet/` | `training_bsrnet2x.h5` |
| **Real-ESRGAN** | ×4 | GAN (real-world degradation) | PyTorch | `Real-ESRGAN/` | `training_realesrgan4x.h5` |

All SR-enhanced datasets maintain:
- `/sen2`: super-resolved multispectral Sentinel-2 data `[0, 2.8]`, `float32`
- `/label`: one-hot LCZ labels (17 classes)
- Identical sample order and label alignment with the baseline `training.h5` / `testing.h5`

## VDSR

Very Deep Super-Resolution network for upsampling LCZ42 imagery.

### trainedVDSRNet_v2.mat
- Pretrained VDSR network from MathWorks
- Trained on IAPR TC-12 Benchmark (natural images)
- Supports 2x, 3x, 4x scale factors
- Operates on single-channel (grayscale/luminance) images

### apply_vdsr_all_bands.m

Applies VDSR to all 10 Sentinel-2 bands independently.

**Input:** 
- X: [32×32×10] patch in [0,255] range
- vdsrNet: Loaded VDSR model
- scaleFactor: 2, 3, or 4

**Output:** 

- X_sr: [64×64×10] or [96×96×10] or [128×128×10] patch

**Usage:**
```matlab
X_sr = apply_vdsr_all_bands(X, vdsrNet, scaleFactor)
```

### precompute_vdsr_dataset.m

Batch processes entire HDF5 file with VDSR.

**Input:** 
- inputH5: Path to training.h5 or testing.h5
- outputH5: Output path (e.g., training_vdsr2x.h5)
- vdsrNet: Loaded VDSR model
- scaleFactor: 2, 3, or 4

**Output:**

New HDF5 file with upsampled /sen2 dataset, labels copied unchanged.

**Usage:**
```matlab
load('trainedVDSRNet_v2.mat');
precompute_vdsr_dataset('training.h5', 'training_vdsr2x.h5', net, 2);
```

## EDSR

Enhanced Deep Super-Resolution (EDSR) network for upsampling LCZ42 imagery.

### Pretrained Model
- **Source:** [eugenesiow/edsr-base](https://huggingface.co/eugenesiow/edsr-base)
- **Framework:** PyTorch (`super-image` library)
- **Scale factors:** ×2, ×3, ×4
- **Type:** CNN-based architecture for single-image super-resolution
- **Operation:** Applied band-by-band (each Sentinel-2 channel independently)

### apply_edsr_sr.py

Applies EDSR to all 10 Sentinel-2 bands independently and saves the upsampled dataset to HDF5.

**Input:**  
- `training.h5` or `testing.h5` (So2Sat LCZ42 format)  
- `scaleFactor`: 2, 3, or 4  
- Pretrained EDSR model (downloaded automatically from Hugging Face)

**Output:**
Upscaled dataset (e.g., `training_edsr2x.h5` or `testing_edsr2x.h5`). The script creates a new HDF5 file with the following datasets:
- /sen2: super-resolved Sentinel-2 data in [0, 2.8], float32
- /label: original one-hot encoded LCZ labels (17 classes)

**Usage:**
```bash
python apply_edsr_sr.py
```

## ESRGAN

Enhanced Super-Resolution Generative Adversarial Network (ESRGAN) for upsampling LCZ42 imagery with semantic triplet grouping.

### trained/ESRGAN100_RGB_Flickr2K_VGG54_2x_Generator_params_epoch300.mat
- Pretrained ESRGAN generator network (`dlnG`)  
- Trained on **Flickr2K** RGB dataset with VGG54 perceptual loss  
- Supports **×2 upscaling**  
- Implemented in **MATLAB Deep Learning Toolbox**  
- Operates on **3-channel RGB** inputs (applied to Sentinel-2 via spectral triplets)

### apply_esrgan_all_bands.m

Applies ESRGAN super-resolution to all 10 Sentinel-2 bands using **semantic triplets** that group spectrally similar channels.

**Input:**  
- `X`: `[32×32×10]` multispectral patch in `[0,255]` range  
- `scaleFactor`: Upscaling factor (×2)  

**Output:**  
- `X_sr`: `[64×64×10]` super-resolved patch  

**Semantic triplet configuration:**
| Triplet | Bands | Semantic Meaning |
|:--------|:-------|:----------------|
| [3,2,1] | B4, B3, B2 | RGB (True Color) |
| [7,5,4] | B8, B6, B5 | Vegetation / Red-Edge |
| [9,10,8] | B11, B12, B8A | Moisture / SWIR-NIR |
| [10,10,6] | B12, B12, B7 | Filler (Spectral Consistency) |

**Usage:**
```matlab
X_sr = apply_esrgan_all_bands(X, 2);
```

Each triplet is normalized per channel to [0,1], processed by ESRGAN and recombined into the full 10-band cube.

### ESRGAN_2xSuperResolution.m

Performs ESRGAN inference on a 3-channel input image using a persistent generator network.

Key Features:
- Loads the pretrained generator once (persistent dlnG)
- Automatically moves the model to GPU if available
- Normalizes inputs to [0,1], rescales outputs to [0,1]
- Returns upscaled image clipped to valid range

**Usage:**

```matlab
I_sr = ESRGAN_2xSuperResolution(I_lr);
```

### precompute_esrgan_dataset.m

Batch processes an entire HDF5 dataset with ESRGAN to generate enhanced Sentinel-2 imagery.

**Input:**

- inputH5: Path to training.h5 or testing.h5
- outputH5: Output file path (e.g., training_esrgan2x.h5)
- scaleFactor: Upscaling factor (2)

**Output:**

- New .h5 file with super-resolved /sen2 dataset
- Original /label copied unchanged

**Usage:**

```matlab
precompute_esrgan_dataset('training.h5', 'training_esrgan2x.h5', 2);
```

## SwinIR

SwinIR — Image restoration using Swin Transformer for Sentinel-2 super-resolution.

### Pretrained Model
- **Source:** [JingyunLiang/SwinIR](https://github.com/JingyunLiang/SwinIR)
- **Paper:** Liang et al., *“SwinIR: Image Restoration Using Swin Transformer,”* ICCV 2021
- **Framework:** PyTorch
- **Scale factor:** ×2 (classical SR)
- **Architecture:** Hierarchical Transformer using shifted windows
- **Operation:** Each Sentinel-2 band is processed independently (replicated to RGB triplets)

### swinir_model.py

Defines and loads a pretrained SwinIR model for classical super-resolution.

**Function:**  

`define_model(scale=2, device="cuda")`

**Description:**

- Loads the official pretrained SwinIR-M ×2 model  
- Automatically downloads weights from GitHub  
- Configured for 3-channel RGB input (per-band replication)

**Usage:**

```python
from swinir_model import define_model
model = define_model(scale=2, device="cuda")
```

### apply_swinir_sr.py

Applies SwinIR super-resolution to all 10 Sentinel-2 bands in LCZ42 HDF5 datasets.

**Input:**
- training.h5 or testing.h5
- Pretrained SwinIR model
- scaleFactor: 2

**Output:**
- training_swinir2x.h5
- testing_swinir2x.h5

**Usage:**

```bash
python apply_swinir_sr.py
```

**Description:**
- Normalizes reflectance values [0, 2.8] → [0, 1]
- Replicates each band to 3 channels (fake RGB)
- Runs SwinIR inference per band
- Reassembles 10-channel super-resolved patch [64×64×10]
- Saves /sen2 and /label to new HDF5 file


### compare_swinir_results.py

Visualizes all 10 Sentinel-2 channels before and after SwinIR enhancement.

**Usage:**

```bash
python compare_swinir_results.py
```

**Output:**

Displays a 2×10 grid comparing:
- Top row: Original bicubic LR channels
- Bottom row: SwinIR super-resolved channels

### compare_swinir_rgb.py

Visualizes RGB composite (B4, B3, B2) before and after SwinIR enhancement.

**Usage:**

```bash
python compare_swinir_rgb.py
```

**Output:**

Side-by-side visualization:
- Left: Bicubic RGB (32×32)
- Right: SwinIR RGB (64×64)

## BSRNet

Blind Super-Resolution Network (BSRNet) and BSRGAN — robust CNN-based SR models for real-world image restoration.

### Pretrained Models

Located in the `weights/` folder:

- **BSRNet.pth** — Trained for **PSNR-oriented ×2** super-resolution  
- **BSRGAN.pth** — Perceptual **GAN variant** (higher visual realism, slightly lower PSNR)  

**Source:** [CSZN/BSRGAN (ETH Zürich)](https://github.com/cszn/BSRGAN)  
**Paper:** Zhang et al., *“Designing a Practical Degradation Model for Deep Blind Image Super-Resolution,”* ICCV 2021  
**Framework:** PyTorch  
**Type:** RRDBNet backbone (ResNet-in-ResNet dense blocks)  
**Scale:** ×2  

### bsrnet_model.py

Defines and loads the BSRNet or BSRGAN model (RRDBNet) using local pretrained weights.

**Function:**  
`define_model(scale=2, device="cuda", weights_path=None, use_gan=False)`

**Description:**
- Imports the **RRDBNet** architecture directly from the local cloned `BSRGAN/` repository  
- Loads either **BSRNet** or **BSRGAN** weights  
- Automatically maps to GPU if available  
- Returns a ready-to-infer PyTorch model

**Usage:**
```python
from bsrnet_model import define_model
model = define_model(scale=2, device="cuda", use_gan=False)
```

### apply_bsrnet_sr.py

Applies BSRNet/BSRGAN super-resolution to Sentinel-2 imagery in HDF5 datasets.

**Input:**
- training.h5 or testing.h5 (LCZ42 format)
- Pretrained BSRNet.pth or BSRGAN.pth
- scaleFactor: 2

**Output:**
- training_bsrnet2x.h5
- testing_bsrnet2x.h5

**Usage:**

```bash
python apply_bsrnet_sr.py
```

**Processing Details**

- Each Sentinel-2 band is normalized [0, 2.8] → [0, 1]
- Each band is replicated to 3 channels (fake RGB) for compatibility
- Batched CUDA inference via RRDBNet backbone
- Reassembled into 10-band super-resolved patches [64×64×10]
- Stored in new HDF5 files with original labels unchanged

## Real-ESRGAN

Real-ESRGAN — GAN-based super-resolution model for high-scale (×4) upsampling of Sentinel-2 imagery.

### Pretrained Model

- **File:** `RealESRGAN_x4plus.pth`
- **Source:** [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- **Paper:** Wang et al., *“Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data,”* ICCVW 2021
- **Framework:** PyTorch + BasicSR
- **Scale factor:** ×4
- **Architecture:** RRDBNet backbone (ResNet-in-ResNet dense blocks)
- **Strength:** Handles real-world noise and degradations for realistic, perceptually rich outputs

### apply_realesrgan_sr.py

Applies Real-ESRGAN super-resolution to Sentinel-2 imagery stored in HDF5 datasets.

**Input:**  

- `training.h5` or `testing.h5`  
- Pretrained model `RealESRGAN_x4plus.pth`  
- `scaleFactor`: 4  

**Output:**  

- `training_realesrgan4x.h5`  
- `testing_realesrgan4x.h5`  

**Usage:**

```bash
python apply_realesrgan_sr.py
```

**Processing Details**
- Each Sentinel-2 band (10 total) is normalized [0, 2.8] → [0, 1]
- Each band is replicated to RGB (fake color triplet)
- Converted to BGR for compatibility with Real-ESRGAN
- Inference performed via RealESRGANer.enhance()
- Output reassembled into [128×128×10] super-resolved multispectral patch
- Converted back to [0, 2.8] range and written to new HDF5 files

## evaluate_sr_metrics_dataset.m

Quantitatively evaluates the SR quality across the entire dataset using **PSNR**, **SSIM**, and **RMSE** metrics.

**Function:** 
 
`metrics = evaluate_sr_metrics_dataset(h5_orig, h5_sr, modality)`

**Arguments:**

| Parameter | Description |
|:-----------|:-------------|
| `h5_orig` | Path to original HDF5 dataset (e.g. `training.h5`) |
| `h5_sr` | Path to SR-enhanced HDF5 dataset |
| `modality` | `"MS"` (Sentinel-2) or `"SAR"` (Sentinel-1)` |

**Output (`metrics` struct):**

- `PSNR`, `SSIM`, `RMSE`: per-patch, per-band values  
- `MeanPSNR`, `MeanSSIM`, `MeanRMSE`: dataset-averaged metrics

**Usage Example:**

```matlab
metrics = evaluate_sr_metrics_dataset( ...
    'data/lcz42/training.h5', ...
    'data/lcz42/training_swinir2x.h5', ...
    'MS');
disp(metrics.MeanPSNR)
```

**Processing Details:**
- Automatically scales reflectance [0, 2.8] → [0, 255]
- Resizes original LR patches to match SR resolution via bicubic interpolation
- Computes PSNR, SSIM, and RMSE band-wise across all samples
- Supports both multispectral (SEN2) and SAR (SEN1) modalities

## Maintainer

**Matteo Rambaldi** — University of Padua  •  MSc Artificial Intelligence and Robotics (2025)
