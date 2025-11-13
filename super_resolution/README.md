# Super-Resolution Pipelines

## Overview

This directory contains 6 different super-resolution (SR) models for enhancing Sentinel-2 imagery from the So2Sat LCZ42 dataset. Each subfolder includes pretrained weights, inference scripts, and batch processing pipelines.

---

## SR Methods Summary

| Method | Scale | Architecture | Framework | Output File |
|:-------|:-----:|:------------|:----------|:------------|
| **VDSR** | ×2, ×3 | CNN (shallow) | MATLAB | `training_vdsr{2,3}x.h5` |
| **EDSR** | ×2, ×4 | CNN (deep) | PyTorch | `training_edsr{2,4}x.h5` |
| **ESRGAN** | ×2 | GAN (semantic triplets) | MATLAB | `training_esrgan2x.h5` |
| **SwinIR** | ×2 | Transformer (shifted windows) | PyTorch | `training_swinir2x.h5` |
| **BSRNet** | ×2 | Blind SR (RRDBNet) | PyTorch | `training_bsrnet2x.h5` |
| **Real-ESRGAN** | ×4 | GAN (real-world degradation) | PyTorch | `training_realesrgan4x.h5` |

### Output Format

All SR-enhanced HDF5 files maintain:
- `/sen2`: Super-resolved Sentinel-2 multispectral data in [0, 2.8] range, `float16`
- `/label`: Original one-hot encoded LCZ labels (17 classes, unchanged)
- **Sample order preserved:** Identical ordering with baseline `training.h5` / `testing.h5`

---

## VDSR

**Very Deep Super-Resolution Network** for upsampling Sentinel-2 imagery.

### Model: trainedVDSRNet_v2.mat

Pretrained VDSR network from MathWorks:
- **Training data:** IAPR TC-12 Benchmark (natural images)
- **Scale factors:** ×2, ×3 (x4 available)
- **Operation:** Single-channel processing (applied per band independently)

### Key Scripts

#### apply_vdsr_all_bands.m

Applies VDSR to all 10 Sentinel-2 bands.

```matlab
X_sr = apply_vdsr_all_bands(X, vdsrNet, scaleFactor)
```

**Input:**
- `X`: [32×32×10] patch in [0, 255] range
- `vdsrNet`: Loaded VDSR model
- `scaleFactor`: 2, 3, or 4

**Output:**
- `X_sr`: [64×64×10] or [96×96×10] or [128×128×10] patch

#### precompute_vdsr_dataset.m

Batch processes entire HDF5 dataset with VDSR.

```matlab
load('trainedVDSRNet_v2.mat');
precompute_vdsr_dataset('training.h5', 'training_vdsr2x.h5', net, 2);
```

**Input:**
- `inputH5`: Path to training.h5 or testing.h5
- `outputH5`: Output file path
- `vdsrNet`: Loaded VDSR model
- `scaleFactor`: 2, 3, or 4

**Output:** New HDF5 file with upsampled `/sen2` dataset, labels copied unchanged.

---

## EDSR

**Enhanced Deep Super-Resolution Network** for CNN-based image upsampling.

### Model

- **Source:** [eugenesiow/edsr-base](https://huggingface.co/eugenesiow/edsr-base) (Hugging Face)
- **Framework:** PyTorch (`super-image` library)
- **Scale factors:** ×2, ×3, ×4
- **Operation:** Band-by-band processing (each Sentinel-2 channel independently)

### Key Script: apply_edsr_sr.py

Applies EDSR to all 10 Sentinel-2 bands and saves upsampled dataset to HDF5.

```bash
python apply_edsr_sr.py
```

**Input:**
- `training.h5` or `testing.h5`
- `scaleFactor`: 2, 4 (3 available)
- Pretrained EDSR model (auto-downloaded from Hugging Face)

**Output:**
- `training_edsr2x.h5` or `testing_edsr2x.h5`
- `/sen2`: Super-resolved data in [0, 2.8], float16
- `/label`: Original one-hot encoded LCZ labels (17 classes)

---

## ESRGAN

**Enhanced Super-Resolution GAN** with semantic triplet grouping for spectrally similar channels.

### Model: trained/ESRGAN100_RGB_Flickr2K_VGG54_2x_Generator_params_epoch300.mat

Pretrained ESRGAN generator (`dlnG`):
- **Training data:** Flickr2K RGB dataset
- **Perceptual loss:** VGG54
- **Scale:** ×2 upscaling
- **Framework:** MATLAB Deep Learning Toolbox
- **Input:** 3-channel RGB (via semantic triplets)

### Key Scripts

#### apply_esrgan_all_bands.m

Applies ESRGAN SR using semantic triplets that group spectrally similar channels.

```matlab
X_sr = apply_esrgan_all_bands(X, 2);
```

**Semantic Triplet Configuration:**

| Triplet | Bands | Semantic Meaning |
|:--------|:------|:----------------|
| [3,2,1] | B4, B3, B2 | RGB (True Color) |
| [7,5,4] | B8, B6, B5 | Vegetation / Red-Edge |
| [9,10,8] | B11, B12, B8A | Moisture / SWIR-NIR |
| [10,10,6] | B12, B12, B7 | Filler (Spectral Consistency) |

Each triplet is:
1. Normalized per channel to [0, 1]
2. Processed by ESRGAN
3. Recombined into full 10-band cube

**Input:**
- `X`: [32×32×10] multispectral patch in [0, 255] range

**Output:**
- `X_sr`: [64×64×10] super-resolved patch

#### ESRGAN_2xSuperResolution.m

Performs ESRGAN inference on 3-channel input.

```matlab
I_sr = ESRGAN_2xSuperResolution(I_lr);
```

**Features:**
- Loads generator once (persistent `dlnG`)
- Automatic GPU deployment if available
- Normalizes inputs to [0, 1], rescales outputs to [0, 1]
- Returns upscaled image clipped to valid range

#### precompute_esrgan_dataset.m

Batch processes entire HDF5 dataset with ESRGAN.

```matlab
precompute_esrgan_dataset('training.h5', 'training_esrgan2x.h5', 2);
```

**Input:**
- `inputH5`: Path to training.h5 or testing.h5
- `outputH5`: Output file path
- `scaleFactor`: Upscaling factor (2)

**Output:** New HDF5 file with super-resolved `/sen2` dataset, labels unchanged.

---

## SwinIR

**Image Restoration using Swin Transformer** — hierarchical transformer architecture with shifted windows for SR.

### Model

- **Source:** [JingyunLiang/SwinIR](https://github.com/JingyunLiang/SwinIR)
- **Paper:** Liang et al., *"SwinIR: Image Restoration Using Swin Transformer,"* ICCV 2021
- **Framework:** PyTorch
- **Scale factor:** ×2 (classical SR)
- **Operation:** Per-band processing (each band replicated to RGB triplets)

### Key Scripts

#### swinir_model.py

Defines and loads pretrained SwinIR model.

```python
from swinir_model import define_model
model = define_model(scale=2, device="cuda")
```

**Function:** `define_model(scale=2, device="cuda")`

- Loads official pretrained SwinIR-M ×2 model
- Auto-downloads weights from GitHub
- Configured for 3-channel RGB input (per-band replication)

#### apply_swinir_sr.py

Applies SwinIR SR to all 10 Sentinel-2 bands in LCZ42 HDF5 datasets.

```bash
python apply_swinir_sr.py
```

**Input:**
- `training.h5` or `testing.h5`
- Pretrained SwinIR model
- Scale factor: 2

**Output:**
- `training_swinir2x.h5`
- `testing_swinir2x.h5`

**Processing:**
1. Normalizes reflectance [0, 2.8] → [0, 1]
2. Replicates each band to 3 channels (fake RGB)
3. Runs SwinIR inference per band
4. Reassembles 10-channel patch [64×64×10]
5. Saves `/sen2` and `/label` to new HDF5 file

---

## BSRNet / BSRGAN

**Blind Super-Resolution Network** — robust CNN-based SR for real-world image degradations.

### Models

Located in `weights/` folder:

| Model | Purpose | Scale | Type |
|:------|:--------|:-----:|:-----|
| **BSRNet.pth** | PSNR-oriented SR | ×2 | CNN |
| **BSRGAN.pth** | Perceptual GAN variant | ×2 | GAN |

**Source:** [CSZN/BSRGAN](https://github.com/cszn/BSRGAN) (ETH Zürich)
**Paper:** Zhang et al., *"Designing a Practical Degradation Model for Deep Blind Image Super-Resolution,"* ICCV 2021
**Framework:** PyTorch
**Architecture:** RRDBNet backbone (ResNet-in-ResNet dense blocks)

### Key Scripts

#### bsrnet_model.py

Defines and loads BSRNet or BSRGAN model.

```python
from bsrnet_model import define_model
model = define_model(scale=2, device="cuda", use_gan=False)
```

**Function:** `define_model(scale=2, device="cuda", weights_path=None, use_gan=False)`

- Imports RRDBNet architecture from local `BSRGAN/` repository
- Loads either BSRNet or BSRGAN weights
- Auto-maps to GPU if available
- Returns ready-to-infer PyTorch model

#### apply_bsrnet_sr.py

Applies BSRNet/BSRGAN SR to Sentinel-2 imagery in HDF5 datasets.

```bash
python apply_bsrnet_sr.py
```

**Input:**
- `training.h5` or `testing.h5` (LCZ42 format)
- Pretrained BSRNet.pth or BSRGAN.pth
- Scale factor: 2

**Output:**
- `training_bsrnet2x.h5`
- `testing_bsrnet2x.h5`

**Processing:**
1. Each Sentinel-2 band normalized [0, 2.8] → [0, 1]
2. Each band replicated to 3 channels (fake RGB)
3. Batched CUDA inference via RRDBNet backbone
4. Reassembled into 10-band super-resolved patches [64×64×10]
5. Stored in new HDF5 with original labels unchanged

---

## Real-ESRGAN

**Real-ESRGAN** — GAN-based SR for high-scale (×4) upsampling of satellite imagery.

### Model: RealESRGAN_x4plus.pth

Pretrained Real-ESRGAN generator:
- **Source:** [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- **Paper:** Wang et al., *"Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data,"* ICCVW 2021
- **Framework:** PyTorch + BasicSR
- **Scale factor:** ×4
- **Architecture:** RRDBNet backbone (ResNet-in-ResNet dense blocks)
- **Strength:** Handles real-world noise and degradations for realistic outputs

### Key Script: apply_realesrgan_sr.py

Applies Real-ESRGAN SR to Sentinel-2 imagery in HDF5 datasets.

```bash
python apply_realesrgan_sr.py
```

**Input:**
- `training.h5` or `testing.h5`
- Pretrained model `RealESRGAN_x4plus.pth`
- Scale factor: 4

**Output:**
- `training_realesrgan4x.h5`
- `testing_realesrgan4x.h5`

**Processing:**
1. Each Sentinel-2 band (10 total) normalized [0, 2.8] → [0, 1]
2. Each band replicated to RGB (fake color triplet)
3. Converted to BGR for Real-ESRGAN compatibility
4. Inference via `RealESRGANer.enhance()`
5. Output reassembled into [128×128×10] super-resolved patch
6. Converted back to [0, 2.8] range and written to new HDF5 files

---

## Evaluation

Two evaluation scripts are provided for quantitatively assessing super-resolution quality:

### evaluate_sr_metrics_dataset.m

Core evaluation function that computes **PSNR**, **SSIM**, and **RMSE** metrics across an entire SR-enhanced dataset.

```matlab
metrics = evaluate_sr_metrics_dataset(h5_orig, h5_sr, num_samples)
```

**Parameters:**

| Parameter | Description |
|:----------|:------------|
| `h5_orig` | Path to original baseline HDF5 (e.g. `training.h5`) |
| `h5_sr` | Path to SR-enhanced HDF5 |
| `num_samples` | Optional: limit evaluation to N samples (default: all) |

**Output (`metrics` struct):**
- `PSNR`, `SSIM`, `RMSE`: per-patch, per-band matrices
- `MeanPSNR`, `MeanSSIM`, `MeanRMSE`: dataset-averaged metrics
- `StdPSNR`, `StdSSIM`, `StdRMSE`: standard deviations
- `MeanPSNR_perBand`, `MeanSSIM_perBand`, `MeanRMSE_perBand`: per-band averages

**Usage Example:**

```matlab
metrics = evaluate_sr_metrics_dataset( ...
    'data/lcz42/training.h5', ...
    'data/lcz42/training_swinir2x.h5', ...
    352000);  % Evaluate all 352K training samples

disp(metrics.MeanPSNR)   % ~30-35 dB for good SR models
disp(metrics.MeanSSIM)   % ~0.8-0.95 for Sentinel-2 bands
```

**Features:**
- **Parallel processing:** Automatically uses Parallel Computing Toolbox if available
- **Batch processing:** 1024-sample batches for memory efficiency
- **Auto-normalization:** Handles [0, 2.8] reflectance scale automatically
- **Progress tracking:** Real-time estimates with ETA
- **Helper functions:**
  - `h5_reader_ms()`: Efficient HDF5 batch reader for `/sen2` data
  - `process_single_sample()`: Per-band PSNR/SSIM/RMSE computation

**Processing Details:**
1. Reads baseline patches in batches from `/sen2` dataset
2. Resizes original LR patches [32×32] to match SR resolution via bicubic interpolation
3. Normalizes to [0, 1] range (auto-detects [0, 2.8] scale)
4. Computes metrics band-wise across all 10 Sentinel-2 channels
5. Aggregates per-patch, per-band, and global statistics

---

### batch_evaluate_ms_models.m

Complete batch evaluation script for systematically comparing all 8 SR-enhanced models.

```matlab
batch_evaluate_ms_models
```

**Configuration:**
Editable at top of script:
- `base_path`: Path to data/lcz42/ directory
- `output_path`: Where to save evaluation results
- `models`: Cell array of model names and .h5 filenames

**Models Evaluated:**
1. VDSRx2, VDSRx3
2. EDSRx2, EDSRx4
3. ESRGANx2
4. SWINIRx2
5. BSRNETx2
6. RealESRGANx4

**Workflow:**

1. **Verification** — Checks all .h5 files exist and have consistent sample counts
2. **Sequential Evaluation** — Processes each model using `evaluate_sr_metrics_dataset()`
3. **Checkpoint Saving** — Saves intermediate results after each model (crash-safe)
4. **Results Aggregation** — Combines metrics into comparison table
5. **Latex Export** — Generates publication-ready table (`.tex` format)
6. **Final Report** — Displays ranked results, identifies best performer

**Output Files:**
- `metrics/checkpoint_ms_<ModelName>.mat` — Individual model checkpoints
- `metrics/ms_metrics_training.mat` — Complete results struct
- `metrics/results_table.tex` — LaTeX table for thesis/paper

**Output Example:**

```
╔══════════════════════════════════════════╗
║      MS EVALUATION RESULTS TABLE         ║
╚══════════════════════════════════════════╝

Method          SSIM        RMSE        PSNR_dB
──────────────────────────────────────────────
Real-ESRGANx4   0.9123      0.0234      34.56
SWINIRx2        0.8954      0.0267      33.42
EDSRx4          0.8756      0.0301      32.18
...
```

**Performance Notes:**
- Estimated ~300 samples/sec per model on M4 Pro
- Full evaluation: ~1-2 hours for all 352K training samples × 8 models
- Enable parallel processing for significant speedup (10+ cores recommended)

**Usage Workflow:**
```matlab
% 1. Edit paths at top of batch_evaluate_ms_models.m
% 2. Run full evaluation
batch_evaluate_ms_models

% 3. Load and inspect results
load('metrics/ms_metrics_training.mat', 'resultsTable');
disp(resultsTable);
```

---

## Citation

If you use any of these super-resolution methods or enhanced datasets, please cite:

```bibtex
@mastersthesis{rambaldi2025enhancedlcz42,
  title={Enhanced LCZ42 Ensembles to Edge AI: Knowledge Distillation and Deployment on Axelera Metis},
  author={Rambaldi, Matteo},
  school={University of Padua},
  year={2025},
  note={GitHub Repository: https://github.com/matteorambaldi/Enhanced-LCZ42-Ensembles-to-EdgeAI-Axelera-Metis}
}
```

Also consider citing the original SR method papers referenced in each section.

---

## License

This project is released under the MIT License.
See the LICENSE file for details.

Pretrained weights and models may be subject to their respective licenses:
- VDSR, EDSR, SwinIR, BSRNet, Real-ESRGAN: See respective GitHub repositories
- ESRGAN: Licensed under Apache 2.0

---

## Author & Attribution

**Project:** Matteo Rambaldi

**Affiliation:** MSc Artificial Intelligence, University of Padua

**Supervision:** Prof. Loris Nanni

**Co-Supervision:** Eng. Cristian Garjitzky

**SR Method Attribution:** See each method's section for original authors and references.
