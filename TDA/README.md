# Topological Data Analysis (TDA)

This folder contains all code for extracting, visualizing, and integrating **Topological Data Analysis (TDA)** features into the LCZ42 classification pipeline.

TDA is used to analyze **internal feature spaces** and **spatial structures** of Sentinel-1/2 imagery, revealing geometric information not captured by standard CNNs.  
These features are later combined with deep embeddings (ResNet18) for improved robustness and interpretability.

## Overview of the TDA Pipeline

The workflow consists of three main stages:

1. **Feature Extraction** – Compute persistence diagrams and persistence images (PIs) from HDF5 patches  
2. **Feature Visualization** – Explore homological dimensions and topological structure  
3. **Model Training** – Train MLPs and CNN–TDA fusion models for LCZ classification

## TDA Feature Extraction

### extract_all_tda_features.py

Computes TDA features (persistence images) for each Sentinel-2 and Sentinel-1 patch.

**Description:**
- Uses **Cubical Persistence** (homology dimensions 0 and 1)
- Converts each patch → persistence diagram → persistence image (PI)
- Produces flattened TDA feature tensors for training

**Output:**
- `tda_MS_features.npy` — shape `(N, 10×2×n_bins×n_bins)`
- `tda_SAR_features.npy` — shape `(N, 8×2×n_bins×n_bins)`

**Usage:**

```bash
python extract_all_tda_features.py
```

## Label Extraction

### extract_labels.py

Extracts and saves class labels aligned with TDA features.

**Output:**
- labels.npy — class indices (1–17)

**Usage:**

```bash
python extract_labels.py
```

## Visualization

### visualize_tda_features.py

Displays persistence images (H₀, H₁) per spectral band to inspect structural differences.

**Usage:**

```bash
python visualize_tda_features.py
```

Produces a 10×2 panel grid (10 bands × 2 homology dimensions).

## Model Architectures

### TDA-only MLP

Defined in models/tda_mlp.py — simple 4-layer MLP for classification from flattened PIs.

**Train Script:** train_tda_mlp.py


**Usage:**

python train_tda_mlp.py

**Produces:**
- tda_mlp_model_ms.pth or tda_mlp_model_sar.pth

## CNN–TDA Fusion Network

Defined in models/fusion_resnet18.py — combines:
- ResNet18 backbone for image input
- TDA-MLP branch for topological features
- Fused classification head (640 → 17 classes)

**Train Script:** train_fusion_resnet18.py

**Usage:**

python train_fusion_resnet18.py --ensemble_type Rand
python train_fusion_resnet18.py --ensemble_type RandRGB
python train_fusion_resnet18.py --ensemble_type ensembleSAR

**Input:**
- Sentinel-2 or SAR patches from HDF5
- Corresponding tda_MS_features.npy or tda_SAR_features.npy
- labels.npy

**Output:**
- fusion_resnet18_rand.pth
- fusion_resnet18_randrgb.pth
- fusion_resnet18_ensemblesar.pth

## Single-Sample TDA Visualization

### TDA-only/extract_patch_for_tda.py

Visualizes the persistence image for one Sentinel-2 patch.

**Usage:**

```bash
python extract_patch_for_tda.py
```

Outputs persistence images for homology dimensions H₀ and H₁ using Giotto-TDA.

⸻

## Technical Notes

- Dependencies: giotto-tda, torch, torchvision, h5py, numpy, tqdm, matplotlib, scikit-learn
- Feature dimensionality:
- 10 Sentinel-2 bands × 2 homology dims × (30×30 bins) → 18,000 features
- 8 Sentinel-1 bands × 2 homology dims × (30×30 bins) → 14,400 features
- Hardware: GPU recommended for training fusion models; CPU sufficient for TDA extraction.

## Maintainer

**Matteo Rambaldi** — University of Padua  •  MSc Artificial Intelligence and Robotics (2025)