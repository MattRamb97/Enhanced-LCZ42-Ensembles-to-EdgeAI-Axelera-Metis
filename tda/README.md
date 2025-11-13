# Topological Data Analysis (TDA)

## Overview

This directory contains code for extracting, visualizing and integrating **Topological Data Analysis (TDA)** features into the LCZ42 classification pipeline.

TDA analyzes **internal feature spaces** and **spatial structures** of Sentinel-1/2 imagery using persistent homology, revealing geometric information not captured by standard CNNs. These features are combined with deep embeddings (ResNet18) to create fusion models for improved robustness and interpretability.

---

## TDA Pipeline Architecture

The complete workflow consists of three main stages:

```
Sentinel-1/2 HDF5 Patches
        │ (32×32×{10,8} bands)
        ▼
Cubical Persistence + Persistence Images
        │ (Giotto-TDA)
        ▼
TDA Feature Tensors
        │ (18K / 14.4K features)
        ▼
TDA-MLP + CNN-ResNet18 Fusion Model
        │ (Concatenated features)
        ▼
LCZ Classification (17 classes)
```

---

## Feature Extraction

### extract_all_tda_features_train_base.py

Computes topological features from **baseline Sentinel-2/SAR** patches.

**Workflow:**
1. Reads patches from `/sen2` (Sentinel-2) and `/sen1` (Sentinel-1) in HDF5
2. Applies **Cubical Persistence** (Giotto-TDA) computing homology dimensions H₀ (connected components) and H₁ (loops)
3. Converts each band's persistence diagram → **Persistence Image (PI)** (30×30 grid)
4. Stacks PIs across all bands into multi-dimensional feature tensor
5. Flattens and saves as HDF5 arrays

**Configuration (hardcoded):**
```python
H5_PATH = "../../data/lcz42/training.h5"
OUTPUT_MS = "../data/tda_MS_features.h5"       # Sentinel-2 features
OUTPUT_SAR = "../data/tda_SAR_features.h5"     # Sentinel-1 features

n_bins = 30              # Persistence image resolution (30×30)
sigma = 0.5              # Gaussian kernel width for PI smoothing
dtype_out = np.float16   # Output precision (saves 50% storage vs float32)
```

**Output Shapes:**
- **Sentinel-2 (MS):** `(352000, 18000)` — 10 bands × 2 dimensions × 30×30 bins
- **Sentinel-1 (SAR):** `(352000, 14400)` — 8 bands × 2 dimensions × 30×30 bins

**Usage:**
```bash
python scripts/extract_all_tda_features_train_base.py
python scripts/extract_all_tda_features_test_base.py
```

**Processing Details:**
- Parallelized across available CPU cores
- Cubical persistence more efficient than Vietoris-Rips for 2D grid data
- H₀ and H₁ capture complementary topological information

---

### extract_all_tda_features_train_SR.py / extract_all_tda_features_test_SR.py

Same TDA extraction workflow but for **super-resolution enhanced variants**.

**Input:** Any SR-enhanced HDF5 (e.g., `training_swinir2x.h5`, `training_edsr4x.h5`)

**Usage:**
```bash
python scripts/extract_all_tda_features_train_SR.py
python scripts/extract_all_tda_features_test_SR.py
```

**Output:** SR-specific feature files (e.g., `tda_MS_features_swinir2x.h5`)

---

## Label Extraction

### extract_labels.py

Extracts LCZ class labels from baseline HDF5 files.

**Input:**
- `/label` dataset (one-hot encoded, 17 classes) from training.h5

**Output:**
- `labels.npy` — class indices 0–16

**Usage:**
```bash
python scripts/extract_labels.py
```

---

## Visualization

### plot_tda_persistence.py

Visualizes persistence images (H₀, H₁) for all spectral bands.

**Output:** 2D heatmap grid
- Rows: 10 (Sentinel-2) or 8 (Sentinel-1) bands
- Columns: 2 homology dimensions

**Usage:**
```bash
python scripts/plot_tda_persistence.py
```

---

## Model Architectures

### TDA-only MLP

Defined in [models/](models/) — simple feedforward classifier from flattened persistence images.

**Architecture:**
```
TDA Features (18,000)
    ↓
FC(18000 → 1024) → BatchNorm → ReLU → Dropout
    ↓
FC(1024 → 512) → BatchNorm → ReLU → Dropout
    ↓
FC(512 → 256) → BatchNorm → ReLU → Dropout
    ↓
FC(256 → 128) → BatchNorm → ReLU
    ↓
FC(128 → 17) — Output logits
```

---

### CNN-TDA Fusion Network

Defined in [models/fusion_resnet18.py](models/fusion_resnet18.py) — combines ResNet18 image processing with TDA topological features.

**Architecture:**

```
Multispectral Patch          TDA Features
(32×32×3 bands)              (18,000)
    │                             │
    ▼                             ▼
ResNet18 Backbone            TDA MLP
(→ 512 features)             (→ 128 features)
    │                             │
    └─────────────┬───────────────┘
                  │
                  ▼
          Concatenate [512, 128]
          (→ 640 features)
                  │
                  ▼
          Fusion Classifier
          FC(640 → 128) → ReLU → Dropout
          FC(128 → 17) — Output
```

**Configuration (hardcoded):**
```python
H5_PATH = "../data/lcz42/training.h5"
LABELS_PATH = "labels.npy"
NUM_CLASSES = 17
EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 3e-4
DROPOUT = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

---

## Model Training

### train_fusion_resnet18.py

Trains CNN-TDA fusion models with three different ensemble variants.

**Usage:**
```bash
python scripts/train_fusion_resnet18.py --ensemble_type Rand
python scripts/train_fusion_resnet18.py --ensemble_type RandRGB
python scripts/train_fusion_resnet18.py --ensemble_type ensembleSAR
```

**Ensemble Variants:**

| Type | Input Modality | Band Selection | Use Case |
|:-----|:---------------|:---------------|:---------|
| **Rand** | Sentinel-2 only | 3 random bands from 10 | Robust to spectral variability |
| **RandRGB** | Sentinel-2 only | True color RGB (B4/B3/B2) + 2 random | Interpretable + augmented |
| **ensembleSAR** | Sentinel-2 + Sentinel-1 | 2 MS bands + 1 SAR band | Multi-modal fusion |

**Dataset Implementation:**

```python
class FusionDataset(Dataset):
    def __getitem__(self, idx):
        # Load pre-extracted TDA features
        tda_features = self.tda_array[idx]        # [18000]

        # Load multispectral patch from HDF5
        ms_patch = h5["sen2"][idx]                # [32, 32, 10]

        # Ensemble-specific band selection
        if ensemble_type == "Rand":
            bands = np.random.choice(10, size=3, replace=False)
            image = ms_patch[..., bands]

        elif ensemble_type == "RandRGB":
            rgb_bands = np.array([3, 2, 1])       # B4, B3, B2
            random_bands = np.random.choice([i for i in range(10)
                                            if i not in [1,2,3]], size=2)
            bands = np.concatenate([rgb_bands, random_bands])
            image = ms_patch[..., bands]

        elif ensemble_type == "ensembleSAR":
            sar_patch = h5["sen1"][idx]           # [32, 32, 8]
            # Fuse 2 MS bands + 1 SAR band
            ...

        return image, tda_features, label
```

**Training Output:**
- `fusion_resnet18_rand.pth` — Random ensemble weights
- `fusion_resnet18_randrgb.pth` — RandRGB ensemble weights
- `fusion_resnet18_ensemblesar.pth` — SAR fusion weights
- Training logs and validation metrics

---

## Utilities

### data/HDF5_info.py

Inspects HDF5 file structure and metadata.

**Usage:**
```bash
python data/HDF5_info.py --h5-path ../../data/lcz42/training.h5
```

**Output:**
```
HDF5 Datasets:
  /label — shape (352000, 17), dtype float64
  /sen1  — shape (352000, 32, 32, 8), dtype float64
  /sen2  — shape (352000, 32, 32, 10), dtype float64
```

### npy-matlab/

Legacy utilities for NumPy ↔ MATLAB array format conversion.

---

## Dependencies

See [requirements.txt](requirements.txt):

```
giotto-tda==0.6.2          # Persistent homology computation
torch==2.3.1               # Deep learning framework
torchvision==0.18.1        # Vision models (ResNet18)
h5py==3.15.1               # HDF5 file I/O
numpy==1.26.4              # Numerical arrays
scikit-learn==1.3.2        # Preprocessing & metrics
matplotlib==3.10.7         # Visualization
tqdm==4.67.0               # Progress bars
```

**Installation:**
```bash
pip install -r requirements.txt
```

---

## Complete Workflow Example

**End-to-end pipeline execution:**

```bash
# 1. Extract TDA features from baseline training/test sets
python scripts/extract_all_tda_features_train_base.py
python scripts/extract_all_tda_features_test_base.py

# 2. Extract corresponding class labels
python scripts/extract_labels.py

# 3. Optional: Visualize topological structures
python scripts/plot_tda_persistence.py

# 4. Train fusion model (choose one variant)
python scripts/train_fusion_resnet18.py --ensemble_type Rand
# or
python scripts/train_fusion_resnet18.py --ensemble_type RandRGB
# or
python scripts/train_fusion_resnet18.py --ensemble_type ensembleSAR

# 5. Evaluate on test set (integrated in training script)
```

---

## Hardware & Performance

| Stage | Hardware | Time | Notes |
|:------|:---------|:-----|:------|
| TDA Extraction | CPU (12 cores) | ~2–3 hours | Parallelized across cores |
| Feature I/O | SSD | ~5–10 min | Write 18K × 352K features |
| Fusion Training | A40 | ~20–30 min | 50 epochs, batch 128 |
| Fusion Training | CPU | ~2–3 hours | Not recommended |

---

## Citation

If you use TDA features or fusion models, please cite:

```bibtex
@mastersthesis{rambaldi2025enhancedlcz42,
  title={Enhanced LCZ42 Ensembles to Edge AI: Knowledge Distillation and Deployment on Axelera Metis},
  author={Rambaldi, Matteo},
  school={University of Padua},
  year={2025},
  note={GitHub Repository: https://github.com/matteorambaldi/Enhanced-LCZ42-Ensembles-to-EdgeAI-Axelera-Metis}
}
```

Also cite Giotto-TDA:
```bibtex
@inproceedings{giansiracusa2021giotto,
  title={Giotto-tda: A Topological Data Analysis Toolkit for Machine Learning},
  author={Giansiracusa, N. and others},
  booktitle={Frontiers in Artificial Intelligence},
  year={2021}
}
```

---

## License

This project is released under the MIT License.
See the LICENSE file for details.

---

## Author & Attribution

**Project:** Matteo Rambaldi

**Affiliation:** MSc Artificial Intelligence, University of Padua

**Supervision:** Prof. Loris Nanni

**Co-Supervision:** Eng. Cristian Garjitzky

**TDA Framework:** Giotto-TDA (Giansiracusa et al., 2021)
