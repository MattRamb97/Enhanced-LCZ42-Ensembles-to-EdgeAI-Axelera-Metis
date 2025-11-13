# DenseNet-201 Ensembles

## Overview

This directory contains the complete **PyTorch implementation** for training **DenseNet-201 ensemble teachers** on the So2Sat LCZ42 dataset. DenseNet-201 is a large-capacity architecture (60M parameters) optimized for high classification accuracy and serving as supervision for downstream tasks (knowledge distillation, super-resolution, TDA fusion).

Three ensemble variants are trained on different input modalities:
- **RAND** — 3 random multispectral (MS) bands per member
- **RANDRGB** — RGB-constrained ensemble (B4, B3, B2 + random MS augmentation)
- **SAR** — MS+SAR fusion (2 MS bands + 1 SAR band per member)

Each variant trains **10 ensemble members** with diverse random seeds for rich supervision diversity.

---

## Directory Structure

```
densenet201_ensembles/
├── scripts/              # Training, evaluation, and utilities
├── models/
│   └── trained/          # Trained ensemble checkpoints (.pth)
├── results/              # Evaluation metrics, confusion matrices, histories
├── requirements.txt      # Python dependencies
└── README.md             # (This file)
```

---

## Core Scripts

### Training Orchestrators

**train_teacher_densenet201.py**

Main entry point for training ensembles. Orchestrates RAND, RANDRGB, and/or SAR training sequentially.

**Usage:**
```bash
python scripts/train_teacher_densenet201.py --mode ALL      # Train all three variants
python scripts/train_teacher_densenet201.py --mode RAND     # Train only RAND
python scripts/train_teacher_densenet201.py --mode RANDRGB  # Train only RANDRGB
python scripts/train_teacher_densenet201.py --mode SAR      # Train only SAR
```

**Configuration (hardcoded):**
- **Epochs:** 12
- **Batch size:** 512 (large for efficient GPU utilization)
- **Learning rate:** 1e-3 (SGD with momentum)
- **Seed:** 42 (reproducibility)
- **Z-score normalization:** enabled
- **SAR despeckling:** enabled
- **Augmentation:** enabled (geometric transforms during training)

---

### Ensemble Training Variants

**rand_densenet201.py**

Train DenseNet-201 ensemble on **3 random multispectral (MS) bands** per member.

**Key Details:**
- Each of 10 members independently selects 3 bands from 10 available MS channels
- Uses pretrained DenseNet-201 (`DenseNet201_Weights.DEFAULT`)
- Outputs: trained models, training histories, ensemble metrics

**Architecture:**
- Input: 3-channel image (3, 224, 224)
- Backbone: DenseNet-201 dense blocks + growth rate layers
- Head: Fully connected layer (1920 → 17 classes)

**Output Structure:**
```python
{
    'models': [10 trained DenseNet201MS instances],
    'bands': [[b1, b2, b3], ...],  # per-member band selection
    'test_top1': float,             # ensemble test accuracy
    'confusion_mat': (17, 17),
    'y_true': array,
    'y_pred': array,
    'histories': dict                # training logs per member
}
```

---

**randrgb_densenet201.py**

Train DenseNet-201 ensemble on **RGB-constrained bands** (B4, B3, B2) per member.

**Differences from RAND:**
- Fixed RGB core (B4, B3, B2) for interpretable visualization
- Better controlled diversity through augmentation rather than band selection
- Same architecture and training pipeline

---

**randsar_densenet201.py**

Train DenseNet-201 ensemble on **MS+SAR fusion** (2 MS + 1 SAR band per member).

**Key Details:**
- Each member concatenates: 2 random MS bands + 1 random SAR band → 3-channel input
- Leverages DenseNet's dense connections to learn cross-modal fusion
- MS bands selected from 10 Sentinel-2 channels
- SAR band selected from 8 Sentinel-1 channels

**Data Flow:**
```
[MS band 1] + [MS band 2] + [SAR band] → [3-channel] → DenseNet-201
```

---

### Utilities

**dataset_reading.py**

Builds PyTorch datastores from HDF5 files with on-demand patch loading and preprocessing.

**Key Features:**
- Lazy loading: patches read on-demand to minimize memory footprint
- Per-channel z-score normalization (optional)
- SAR despeckling via Lee filter (optional)
- Geometric augmentation: random rotation, flip, crop (training only)
- Handles both MS (10 bands) and SAR (8 bands) modalities

**Usage:**
```python
from dataset_reading import DatasetReading

cfg = {
    'trainTable': train_df,
    'testTable': test_df,
    'useZscore': True,
    'useSARdespeckle': True,
    'useAugmentation': True,
}
dsTrain, dsTest, info = DatasetReading(cfg)
```

---

**h5_reader.py**

Low-level utility to read individual patches from HDF5 files.

**Input:**
- `h5_path`: Path to HDF5 file
- `index`: 1-based sample index
- `modality`: `"MS"` or `"SAR"`

**Output:**
- Single patch as numpy array: (32, 32, C) where C = 10 for MS, C = 8 for SAR

**Usage:**
```python
from h5_reader import h5_reader

X = h5_reader("data/lcz42/training.h5", 12345, "MS")  # [32, 32, 10]
```

---

**make_tables_from_h5.py**

Generates index lookup tables from HDF5 datasets for efficient batch loading.

**Input:**
- Path to folder containing `training.h5` and `testing.h5`

**Output:**
- `tables_MS.mat` — Train/test tables for MS patches
- `tables_SAR.mat` — Train/test tables for SAR patches

**Table Columns:**

| Column | Type | Description |
|:-------|:-----|:------------|
| Path | string | Absolute path to HDF5 file |
| Label | int | LCZ class (1–17) |
| Index | int | 1-based position in H5 dataset |
| Modality | string | `"MS"` or `"SAR"` |

**Usage:**
```bash
python scripts/make_tables_from_h5.py --data-root ../../data/lcz42
```

---

**enable_gpu.py**

Detects and initializes GPU acceleration automatically.

**Usage:**
```python
from enable_gpu import enable_gpu
device = enable_gpu(verbose=True)  # Returns 'cuda' or 'cpu'
```

---

### Inspection & Visualization

**inspect_results.py**

Plots training curves, confusion matrices, and per-class metrics for trained ensembles.

**Generates:**
- Training loss/accuracy curves per member
- Ensemble confusion matrix (17×17)
- Per-class precision, recall, F1 scores
- Aggregate statistics

**Usage:**
```bash
python scripts/inspect_results.py --mode RAND
```

---

## Trained Models (`models/trained/`)

| File | Ensemble | Input | Members | Top-1 Acc. |
|:-----|:---------|:------|:-------:|:----------:|
| `Rand_densenet201.pth` | RAND | 3 random MS bands | 10 | ~80–83% |
| `RandRGB_densenet201.pth` | RANDRGB | RGB (B4,B3,B2) | 10 | ~79–82% |
| `SAR_densenet201.pth` | SAR | 2 MS + 1 SAR | 10 | ~77–80% |

**Model Loading:**
```python
import torch

ensemble = torch.load("models/trained/Rand_densenet201.pth", map_location='cpu')
probs = ensemble(image_tensor)  # Forward pass
```

---

## Results Structure

```
results/
├── rand/
│   ├── rand_summary.json           # Ensemble metrics & configuration
│   ├── rand_history.csv            # Per-epoch loss/accuracy
│   ├── rand_members.csv            # Per-member final metrics
│   └── rand_eval_TEST.h5           # Test predictions (HDF5)
├── randrgb/
│   ├── randrgb_summary.json
│   ├── randrgb_history.csv
│   ├── randrgb_members.csv
│   └── randrgb_eval_TEST.h5
├── sar/
│   ├── sar_summary.json
│   ├── sar_history.csv
│   ├── sar_members.csv
│   └── sar_eval_TEST.h5
└── fusion/
    ├── densenet201_sumrule_summary.json      # Cross-mode fusion (RAND+RANDRGB+SAR)
    └── densenet201_sumrule_eval_TEST.h5
```

---

## Training Workflow

**Step 1: Install dependencies**
```bash
cd densenet201_ensembles
pip install -r requirements.txt
```

**Step 2: Prepare HDF5 tables**
```bash
python scripts/make_tables_from_h5.py --data-root ../../data/lcz42
```

**Step 3: Train ensembles**
```bash
python scripts/train_teacher_densenet201.py --mode ALL
```

**Step 4: Inspect results**
```bash
python scripts/inspect_results.py
```

---

## Architecture Details

### DenseNet201MS Module

High-capacity DenseNet-201 adapted for 3-channel Sentinel data:

```python
DenseNet201MS(
  features: DenseNet201_features [
    conv0, norm0, relu,
    denseblock1, transition1,
    denseblock2, transition2,
    denseblock3, transition3,
    denseblock4, norm5
  ],
  avgpool: AdaptiveAvgPool2d(1, 1),
  fc: Linear(1920 → 17)
)
```

**Parameters:** 60.3M (vs. 11.2M for ResNet-18)

**Key Strengths:**
- Dense connections enable feature reuse and gradient flow
- Growth rate (32) balances expressiveness and parameter count
- Pretrained ImageNet weights transfer well to Sentinel data

---

### Ensemble Aggregation

At inference, predictions from all 10 members are averaged (sum-rule fusion):

```python
probs_ensemble = mean(softmax(logits_member_1), ..., softmax(logits_member_10))
pred = argmax(probs_ensemble)
```

This **sum-rule fusion** provides:
- Noise reduction through averaging
- Diversity benefit from independent random seeds
- Robust per-class confidence scores for downstream tasks

---

## Hardware & Performance

| Hardware | Batch Size | Time per Epoch | Total (12 ep.) |
|:---------|:-----------|:---------------|:--------------|
| A40 | 512 | ~23-25 min | ~3.5-3.8 hours |

**Memory Requirements:**
- DenseNet-201: ~10–12 GB per GPU with batch=512
- Adjust batch size down if OOM occurs

---

## Cluster Considerations

**NFS I/O Optimization:**
- `DatasetReading` uses on-demand H5 reading → minimal memory footprint
- Large batch sizes help amortize I/O overhead
- With concurrent jobs, consider reducing `NUM_WORKERS` to 2–4

**Reproducibility:**
- Fixed seed (42) ensures deterministic results
- GPU operations non-deterministic by default; use `torch.backends.cudnn.deterministic = True` if needed

**Training Time:**
- Full pipeline (RAND + RANDRGB + SAR) ≈ 5–8 hours on A100
- Per-mode (single variant) ≈ 1.5–2.5 hours on A100

---

## Use Cases

**DenseNet-201 Ensembles are ideal for:**

1. **High-Accuracy Teachers** — Supervision signals for knowledge distillation to lightweight students (ResNet-18)
2. **Super-Resolution Guidance** — Teacher predictions for SR-assisted classification pipelines
3. **TDA Feature Learning** — Ensemble labels for learning topological data analysis features
4. **Fusion Architecture Training** — Multi-modal (MS+SAR) fusion with diverse predictions
5. **Baseline Comparisons** — Upper-bound accuracy for measuring distillation efficiency

---

## Citation

If you use this ensemble training code, please cite:

```bibtex
@mastersthesis{rambaldi2025enhancedlcz42,
  title={Enhanced LCZ42 Ensembles to Edge AI: Knowledge Distillation and Deployment on Axelera Metis},
  author={Rambaldi, Matteo},
  school={University of Padua},
  year={2025},
  note={GitHub Repository: https://github.com/matteorambaldi/Enhanced-LCZ42-Ensembles-to-EdgeAI-Axelera-Metis}
}
```

---

## License

This project is released under the MIT License. See the LICENSE file for details.

---

## Author & Attribution

**Project:** Matteo Rambaldi

**Affiliation:** MSc Artificial Intelligence, University of Padua

**Supervision:** Prof. Loris Nanni

**Co-Supervision:** Eng. Cristian Garjitzky
