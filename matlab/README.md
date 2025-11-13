# MATLAB Teacher Ensemble Training

## Overview

This directory contains the complete MATLAB implementation for training **teacher ensembles** that serve as supervision signals for the Enhanced LCZ42 pipeline. Two architectures are supported:

- **DenseNet-201 Ensembles** — Large capacity teachers (densenet201_ensembles/)
- **ResNet-18 Ensembles** — Lightweight teachers (resnet18_ensembles/)

Both architectures support three ensemble variants for diverse supervision:
- **RAND** — 3 random multispectral (MS) bands per member
- **RANDRGB** — RGB-constrained ensemble (B4, B3, B2 + random MS bands)
- **SAR** — MS+SAR fusion (2 MS bands + 1 SAR band per member)

---

## Shared Utilities

All utilities shared across ensemble types are located in [resnet18_ensembles/](resnet18_ensembles/):

### make_tables_from_h5.m

Generates index lookup tables from HDF5 datasets for efficient on-demand data loading.

**Input:**
- Path to folder containing `training.h5` and `testing.h5`

**Output:**
- `tables_MS.mat` — Contains `train_MS`, `test_MS` tables
- `tables_SAR.mat` — Contains `train_SAR`, `test_SAR` tables

**Table Format:**

| Column | Type | Description |
|:-------|:-----|:------------|
| Path | string | Absolute path to HDF5 file |
| Label | categorical | LCZ class (1–17) |
| Index | double | 1-based position in H5 array |
| Modality | string | `"MS"` (Sentinel-2) or `"SAR"` (Sentinel-1) |

**Usage (on cluster):**
```matlab
make_tables_from_h5('/ext/rambaldima/3271064/lcz42');
```

---

### h5_reader.m

Low-level utility to read individual patches from HDF5 files on-demand.

**Input:**
- `path`: HDF5 file path
- `index`: 1-based position in dataset
- `modality`: `"MS"` or `"SAR"`

**Output:**
- Single patch as `[32, 32, C]` single precision where:
  - C = 10 for MS (Sentinel-2 bands B2–B12)
  - C = 8 for SAR (Sentinel-1 VH/VV polarizations)

**Usage:**
```matlab
X = h5_reader("data/lcz42/training.h5", 12345, "MS");  % [32, 32, 10]
```

---

### DatasetReading.m

Builds PyTorch-style datastores from index tables with preprocessing pipeline.

**Input:** Configuration struct with:
- `trainTable`, `testTable`: Tables from `make_tables_from_h5`
- `useZscore`: Apply z-score normalization (bool, default: false)
- `useSARdespeckle`: Apply Lee despeckling to SAR (bool, default: false)
- `useAugmentation`: Random geometric augmentation on training (bool, default: false)
- `reader.customFcn`: Function handle, typically `@(row) h5_reader(row.Path, row.Index, row.Modality)`

**Output:**
- `dsTrain`: Training datastore with augmentation
- `dsTest`: Test datastore (no augmentation)
- `info`: Metadata struct with classes, normalization stats

**Preprocessing Pipeline:**
1. **Scaling** — MS: [0, 2.8] → [0, 255]; SAR: clip + scale to [0, 255]
2. **Despeckling** — Optional Lee filtering for SAR
3. **Normalization** — Optional per-channel z-score
4. **Resizing** — To network input dimensions (usually 224×224)
5. **Augmentation** — Random geometric transforms (training only)

**Usage:**
```matlab
cfg.trainTable = train_MS;
cfg.testTable = test_MS;
cfg.useZscore = true;
cfg.useAugmentation = true;
cfg.reader.customFcn = @(row) h5_reader(row.Path, row.Index, row.Modality);
[dsTrain, dsTest, info] = DatasetReading(cfg);
```

---

### EnableGPU.m

Detects and initializes GPU acceleration (if available).

**Usage:**
```matlab
EnableGPU(1);  % Enable GPU with verbosity
```

---

## DenseNet-201 Ensembles

Located in [densenet201_ensembles/](densenet201_ensembles/). Uses pretrained DenseNet-201 from `densenet201_pretrained.mat`.

### train_teachers.m

Main orchestrator for DenseNet-201 ensemble training (unmodified architecture).

**Usage:**
```matlab
train_teachers('RAND')      % Train RAND ensemble
train_teachers('RANDRGB')   % Train RANDRGB ensemble
train_teachers('SAR')       % Train SAR ensemble
train_teachers('ALL')       % Train all three (sequential)
```

**Configuration:**
- Epochs: 12
- Batch size: 128
- Learning rate: 1e-3
- Z-score normalization: enabled
- SAR despeckling: enabled
- Augmentation: enabled

**Output:**
- `resRand.mat` — RAND ensemble results
- `resRandRGB.mat` — RANDRGB ensemble results
- `resSAR.mat` — SAR ensemble results

---

### train_teachers_v2.m

Updated orchestrator with pretrained weights from `densenet201_pretrained.mat`.

**Differences from v1:**
- Loads network from `.mat` file instead of calling `densenet201('Weights','none')`
- Supports fine-tuning of pretrained weights

**Usage:** (same as v1)
```matlab
train_teachers_v2('ALL')
```

---

### Rand_DenseNet.m / Rand_DenseNet_v2.m

Train DenseNet-201 ensemble on **3 random MS bands** per member.

**Configuration (Rand_DenseNet_v2.m):**
- Members: 10
- Epochs: 12
- Batch size: 128
- LR: 1e-3
- RNG seed: 1337

**Key Details:**
- Each member independently selects 3 random MS bands
- Outputs ensemble statistics: per-member accuracy, confusion matrix, average softmax scores

**Output Fields:**
```matlab
res.members(m).net        % Trained network
res.members(m).bands      % Selected band indices
res.members(m).valAcc     % Validation accuracy
res.testTop1              % Ensemble test accuracy
res.confusionMat          % Confusion matrix (17×17)
res.scoresAvg             % Averaged softmax scores
```

---

### RandRGB_DenseNet.m / RandRGB_DenseNet_v2.m

Train DenseNet-201 ensemble on **RGB-constrained bands** (B4, B3, B2) + optional random augmentation.

**Differences from Rand:**
- Fixed RGB bands (B4, B3, B2) instead of random selection
- Better interpretability for visualization

---

### ensembleSARchannel_DenseNet.m / ensembleSARchannel_DenseNet_v2.m

Train DenseNet-201 ensemble on **3 random SAR bands** per member.

**Key Details:**
- Selects 3 bands from 8 available SAR channels
- Applies despeckling during preprocessing

---

## ResNet-18 Ensembles

Located in [resnet18_ensembles/](resnet18_ensembles/). Lightweight architecture for edge deployment, loaded from `resnet18_pretrained.mat`.

### train_teacher_ResNet18.m

Main orchestrator for ResNet-18 ensemble training.

**Usage:**
```matlab
train_teacher_ResNet18('RAND')      % Train RAND ensemble
train_teacher_ResNet18('RANDRGB')   % Train RANDRGB ensemble
train_teacher_ResNet18('SAR')       % Train SAR ensemble
train_teacher_ResNet18('ALL')       % Train all three (default)
```

**Configuration:**
- Epochs: 10
- Batch size: 512 (larger for smaller model)
- Learning rate: 1e-3
- Z-score normalization: enabled
- SAR despeckling: enabled
- Augmentation: enabled

**Output:**
- `resRand_resnet18.mat` — RAND ensemble results
- `resRandRGB_resnet18.mat` — RANDRGB ensemble results
- `resSAR_resnet18.mat` — SAR ensemble results

---

### Rand_ResNet18.m

Train ResNet-18 ensemble on **3 random MS bands** per member.

**Configuration:**
- Members: 10
- Epochs: 10
- Batch size: 512
- LR: 1e-3
- RNG seed: 1337

**Lightweight advantages:**
- 12M parameters vs. 60M for DenseNet-201
- Suitable for edge inference on Axelera Metis
- Faster training and inference

---

### RandRGB_ResNet18.m

Train ResNet-18 ensemble on **RGB-constrained bands** (B4, B3, B2).

**Same structure as Rand_ResNet18 with fixed RGB band selection.**

---

### randSAR_ResNet18.m

Train ResNet-18 ensemble on **MS+SAR fusion** (2 MS + 1 SAR band per member).

**Key Details:**
- Concatenates 2 random MS bands + 1 random SAR band into 3-channel input
- Exploits ResNet-18's 3-channel architecture design

---

## Ensemble Architecture Comparison

| Aspect | DenseNet-201 | ResNet-18 |
|:-------|:------------|:----------|
| **Parameters** | 60M | 12M |
| **Epochs** | 12 | 10 |
| **Batch size** | 128 | 512 |
| **Use case** | High-accuracy teachers | Edge deployment |
| **Training time** | ~4-5 days | ~2-3 days |

---

## Complete Training Workflow

**Step 1: Prepare HDF5 tables**
```matlab
make_tables_from_h5('../../data/lcz42');
```

**Step 2: Train DenseNet-201 ensembles (recommended for high accuracy)**
```matlab
train_teachers_v2('ALL');  % Trains RAND, RANDRGB, SAR sequentially
```

**Step 3: (Alternative) Train ResNet-18 ensembles (for edge deployment)**
```matlab
train_teacher_ResNet18('ALL');
```

**Step 4: Verify outputs**
```matlab
load('matlab/resRand.mat');
disp(resRand.testTop1);  % Ensemble accuracy
disp(size(resRand.members));  % Number of members
```

---

## Cluster Considerations

**NFS I/O Optimization:**
- `DatasetReading.m` uses on-demand HDF5 reading to minimize memory footprint
- Large batch sizes (512) help amortize I/O overhead
- Consider reducing NUM_WORKERS on NFS with many concurrent jobs

**GPU Memory:**
- DenseNet-201: ~6–8 GB for batch=128
- ResNet-18: ~4–5 GB for batch=512
- Adjust batch sizes if OOM errors occur

---

## Citation

If you use this MATLAB pipeline in your research, please cite:

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