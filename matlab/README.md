# MATLAB Teacher Ensemble Training
This folder contains all MATLAB utilities used to train, validate, and export the *teacher ensembles* used in the Enhanced LCZ42 pipeline.

## Pipeline Overview

### EnableGPU.m

Utility to check and enable GPU usage. Automatically selects GPU if available.

###  h5_reader.m

Returns a single patch from `/sen1` or `/sen2` as `[H W C]` single precision.

**Input:**

- pth: Path to HDF5 file
- index: Patch index (1-based integer)
- modality: "MS" (Sentinel-2) or "SAR" (Sentinel-1)

**Output:** 

- X: Single patch as [32×32×C] array where C=10 for MS, C=8 for SAR

Sentinel-2 Bands (10 channels): B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12\
Sentinel-1 Bands (8 channels): VH imaginary, VV real, VV imaginary, VH Lee-filtered intensity, VV Lee-filtered intensity, Covariance off-diagonal real, Covariance off-diagonal imaginary

**Usage:**

```matlab
X = h5_reader(".../training.h5", 12345, "MS");  % 32x32x10
```

### make_tables_from_h5.m

Creates index tables from HDF5 files for training and testing.

**Important:** Validation set is ignored to prevent cross-domain contamination between training and test cities.

**Input:** Path to folder containing training.h5 and testing.h5

**Output:** 
- tables_MS.mat (train_MS, test_MS)
- tables_SAR.mat (train_SAR, test_SAR)

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| Path | string | Absolute path to source HDF5 file | `"/ext/.../training.h5"` |
| Label | categorical | LCZ class label (1-17) | `categorical(5)` |
| Index | double | Position in HDF5 array (1-based) | `12345` |
| Modality | string | Sensor type | `"MS"` or `"SAR"` |

**Usage (on cluster):**
```matlab
make_tables_from_h5('/ext/rambaldima/3271064/lcz42');
```

### DatasetReading.m

Builds PyTorch-style datastores from index tables with on-demand H5 reading and preprocessing.

**Input:** Configuration struct with:
- `trainTable`, `testTable`: Tables from make_tables_from_h5
- `useZscore`: Apply z-score normalization (default: false)
- `useSARdespeckle`: Apply despeckling to SAR (default: false)
- `useAugmentation`: Random geometric augmentation on training (default: false)
- `reader.customFcn`: Function handle to read H5 patches (e.g., @h5_reader)

**Output:**
- `dsTrain`: Training datastore with augmentation
- `dsTest`: Test datastore (no augmentation)
- `info`: Struct with classes, normalization stats, metadata

**Preprocessing Pipeline:**
1. Paper-faithful scaling: MS [0,2.8]→[0,255], SAR clip+scale to [0,255]
2. Optional SAR despeckling (Lee filter)
3. Optional z-score normalization per channel
4. Resize to network input size
5. Random geometric augmentation (training only)

**Usage:**
```matlab
cfg.trainTable = train_MS;
cfg.testTable = test_MS;
cfg.reader.customFcn = @(row) h5_reader(row.Path, row.Index, row.Modality);
[dsTr, dsTe, info] = DatasetReading(cfg);
```

### train_teachers.m

Main training orchestrator. Trains one or more teacher ensembles.

**Usage:**
```matlab
train_teachers('RAND')     % Train only multispectral random bands
train_teachers('RANDRGB')  % Train only RGB-constrained
train_teachers('SAR')      % Train only SAR
train_teachers('ALL')      % Train all three (sequential)
```
### Rand_DenseNet.m

Trains a DenseNet-201 ensemble using 3 randomly selected multispectral (MS) bands per member.

**Input:** Struct `cfgT` with:
- `dsTrain`, `dsTest`: Datastores from DatasetReading
- `info`: Struct with `classes` and metadata
- Optional: `numMembers`, `maxEpochs`, `miniBatchSize`, `learnRate`, `rngSeed`

**Output:** Struct with:
- `.members`, `.testTop1`, `.confusionMat`, `.yTrue`, `.yPred`, `.scoresAvg`

**Key Details:**
- Uses `densenet201('Weights','none')` to avoid requiring pretrained support packages on cluster.
- Each model is trained on a different random MS triplet.
- Softmax scores are averaged over ensemble.

### RandRGB_DenseNet.m

Variant of Rand_DenseNet constrained to RGB (B4, B3, B2). Same pipeline and structure, just fixed bands.

### ensembleSARchannel_DenseNet.m

Trains DenseNet-201 ensemble on 3 SAR channel combinations.

**Input:** Struct `cfgT` (same format)

**Output:** Struct with ensemble results

**Implementation:**
- Randomly selects 3 bands from SAR

### train_teachers_v2.m / train_teacher_ResNet18.m

Main training orchestrator for training all ensembles using the updated _v2 models or ResNet18 models.

**Usage:**
```matlab
train_teachers_v2('RAND')     % Train only multispectral random bands
train_teachers_v2('RANDRGB')  % Train only RGB-constrained
train_teachers_v2('SAR')      % Train MS+SAR (2+1) ensemble only
train_teachers_v2('ALL')      % Train all three (sequential)

or 

train_teacher_ResNet18('RAND')     % Train only multispectral random bands
train_teacher_ResNet18('RANDRGB')  % Train only RGB-constrained
train_teacher_ResNet18('SAR')      % Train MS+SAR (2+1) ensemble only
train_teacher_ResNet18('ALL')      % Train all three (sequential)
```
### Rand_DenseNet_v2.m / Rand_ResNet18.m

Updated version of `Rand_DenseNet.m` that supports **pretrained DenseNet-201** and **pretrained ResNet18** loaded from `.mat` files. Use when `densenet201_pretrained.mat` or `resnet18_pretrained.mat` is available and pretrained weights are desired.

**Differences from `Rand_DenseNet.m`:**
- Loads the network from `densenet201_pretrained.mat` or  `resnet18_pretrained.mat` instead of calling `densenet201('Weights','none')`
- Everything else remains unchanged: each model uses a different random MS triplet.

**Required:** 
 
`.mat` file `matlab/densenet201_pretrained.mat` or `resnet18_pretrained.mat`

### RandRGB_DenseNet_v2.m / RandRGB_ResNet18.m

Updated RGB-constrained ensemble with pretrained DenseNet-201 or ResNet18. Functionally identical to RandRGB_DenseNet.m, but uses pretrained weights.

### ensembleSARchannel_DenseNet_v2.m / randSAR_ResNet18.m

Trains DenseNet-201 or ResNet18 ensemble on 2 MS + 1 SAR channel combinations.

**Input:** Struct `cfgT` (same format)

**Output:** Struct with ensemble results

**Implementation:**
- Randomly selects 2 bands from MS and 1 from SAR
- Concatenates into a 3-channel input

### Summary

This MATLAB pipeline provides the teacher ensembles that serve as supervision for:
1. **Super-resolution models** (trained separately in `super-resolution/`).
2. **Knowledge distillation** (student models in `distillation/`).

All trained networks can be exported to ONNX format using `deployment/Export_ONNX.m`.

## Maintainer

**Matteo Rambaldi** — University of Padua  •  MSc Artificial Intelligence and Robotics (2025)