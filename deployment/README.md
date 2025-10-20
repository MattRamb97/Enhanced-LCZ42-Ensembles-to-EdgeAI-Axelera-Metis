# Deployment and EdgeAI Integration

This module contains all scripts used for **exporting trained LCZ42 ensemble models** to the **ONNX format** and preparing them for deployment on the **Axelera Metis Edge AI accelerator**.

## Contents

| File | Description |
|:-----|:-------------|
| **`Export_ONNX.m`** | Core MATLAB utility that exports trained `dlnetwork` models to ONNX (Opset 17). Automatically generates `preprocessing.json` and `labels.txt` with metadata and class mappings. |
| **`run_export.m`** | Example script to export all ensemble members (`resRand`, `resRandRGB`, `resSAR`) to ONNX format and store results in `deployment/onnx/`. |
| **`SelfTest_ONNX.m`** | Lightweight verification script to import exported ONNX models back into MATLAB and ensure integrity of layers and weights. |

## Usage

### Export models to ONNX

```matlab
addpath(genpath('matlab')); 
addpath('deployment');

load matlab/resRand.mat
load matlab/resRandRGB.mat
load matlab/resSAR.mat

opts.outDir = "deployment/onnx";
opts.modelNames = {'dense_rand','dense_randrgb','dense_sar'};

Export_ONNX({resRand, resRandRGB, resSAR}, opts);
```

This produces:

```bash
deployment/onnx/
├── dense_rand.onnx
├── dense_randrgb.onnx
├── dense_sar.onnx
├── preprocessing.json
└── labels.txt
```

### Verify exported models

```matlab
SelfTest_ONNX("deployment/onnx/dense_rand.onnx")
```

This performs a round-trip check to confirm that:
- The ONNX model is readable
- Layer structure matches the original dlnetwork
- Weights are correctly embedded

## Metadata Description

Export_ONNX.m automatically creates a preprocessing.json file containing:

```json
{
  "input": {"layout": "NCHW", "dtype": "float32", "shape": [1, 3, 32, 32]},
  "scaling": {
    "optical": {"type": "linear", "src_range": [0, 2.8], "dst_range": [0, 255]},
    "sar": {"type": "clip_linear", "clip": [-0.5, 0.5], "dst_range": [0, 255]}
  },
  "zscore": {"enabled": true, "mu": [...], "sigma": [...]},
  "labels": ["LCZ_1", "LCZ_2", ..., "LCZ_17"]
}
```

This metadata ensures consistent preprocessing between MATLAB training, Python inference, and EdgeAI deployment.

## Maintainer

**Matteo Rambaldi** — University of Padua  •  MSc Artificial Intelligence and Robotics (2025)