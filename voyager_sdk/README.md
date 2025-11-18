# Voyager SDK: Edge AI Deployment on Axelera Metis

This directory contains utilities and configurations for deploying the distilled ResNet18 student models on the **Axelera Metis AI accelerator** using the **Voyager SDK**.

---

## Overview

**Axelera Metis** is a dedicated edge AI inference accelerator optimized for:
- ✅ Real-time inference with <100ms latency
- ✅ Low power consumption (< 5W)
- ✅ INT8 quantization with minimal accuracy loss
- ✅ Support for ONNX models (Opset 17+)

This directory bridges the gap between trained PyTorch models and production deployment on Metis hardware.

---

## Directory Structure

```
voyager_sdk/
├── eval_lcz42_png.py                      # Evaluate student on PNG dataset
├── prepare_lcz42_voyager_deployment.py    # Convert HDF5 → PNG (Voyager format)
├── resnet18-imagenet-onnx.yaml            # Axelera Metis deployment config
├── data/LCZ42/
│   ├── calibration/                       # INT8 quantization calibration set (510 images)
│   ├── validation/                        # Validation set (3,400 images)
│   └── labels.txt                         # 17 LCZ class names
└── README.md
```

---

## Core Scripts

### eval_lcz42_png.py

**Purpose**: Evaluate the distilled ResNet18 student on PNG-format dataset (Voyager compatible).

**Features**:
- Loads PNG images from `data/LCZ42/validation/`
- Applies Z-score normalization (μ/σ computed during training)
- Computes per-class accuracy, precision, recall, F1-score
- Generates confusion matrix

**Usage**:
```bash
python eval_lcz42_png.py
```

**Expected Output**:
```
[INFO] Loading PNG dataset from data/LCZ42/validation/
[INFO] Found 3,400 validation images (200/class)
[INFO] Using device: cuda
[INFO] Loading student checkpoint: ../distillation/checkpoints/resnet18_to_resnet18/student_resnet18_last.pth
[EVAL] Processing class 01_CompactHighRise... 200/200
[EVAL] Processing class 02_CompactMidRise... 200/200
...
[RESULT] Test accuracy: 63.84% (2183/3400)
[RESULT] Per-class metrics saved to results.csv
```

---

### prepare_lcz42_voyager_deployment.py

**Purpose**: Convert HDF5 dataset to PNG format (Voyager SDK compatible).

**Pipeline**:
1. Reads HDF5 files (`training.h5`, `testing.h5`)
2. Applies Z-score normalization (all 10 channels)
3. Selects RGB bands (B4, B3, B2 = indices 2, 1, 0)
4. Rescales to [0, 255] uint8 range for PNG storage
5. Organizes into class folders: `01_CompactHighRise/`, `02_CompactMidRise/`, etc.

**Dataset Split**:
- **Calibration**: 30 images/class (510 total) for INT8 quantization
- **Validation**: 200 images/class (3,400 total) for accuracy testing

**Usage**:
```bash
python prepare_lcz42_voyager_deployment.py
```

**Output Structure**:
```
data/LCZ42/
├── calibration/
│   ├── class_01_00000.png      # Format: class_XX_YYYYY.png
│   ├── class_01_00001.png
│   └── ... (30/class, 510 total)
├── validation/
│   ├── 01_CompactHighRise/
│   │   ├── 000001.png
│   │   ├── 000002.png
│   │   └── ... (200/class)
│   ├── 02_CompactMidRise/
│   │   └── ... (200/class)
│   └── ... (17 classes)
└── labels.txt                  # 17 LCZ class names (one per line)
```

---

## Deployment Workflow

### Step 1: Prepare Dataset

```bash
python prepare_lcz42_voyager_deployment.py
```

This generates PNG files in Voyager format with proper normalization and class organization.

### Step 2: Evaluate on PNG

```bash
python eval_lcz42_png.py
```

This verifies that PNG conversion didn't introduce artifacts and baseline accuracy is maintained.

### Step 3: Export Student to ONNX

From the `distillation/` folder:

```bash
cd ../distillation
python export_student_resnet18.py \
  --checkpoint checkpoints/resnet18_to_resnet18/student_resnet18_best.pth \
  --output checkpoints/resnet18_to_resnet18/student_resnet18_rgb.onnx
```

### Step 4: Deploy to Axelera Metis

Copy the ONNX model and configuration to Voyager SDK:

```bash
# Copy ONNX model
cp ../distillation/checkpoints/resnet18_to_resnet18/student_resnet18_rgb.onnx \
   customers/mymodels/student_resnet18_rgb.onnx

# Copy deployment config
# The resnet18-imagenet-onnx.yaml file specifies:
# - Input shape: 1 × 3 × 224 × 224
# - INT8 quantization with calibration set
# - Batch inference for real-time performance
```

---

## Configuration Details

### resnet18-imagenet-onnx.yaml

**Key Parameters**:

```yaml
input_shape: [1, 3, 224, 224]         # Batch × RGB × Height × Width
opset_version: 17                      # ONNX opset (Metis requirement)
quantization: INT8                     # 8-bit quantization for efficiency
calibration_set: "data/LCZ42/calibration/"  # INT8 calibration images
preprocessing:
  normalization: z-score               # Per-channel normalization
  mean: [11.27, 9.95, 9.21]            # RGB channel means (B4, B3, B2)
  std: [3.61, 4.35, 6.04]              # RGB channel std devs
  scale_to_255: true                   # Rescale [0,2.8] → [0,255]
```

### Normalization Details

The preprocessing pipeline ensures consistency across training, evaluation, and deployment:

1. **Training/Validation** (HDF5):
   - Compute μ/σ from all 10 Sentinel-2 channels (training set)
   - Apply z-score: `x_norm = (x - μ) / (σ + ε)`

2. **PNG Conversion**:
   - Same μ/σ applied to original HDF5 images
   - Extract RGB (indices 2, 1, 0) from normalized tensor
   - Rescale to [0, 255] for uint8 PNG storage

3. **Inference**:
   - Load PNG as uint8
   - Rescale back to [0, 1]
   - Apply z-score normalization with same μ/σ
   - Input to model

---

## Expected Results

| Model | Dataset | Top-1 Acc | Precision | Recall | F1-score |
|-------|---------|-----------|-----------|--------|----------|
| ResNet18 (HDF5) | Training | 99.2% | — | — | — |
| ResNet18 (HDF5) | Test (24K) | 63.84% | 64.2% | 63.8% | 62.9% |
| ResNet18 (PNG) | Validation (3.4K) | ~63.8% | ~64.2% | ~63.8% | ~62.9% |

**Note**: PNG results should be **within ±0.5%** of HDF5 due to discretization to [0, 255].

---

## Hardware Specifications (Axelera Metis)

| Metric | Value |
|--------|-------|
| Peak Throughput | 500 TOPS (INT8) |
| Peak Power | 4.8W |
| Latency (Batch-1) | <50ms |
| Latency (Batch-32) | <1ms per sample |
| Memory | 4 GB GDDR6 |
| Supported Precisions | INT8, INT16, BF16, FP32 |

---

## Troubleshooting

### Issue: PNG images look wrong (grayscale or inverted)

**Cause**: Rescaling error during PNG conversion
**Solution**: Verify μ/σ values match training dataset, check [0, 2.8] → [0, 255] scaling

### Issue: Accuracy drop > 1% on PNG vs HDF5

**Cause**: Normalization inconsistency or rounding errors
**Solution**:
- Verify same μ/σ used in both pipelines
- Check that PNG rescaling uses 255.0 (not 256)
- Ensure RGB indices are (2, 1, 0) not (0, 1, 2)

### Issue: Axelera Metis calibration fails

**Cause**: Insufficient or unrepresentative calibration images
**Solution**: Ensure 30 images/class evenly distributed across LCZ types

---

## Integration with Voyager SDK

The Voyager SDK provides:
- ONNX model loading and optimization
- INT8 quantization with calibration
- Multi-threaded batch inference
- Hardware-specific operator fusion

**Required Files**:
1. `student_resnet18_rgb.onnx` — Trained model
2. `resnet18-imagenet-onnx.yaml` — Deployment config
3. `data/LCZ42/calibration/` — Quantization calibration set

---

## Citation

If you use this deployment pipeline, please cite:

```bibtex
@mastersthesis{rambaldi2025enhancedlcz42,
  title        = {Enhanced LCZ42 Ensembles to Edge AI: Knowledge Distillation and Deployment on Axelera Metis},
  author       = {Rambaldi, Matteo},
  school       = {University of Padua},
  year         = {2025},
  note         = {GitHub Repository: https://github.com/matteorambaldi/Enhanced-LCZ42-Ensembles-to-EdgeAI-Axelera-Metis}
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
**Edge AI Platform:** Axelera Metis
