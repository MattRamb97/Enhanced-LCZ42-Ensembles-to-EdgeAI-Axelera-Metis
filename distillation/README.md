# Knowledge Distillation: From Teacher Ensemble to Student Model

This module implements **knowledge distillation (KD)** to compress large LCZ42 ensembles (trained in MATLAB and exported to ONNX) into compact, deployable student networks for **embedded inference** on the Axelera Metis accelerator.

## Overview

| Component | Purpose |
|:-----------|:--------|
| **Teacher (ONNX)** | Ensemble of high-accuracy DenseNet teachers exported via `deployment/Export_ONNX.m`. |
| **Student (PyTorch)** | Lightweight CNN or Transformer trained to mimic the teacher ensemble’s softened predictions. |
| **Distillation Objective** | Combines *hard cross-entropy* with *soft KL divergence* from teacher logits. |
| **Output** | Quantization-ready ONNX model (e.g. MobileNetV3 or MobileNetV4) for EdgeAI deployment. |

## Files

| File | Description |
|:-----|:-------------|
| **`distill_student.py`** | Baseline KD training using **MobileNetV3-Small** at 32×32 input resolution. |
| **`distill_student_v4.py`** | Advanced KD training using **MobileNetV4-Medium** (via `timm`, 224×224). Adds z-score normalization and ONNX export for Metis. |

## Workflow

```bash
Teacher (DenseNet ensemble, MATLAB)
        │
        ▼
Export to ONNX  →  deployment/onnx/*.onnx
        │
        ▼
distillation/distill_student_v4.py
        │
        ▼
Student (MobileNetV4-Medium)
        │
        ▼
Exported ONNX: deployment/student_mbv4m_224.onnx
```

### Core Idea — Knowledge Distillation

The training objective is:

```math
L = \alpha , CE(y_s, y) ;+; (1 - \alpha), T^2 , KL(p_s^{(T)} || p_t^{(T)})
```

Where:
- ( CE ): cross-entropy with ground-truth labels
- ( KL ): Kullback–Leibler divergence between teacher & student logits
- ( T ): temperature (softens distributions)
- ( \alpha ): weighting factor (default = 0.7)

Implemented in DistillLoss class:

```python
class DistillLoss(nn.Module):
    def __init__(self, T=2.0, alpha=0.7):
        ...
```

### Teacher: ONNX Ensemble Wrapper

Teachers are loaded from exported ONNX models (deployment/onnx/*.onnx)
and averaged at logit level:

```python
teacher = ONNXTeacher(list(Path("deployment/onnx").glob("dense_*.onnx")))
logits = teacher.predict_logits(x_np)  # returns mean logits across ensemble
```

### Student Architectures

| Model | Framework | Input | Params | Notes |
|:-----------|:--------|:--------|:--------|:--------|
| **MobileNetV3-Small** | TorchVision | 32×32 | ~2.5 M | Fast baseline |
| **MobileNetV4-Medium** | TorchVision | 224×224 | ~5 M | Better accuracy / distillation fidelity |
| **EfficientNet-B0** | TorchVision | 224×224 | ~5.3 M | Fallback if TIMM unavailable |


```python
student = build_student(num_classes, arch="mobilenetv4m")
```

### Dataset & Preprocessing

Each image is preprocessed according to the exported preprocessing.json:

```json
{
  "scaling": {"optical": {"src_range": [0, 2.8], "dst_range": [0, 255]}},
  "zscore": {"enabled": true, "mu": [...], "sigma": [...]}
}
```

The dataset class applies resizing and normalization:

```python
transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mu, std=sd)
])
```

### Training Example

```python
from distill_student_v4 import *

onnx_models = list(Path("deployment/onnx").glob("dense_*.onnx"))
teacher = ONNXTeacher(onnx_models)

train_ds = LCZ42Dataset(train_paths, train_labels, "deployment/onnx/preprocessing.json")
val_ds   = LCZ42Dataset(val_paths, val_labels,   "deployment/onnx/preprocessing.json")

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=128)

student = build_student(num_classes=len(train_ds.meta["labels"]), arch="mobilenetv4m")
student = train_student(student, teacher, train_loader, val_loader, num_classes=17, epochs=30, lr=3e-4)
export_student(student, "deployment/student_mbv4m_224.onnx")
```

### Export for Deployment

The final student is exported to ONNX (Opset 17, NCHW layout):

```python
export_student(student, "deployment/student_mbv4m_224.onnx")
```

Output:

```bash
deployment/
├── student_mbv4m_224.onnx
├── onnx/
│   ├── dense_rand.onnx
│   ├── dense_randrgb.onnx
│   ├── dense_sar.onnx
│   ├── preprocessing.json
│   └── labels.txt
```

## Maintainer

**Matteo Rambaldi** — University of Padua  •  MSc Artificial Intelligence and Robotics (2025)