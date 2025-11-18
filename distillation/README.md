# Knowledge Distillation: Teacher Ensembles → Compact Student Models

This directory contains the complete **knowledge distillation (KD) pipeline** for compressing large pretrained ensembles (DenseNet201 from `densenet201_ensembles/`, ResNet18 from `resnet18_ensembles/`) into efficient student networks suitable for **embedded inference** on resource-constrained hardware (Axelera Metis).

---

## Overview

**Knowledge Distillation** transfers learned representations from a high-capacity teacher model (or ensemble) to a lightweight student model. The student learns to mimic the teacher's soft probability distributions while maintaining similar classification accuracy with dramatically reduced parameters and computation.

### Teacher-to-Student Pathways

| Teacher | Student | Teacher Params | Student Params | Compression |
|---------|---------|---|---|---|
| **RAND DenseNet201 (10-member ensemble)** | **ResNet18** | 60M × 10 | 12M | **50×** |
| **RAND ResNet18 (10-member ensemble)** | **ResNet18** | 12M × 10 | 12M | **10×** |

Both pipelines are fully implemented:
- `distill_resnet18_student.py`: ResNet18 ensemble → ResNet18 student (baseline)
- `distill_densenet201_student.py`: DenseNet201 ensemble → ResNet18 student (aggressive compression)

---

## Directory Structure

```
distillation/
├── distill_resnet18_student.py          # KD training: ResNet18 ensemble → ResNet18 student
├── distill_densenet201_student.py       # KD training: DenseNet201 ensemble → ResNet18 student
├── eval_resnet18_student.py             # Evaluate ResNet18 student (flexible --checkpoint arg)
├── export_student_resnet18.py           # Export ResNet18 student to ONNX (flexible args)
├── checkpoints/
│   ├── resnet18_to_resnet18/            # Checkpoints for ResNet18→ResNet18 distillation
│   │   ├── student_resnet18_best.pth
│   │   ├── student_resnet18_last.pth
│   │   ├── student_resnet18_epoch*.pth
│   │   └── kd_history.json
│   └── densenet201_to_resnet18/         # Checkpoints for DenseNet201→ResNet18 distillation
│       ├── student_resnet18_from_densenet201_best.pth
│       ├── student_resnet18_from_densenet201_last.pth
│       ├── student_resnet18_from_densenet201_epoch*.pth
│       └── kd_history_densenet201_resnet18.json
└── README.md
```

---

## Core Scripts

### Training Scripts

#### distill_resnet18_student.py

**Purpose**: Distill a 10-member RAND ResNet18 ensemble into a single ResNet18 student via knowledge distillation.

**Key Components**:
- **Teacher**: `RandResNet18Teacher` — loads 10 trained ResNet18 members and averages their logits
- **Student**: Standard `resnet18()` from torchvision (NOT wrapped)
- **Loss**: Hybrid KD loss = α·CE(student, labels) + (1-α)·T²·KL(student_soft || teacher_soft)

**Configuration (dataclass KDConfig, lines 231–245)**:

| Parameter | Value | Notes |
|-----------|-------|-------|
| **epochs** | 30 | Training epochs |
| **batch_size** | 512 | Large batch size for stable gradient estimation |
| **lr** | 2e-4 | Conservative LR for distillation (soft targets require precision) |
| **weight_decay** | 1e-4 | L2 regularization |
| **temperature** | 3.0 | Soft probability temperature (softens distributions for knowledge transfer) |
| **alpha** | 0.7 | Hard loss weight: 0.7·CE + 0.3·KL |
| **num_workers** | 8 | Data loader threads |
| **seed** | 42 | Reproducibility |
| **student_pretrained** | False | Initialize student **without** ImageNet pretrained (train from scratch) |
| **use_sar_despeckle** | False | SAR despeckling disabled for RGB-only training |

**Architecture**:

```
Teacher (RAND ResNet18 Ensemble, 10 members)
        │ (each member processes 3 RGB bands B4/B3/B2)
        ├── Member 1 logits (B, 17)
        ├── Member 2 logits (B, 17)
        ...
        ├── Member 10 logits (B, 17)
        └── Average logits (B, 17) → Soft targets (temperature-scaled)
                │
                ▼
    Distillation Loss
                │
                ▼
    Student (Single ResNet18)
        │ (processes 3 RGB bands B4/B3/B2)
        ├── Backbone: conv1 → layer1-4 → avgpool
        └── Classifier: fc (512 → 17)
```

**Output**:
- Checkpoint: `checkpoints/resnet18_to_resnet18/student_resnet18_last.pth`
- Metrics: Per-epoch train/val loss and accuracy logged to stdout

---

#### distill_densenet201_student.py

**Purpose**: Distill a 10-member RAND DenseNet201 ensemble into a single ResNet18 student (aggressive compression: 60M→12M).

**Key Differences from distill_resnet18_student.py**:

| Aspect | ResNet18→ResNet18 | DenseNet201→ResNet18 |
|--------|---|---|
| **Teacher Architecture** | ResNet18 (12M × 10) | DenseNet201 (60M × 10) |
| **Student Architecture** | ResNet18 (12M) | ResNet18 (12M) |
| **Compression Ratio** | 10× | 50× |
| **Temperature** | 3.0 | 3.0 |
| **Training Time** | ~30 epochs × 45 min/epoch | ~30 epochs × 90 min/epoch |
| **Expected Accuracy** | ~63.8% | ~60-62% (estimated) |

**Configuration (dataclass KDConfig, lines 231–245)**:

Identical to distill_resnet18_student.py:
- **epochs**: 30
- **batch_size**: 512 (with num_workers=4, reduced from 8 to avoid OOM with DenseNet201 teacher)
- **temperature**: 3.0
- **alpha**: 0.7
- **lr**: 2e-4
- **weight_decay**: 1e-4

**Teacher Architecture** (lines 94–120):

```python
class RandDenseNet201Teacher(nn.Module):
    def __init__(self, num_members: int, num_classes: int):
        self.models = nn.ModuleList([DenseNet201MS(...) for _ in range(num_members)])
        self.bands_matrix = torch.zeros(num_members, 3)  # 3 RGB bands per member

    def forward(self, x: torch.Tensor):
        # For each member: select 3 RGB bands, pass through DenseNet201, average logits
        logits_sum = None
        for model, band_idx in zip(self.models, self.bands_matrix):
            subset = torch.index_select(x, dim=1, index=band_idx)
            logits = model(subset)
            logits_sum = logits if logits_sum is None else logits_sum + logits
        return logits_sum / len(self.models)
```

**Student Architecture** (lines 222–223):

```python
student = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if USE_STUDENT_PRETRAINED else None)
student.fc = nn.Linear(student.fc.in_features, NUM_CLASSES)
```

**Critical Note**: Standard `resnet18()` from torchvision, NOT wrapped in Sequential. This ensures:
- Clean layer naming (layer1-4, not features.0-N) for checkpoint compatibility
- Seamless ONNX export without layer name mismatches
- Consistent evaluation in eval_densenet201_student.py

**Output**:
- Best checkpoint: `checkpoints/densenet201_to_resnet18/student_resnet18_from_densenet201_best.pth`
- Last checkpoint: `checkpoints/densenet201_to_resnet18/student_resnet18_from_densenet201_last.pth`
- Training curves saved to stdout

---

### Evaluation Scripts

#### eval_resnet18_student.py

**Purpose**: Evaluate ResNet18 student on the complete HDF5 test set (works for ANY ResNet18 student checkpoint).

**Configuration**:
- **Default checkpoint**: `checkpoints/resnet18_to_resnet18/student_resnet18_last.pth`
- **Input**: HDF5 test set with z-score normalization (μ/σ computed from training data)
- **Output**: Test accuracy + per-sample predictions

**Features**:
- **Argparse support**: `--checkpoint <path>` to evaluate any checkpoint
- Works for both ResNet18→ResNet18 and DenseNet201→ResNet18 students
- Reuses DatasetReading utilities for consistent preprocessing

**Usage**:
```bash
# Evaluate ResNet18→ResNet18 student (default)
python eval_resnet18_student.py

# Evaluate DenseNet201→ResNet18 student
python eval_resnet18_student.py \
  --checkpoint checkpoints/densenet201_to_resnet18/student_resnet18_from_densenet201_best.pth
```

---

### Export Scripts

#### export_student_resnet18.py

**Purpose**: Export ResNet18 student to ONNX format for Axelera Metis deployment (works for ANY checkpoint).

**Configuration (defaults, overridable via argparse)**:
- **Default checkpoint**: `checkpoints/resnet18_to_resnet18/student_resnet18_last.pth`
- **Default output**: `checkpoints/resnet18_to_resnet18/student_resnet18_rgb.onnx`
- **Input shape**: (1, 3, 224, 224) — batch=1, RGB=3, spatial=224×224
- **OPSET_VERSION**: 17 (Axelera Metis requirement)

**Features**:
- **Argparse support**: `--checkpoint <path> --output <path>`
- Works for both ResNet18→ResNet18 and DenseNet201→ResNet18 students
- ONNX opset 17 compatible with Axelera Metis accelerator

**Usage**:
```bash
# Export ResNet18→ResNet18 student (default)
python export_student_resnet18.py

# Export DenseNet201→ResNet18 student
python export_student_resnet18.py \
  --checkpoint checkpoints/densenet201_to_resnet18/student_resnet18_from_densenet201_best.pth \
  --output checkpoints/densenet201_to_resnet18/student_resnet18_from_densenet201_rgb.onnx
```

**Output**:
- ONNX file with:
  - Input name: `input` (shape: batch × 3 × 224 × 224)
  - Output name: `logits` (shape: batch × 17)
  - Dynamic batch axis (supports variable batch sizes)

---

## Knowledge Distillation Theory

### Loss Function

The hybrid KD loss combines hard (ground-truth) and soft (teacher-mimicking) objectives:

```math
L = α · L_{CE} (ŷ_s, y) + (1 - α) · T^{2} · L_{KL} (\sigma(z_s/T), \sigma(z_{t }/ T))
```

Where:
- **L_CE**: Cross-entropy loss with ground-truth labels `y`
- **L_KL**: Kullback-Leibler divergence between student and teacher soft targets
- **T**: Temperature (3.0 default) — higher T softens the probability distributions
- **α**: Hard loss weight (0.7) — balance between matching labels and mimicking teacher
- **z_s, z_t**: Student and teacher logits (before softmax)
- **σ**: Softmax function

**Intuition**:
- **Hard loss** (α=0.7): Ensures student learns correct labels
- **Soft loss** (1-α=0.3, T²-weighted): Captures rich information in teacher's probability distribution over wrong classes
- **Temperature**: T=3.0 provides balance between soft targets diffusion and knowledge transfer (too high → targets too uniform, too low → targets like hard labels)

### Why Knowledge Distillation Works

1. **Dark knowledge**: Teacher's soft predictions encode implicit relationships between classes (e.g., similar LCZ types produce similar teacher predictions)
2. **Rich supervision**: Student learns from 17-class soft targets, not just one-hot hard labels
3. **Efficient transfer**: Highly engineered architectures (DenseNet, ResNet) provide good inductive bias; student adapts this quickly

---

## Data & Normalization

Both training and evaluation use **z-score normalization** computed dynamically from the training set via `DatasetReading`:

**Normalization Formula**:
```math
x_{normalized} = (x - μ) / (σ + ε)
```
where ε = 1e-6

**Computed μ/σ (all 10 Sentinel-2 channels, from training data)**:

| Channel | μ (mean) | σ (std) |
|---------|----------|---------|
| B2 | 11.271 | 3.605 |
| B3 | 9.952 | 4.351 |
| B4 | 9.206 | 6.044 |
| B5 | 10.404 | 5.791 |
| B6 | 14.505 | 7.053 |
| B7 | 16.527 | 8.289 |
| B8 | 15.899 | 8.395 |
| B8A | 17.760 | 9.257 |
| B11 | 14.051 | 9.100 |
| B12 | 9.931 | 7.997 |

**Pipeline**:
1. DatasetReading computes μ/σ from **all 10 channels** during initialization
2. Each sample is normalized across all 10 channels
3. eval_resnet18_student.py extracts RGB (indices 2,1,0 = B4/B3/B2) from normalized tensor
4. **Key point**: Normalization happens **before** RGB extraction → correct statistics

---

## Hyperparameter Tuning

| Parameter | Effect | Suggested Range |
|-----------|--------|---|
| **TEMPERATURE** | Higher → softer distributions → more knowledge transfer | 2.0–8.0 |
| **ALPHA** | Higher → more hard loss weight | 0.5–0.9 |
| **LEARNING_RATE** | Convergence speed and final accuracy | 1e-4 to 1e-3 |
| **BATCH_SIZE** | Gradient estimation quality (larger → smoother) | 128–512 |
| **EPOCHS** | Total training time | 20–50 |
| **GRAD_CLIP** | Stability (clip if gradients explode) | 0.5–2.0 |

---

## Citation

If you use this knowledge distillation framework in your research, please cite:

```bibtex
@mastersthesis{rambaldi2025enhancedlcz42,
  title        = {Enhanced LCZ42 Ensembles to Edge AI: Knowledge Distillation and Deployment on Axelera Metis},
  author       = {Rambaldi, Matteo},
  school       = {University of Padua},
  year         = {2025},
  note         = {GitHub Repository: https://github.com/matteorambaldi/Enhanced-LCZ42-Ensembles-to-EdgeAI-Axelera-Metis}
}
```

**Key References**:
- Hinton, G. E., Vinyals, O., & Dean, J. (2015). "Distilling the Knowledge in a Neural Network." *arXiv preprint arXiv:1503.02531*.
- Gou, J., Yu, B., Maybank, S. J., & Tao, D. (2021). "Knowledge Distillation: A Good Teacher Is Patient." *arXiv preprint arXiv:2106.05237*.

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
