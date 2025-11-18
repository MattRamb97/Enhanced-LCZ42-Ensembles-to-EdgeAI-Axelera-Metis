# Fusion DenseNet201 + TDA (No SR) — LCZ42 Classification

This module trains **DenseNet201 + TDA fusion ensembles** on the So2Sat LCZ42 dataset using **only baseline (original 32×32) imagery, without any super-resolution enhancement**.

## Purpose

This is an **ablation study** designed to isolate the contribution of **Topological Data Analysis (TDA)** fusion from the effects of super-resolution. All models use the original 32×32 Sentinel patches without SR preprocessing.

## Key Features

- ✅ **No Super-Resolution**: Only baseline, original 32×32 imagery
- ✅ **TDA Fusion Enabled**: Combines CNN features with topological descriptors (H₀, H₁ homology)
- ✅ **Three Band Selection Modes**:
  - `RAND`: Three random Sentinel-2 bands (10 ensemble members)
  - `RANDRGB`: Two random MS bands + one fixed RGB band (B4/B3/B2)
  - `SAR`: Two random MS bands + one Sentinel-1 SAR band (MS + SAR fusion)
- ✅ **Ensemble Fusion**: Sum-rule averaging across members per mode and globally

---

## Architecture

### TDAFusionDenseNet201

A two-stream late fusion architecture:

```
Input: image (B, 3, H, W)  +  tda (B, 18000/14400)
       │                         │
       ├──→ DenseNet201 backbone │
       │    (1920-dim features)  │
       │                      TDA MLP:
       │                      {input → 1024 → 512 → 256 → 128}
       │                         │
       ├─── Concat ──→ (B, 2048)─┘
            │
            └──→ LayerNorm ──→ Dropout ──→ Classifier {256 → 17}
```

---

## Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| **SEED** | 42 | Reproducibility |
| **EPOCHS** | 12 | Per ensemble member |
| **BATCH_SIZE** | 128 | Reduced due to TDA memory overhead |
| **LEARNING_RATE** | 1e-3 | SGD with momentum |
| **WEIGHT_DECAY** | 1e-4 | L2 regularization |
| **LABEL_SMOOTHING** | 0.1 | Cross-entropy smoothing |
| **OPTIMIZER** | SGD(momentum=0.9) | Standard SGD |
| **SCHEDULER** | None | Constant LR |
| **NUM_WORKERS** | 6 | Parallel data loading |
| **USE_ZSCORE** | True | Z-score normalization |
| **USE_SAR_DESPECKLE** | True | Median filtering for SAR |
| **USE_AUGMENTATION** | True | Random rotations/flips/crops |

**Data Variants**: Only baseline (no SR):
- `baseline1`: Original 32×32 (352K train / 24K test)
- `baseline2`: Same as baseline1

---

## Directory Layout

```
fusion_ensembles_no_SR/
├── scripts/
│   ├── train_teacher_fusion.py    # Orchestrator (trains all modes, baseline only)
│   ├── rand_fusion.py             # FusionTDADataset & training loop
│   ├── fusion_densenet201.py      # TDAFusionDenseNet201 model
│   ├── dataset_reading.py         # HDF5 dataloader with z-score
│   └── utils_results.py           # HDF5 logging
├── models/
│   └── trained/                   # Saved checkpoints (.pth)
├── results/
│   ├── RAND/                      # RAND mode results
│   ├── RANDRGB/                   # RANDRGB mode results
│   ├── SAR/                       # SAR mode results
│   └── fusion/                    # Cross-mode fusion results
└── README.md
```

---

## Training

### Quick Start

```bash
cd scripts/

# Train all modes (RAND, RANDRGB, SAR)
python train_teacher_fusion.py --mode ALL

# Or individual modes
python train_teacher_fusion.py --mode RAND
python train_teacher_fusion.py --mode RANDRGB
python train_teacher_fusion.py --mode SAR
```

### Runtime

- **Per Member**: ~45 min (12 epochs on A100)
- **Full Pipeline** (30 members × 3 modes): ~24 hours
- **Total Training Time**: < 1 day (vs 200+ hours for fusion_ensembles with all SR variants)

---

## Expected Results

### Per-Mode Accuracy (Baseline Only, No SR)

| Mode | Top-1 Acc | Precision | Recall | F1-score |
|------|-----------|-----------|--------|----------|
| RAND | ~68% | ~68% | ~68% | ~67% |
| RANDRGB | ~67% | ~67% | ~67% | ~66% |
| SAR | ~70% | ~70% | ~70% | ~69% |

### Fusion Results

| Configuration | Top-1 Acc |
|---|---|
| RAND Sum-rule (10 members) | ~71.5% |
| RANDRGB Sum-rule (10 members) | ~70.5% |
| SAR Sum-rule (10 members) | ~72.0% |
| **Cross-Mode Fusion (30 total)** | **~73.5%** |

**TDA Contribution**: +1.5-2% over baseline CNN alone

**Comparison with fusion_ensembles/**:
- No SR: ~73.5%
- With SR+TDA: ~76%
- **SR Contribution**: +2.5-3%

---

## Ablation Analysis

This module enables two important ablation studies:

1. **TDA Alone** (this module): ~73.5% (TDA contribution: +1.5-2%)
2. **SR Alone** (fusion_ensembles_no_TDA): ~75.0-76% (SR contribution: +2.5-3%)
3. **SR + TDA** (fusion_ensembles): ~76%+ (combined: ~3-3.5%)

---

## Citation

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

MIT License — See LICENSE file for details.

---

## Author & Attribution

**Project:** Matteo Rambaldi
**Affiliation:** MSc Artificial Intelligence, University of Padua
**Supervision:** Prof. Loris Nanni
**Co-Supervision:** Eng. Cristian Garjitzky
**TDA Framework:** Giotto-TDA (Giansiracusa et al., 2021)
