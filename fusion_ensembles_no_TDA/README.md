# DenseNet201 with Super-Resolution (No TDA) — LCZ42 Classification

This module trains **DenseNet201 ensembles** on the So2Sat LCZ42 dataset using all **super-resolution (SR) enhancement methods, without Topological Data Analysis (TDA) fusion**.

## Purpose

This is an **ablation study** designed to isolate the contribution of **Super-Resolution enhancement** from the effects of TDA fusion. All models use enhanced (upscaled) imagery via SR methods without additional TDA features.

## Key Features

- ✅ **Super-Resolution Enabled**: 8 SR variants (baseline, VDSR, EDSR, ESRGAN, SwinIR, BSRNet, Real-ESRGAN)
- ✅ **No TDA**: Only standard DenseNet201 backbone, no topological descriptors
- ✅ **Three Band Selection Modes**:
  - `RAND`: Three random Sentinel-2 bands (10 ensemble members per SR variant)
  - `RANDRGB`: Two random MS bands + one fixed RGB band (B4/B3/B2)
  - `SAR`: Two random MS bands + one Sentinel-1 SAR band (MS + SAR fusion)
- ✅ **Multi-Scale Training**: Baseline (32×32) and upscaled (64×64 or 128×128) versions

---

## Architecture

### DenseNet201MS

Standard DenseNet201 backbone adapted for Sentinel-2 three-band inputs:

```
Input: image (B, 3, 224, 224)
       │
       └──→ DenseNet201 features
            (1920-dim after pooling)
            │
            └──→ Adaptive Avg Pool2d
                 │
                 └──→ Classifier: {1920 → 17 classes}
```

**Key Parameters**:
- **Backbone**: DenseNet201 (ImageNet pretrained)
- **Input Channels**: 3 (MS or MS+SAR)
- **Total Parameters**: ~60M (vs ~21M for TDA fusion)
- **No Fusion**: Direct classification from CNN features

---

## Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| **SEED** | 42 | Reproducibility |
| **EPOCHS** | 12 | Training epochs per member |
| **BATCH_SIZE** | 512 | Large batch (no TDA memory overhead) |
| **LEARNING_RATE** | 1e-3 | SGD with momentum |
| **WEIGHT_DECAY** | 1e-4 | L2 regularization |
| **LABEL_SMOOTHING** | 0.1 | Cross-entropy label smoothing |
| **OPTIMIZER** | SGD(momentum=0.9) | Standard SGD |
| **SCHEDULER** | None | Constant LR |
| **NUM_WORKERS** | 6 | Parallel data loading |
| **USE_ZSCORE** | True | Z-score normalization |
| **USE_SAR_DESPECKLE** | True | Median filtering for SAR |
| **USE_AUGMENTATION** | True | Random rotations/flips/crops |
| **USE_AMP** | True | Automatic Mixed Precision (AMP) |
| **USE_EMA** | True | Exponential Moving Average |

**Super-Resolution Variants** (8 total):

| Variant | Method | Scale | Patches per Config |
|---------|--------|-------|----------|
| baseline1 | None | 1× | 352K |
| baseline2 | None | 1× (duplicate) | 352K |
| vdsr2x | VDSR | 2× | 352K |
| edsr2x | EDSR | 2× | 352K |
| esrgan2x | ESRGAN | 2× | 352K |
| edsr4x | EDSR | 4× | 352K |
| swinir2x | SwinIR | 2× (Transformer) | 352K |
| bsrnet2x | BSRNet | 2× (Blind SR) | 352K |
| realesrgan4x | Real-ESRGAN | 4× (Real-world) | 352K |

---

## Directory Layout

```
fusion_ensembles_no_TDA/
├── scripts/
│   ├── train_teacher_fusion.py    # Orchestrator (trains all SR variants, no TDA)
│   ├── rand_fusion.py             # Standard DenseNet201 training (train_rand_fusion)
│   ├── fusion_densenet201.py      # Standard DenseNet201MS model (no TDA)
│   ├── dataset_reading.py         # HDF5 dataloader with z-score
│   └── utils_results.py           # HDF5 logging
├── models/
│   └── trained/                   # Saved checkpoints (.pth)
├── results/
│   ├── RAND/                      # RAND mode results (all SR variants)
│   ├── RANDRGB/                   # RANDRGB mode results (all SR variants)
│   ├── SAR/                       # SAR mode results (all SR variants)
│   └── fusion/                    # Cross-mode fusion results
└── README.md
```

---

## Training

### Quick Start

```bash
cd scripts/

# Train all modes and all SR variants
python train_teacher_fusion.py --mode ALL

# Or specific mode
python train_teacher_fusion.py --mode RAND
```

### Runtime

- **Per Member**: ~45 min (12 epochs on A100)
- **Per SR Variant**: ~7.5 hours (10 members × 45 min)
- **Full Pipeline** (8 SR variants × 3 modes × 10 members): ~180 hours (~7.5 days on A100)

---

## Expected Results

### Per-Mode Accuracy (with SR, no TDA)

| Mode | Baseline | VDSR 2× | EDSR 2× | EDSR 4× | SwinIR 2× | BSRNet 2× | Real-ESRGAN 4× | **Best SR** |
|------|----------|---------|---------|---------|-----------|-----------|---------------|-----------|
| RAND | ~68% | ~70% | ~71% | ~70% | ~71% | ~70% | ~71% | **~71%** |
| RANDRGB | ~67% | ~69% | ~70% | ~69% | ~70% | ~69% | ~70% | **~70%** |
| SAR | ~70% | ~72% | ~73% | ~72% | ~73% | ~72% | ~73% | **~73%** |

### Fusion Results

| Configuration | Accuracy |
|---|---|
| RAND Sum-rule | ~73.5% |
| RANDRGB Sum-rule | ~72.5% |
| SAR Sum-rule | ~74.5% |
| **Cross-Mode Fusion** | **~75.0-76%** |

**SR Contribution**: +2.5-3% over baseline (no SR, no TDA)

**Comparison with Other Modules**:
- Baseline DenseNet (no SR, no TDA): ~72%
- TDA Only (no SR): ~73.5%
- **SR Only (no TDA): ~75.0-76%** ← this module
- SR + TDA: ~76%+

---

## Ablation Analysis

This module enables comprehensive ablation studies:

1. **Baseline Only** (standard DenseNet201): ~72% (benchmark)
2. **TDA Only** (fusion_ensembles_no_SR): ~73.5% (TDA contribution: +1.5-2%)
3. **SR Only** (this module): **~75.0-76%** (SR contribution: +2.5-3%)
4. **SR + TDA** (fusion_ensembles): ~76%+ (combined: +4-4.5%)

**Key Finding**: Super-resolution is more effective (+3%) than TDA fusion (+2%) for improving DenseNet201 accuracy on LCZ42.

---

## Key Differences from fusion_ensembles/

| Aspect | fusion_ensembles/ | fusion_ensembles_no_TDA/ |
|--------|---|---|
| **Architecture** | DenseNet201 + TDA MLP (late fusion) | Standard DenseNet201 (no TDA) |
| **Parameters** | ~21M (with TDA) | ~60M (backbone only) |
| **SR Variants** | 8 | 8 |
| **Total Models** | 240 (30 members × 8 SR variants) | 240 (30 members × 8 SR variants) |
| **TDA Input Dim** | 18,000 / 14,400 | N/A |
| **Best Result** | ~76%+ (SR+TDA) | **~75.0-76%** (SR only) |
| **TDA Contribution** | +1.5-2% | N/A |
| **SR Contribution** | +2.5-3% | +2.5-3% |

---

## Optimization Techniques

This module uses:
- ✅ **Automatic Mixed Precision (AMP)**: `torch.amp.autocast` + `GradScaler`
- ✅ **Exponential Moving Average (EMA)**: Shadow weights for stability
- ✅ **Gradient Clipping**: Prevents divergence with large batch sizes
- ✅ **Label Smoothing**: Regularization (ε=0.1)

These techniques are safe with **hard labels** and help stabilize training of large ensembles.

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
**Super-Resolution Methods:** EDSR, ESRGAN, SwinIR, BSRNet, Real-ESRGAN, VDSR
