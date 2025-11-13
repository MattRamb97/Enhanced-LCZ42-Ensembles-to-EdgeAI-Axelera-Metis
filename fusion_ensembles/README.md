# Fusion DenseNet201 + TDA — LCZ42 Classification

This module trains **DenseNet201 + TDA fusion ensembles** on the So2Sat LCZ42 dataset, combining multispectral (MS) and optionally SAR raster data with pre-computed topological descriptors (TDA) to enhance Local Climate Zone recognition. Each ensemble explores:

- **Band Selection Modes**:
  - `RAND`: Three Sentinel-2 bands selected at random (10 ensemble members per SR variant)
  - `RANDRGB`: Two random Sentinel-2 bands + one fixed RGB band (10 ensemble members per SR variant)
  - `SAR`: Two random Sentinel-2 bands + one Sentinel-1 SAR band (10 ensemble members per SR variant)

- **Super-Resolution (SR) Variants**: Baseline (no SR), VDSR (2×), EDSR (2×/4×), ESRGAN (2×), SwinIR (2×), Real-ESRGAN (4×), BSRNet (2×)

- **Fusion Strategies**:
  - Per-configuration sum-rule (average predictions across 10 ensemble members per mode/SR pair)
  - Global sum-rule (`fusion_densenet201_full_sumrule`) fusing RAND + RANDRGB + SAR globally

---

## Directory Layout

```bash
fusion_ensembles/
├── scripts/
│   ├── train_teacher_fusion.py    # Main orchestrator: iterates all modes, SR variants, ensemble members
│   ├── rand_fusion.py             # FusionTDADataset & train_fusion_member(): shared training logic
│   ├── fusion_densenet201.py      # TDAFusionDenseNet201 model: DenseNet201 + TDA MLP + fusion classifier
│   ├── dataset_reading.py         # DatasetReading: loads HDF5, applies z-score normalization & augmentation
│   └── utils_results.py           # HDF5 logging: save_h5_results()
├── models/
│   └── trained/                   # Saved model weights (.pth files)
├── results/
│   ├── RAND/                      # Metrics/predictions for RAND configuration
│   ├── RANDRGB/                   # Metrics/predictions for RANDRGB configuration
│   ├── SAR/                       # Metrics/predictions for SAR configuration
│   └── fusion/                    # Global fusion metrics
└── README.md
```

---

## Architecture Overview

### TDAFusionDenseNet201 (fusion_densenet201.py)

A two-stream fusion architecture:

```
Input: image (B, C, H, W)  +  tda (B, 18000)
       │                        │
       ├──→ DenseNet201 backbone  │
       │    (features: (B, 1920)) │
       │                        │
       │                        └──→ TDA MLP: {1024→512→256→128}
       │                              (output: (B, 128))
       │
       ├─── Concat ──→ (B, 2048) ──→ LayerNorm ──→ Fusion Dropout
            │
            └──→ Classifier: {256 → 17 classes}
```

**Key Configuration**:
- **DenseNet201 backbone**: Pretrained on ImageNet (frozen initially, then fine-tuned)
- **TDA MLP hidden dims**: [1024, 512, 256, 128] with BatchNorm, ReLU, and dropout
- **Fusion dimension**: 1920 (CNN features) + 128 (TDA features) = 2048
- **Dropout rates**: 0.5 (standard), 0.25 (CNN/TDA MLP first layer)
- **Activation**: ReLU
- **Normalization**: LayerNorm applied to fused features before classification

**Number of Parameters**:
- DenseNet201 features: ~18.4M (backbone)
- TDA MLP: ~2.1M (4 FC layers + BatchNorms)
- Fusion classifier: ~0.66M
- **Total**: ~21M parameters

---

## Training Configuration

All hyperparameters are hardcoded in `train_teacher_fusion.py` (lines 18–35):

| Parameter | Value | Notes |
|-----------|-------|-------|
| **SEED** | 42 | Random seed for reproducibility |
| **EPOCHS** | 12 | Training epochs per ensemble member |
| **BATCH_SIZE** | 128 | Batch size (reduced from 512 due to fusion memory overhead) |
| **LEARNING_RATE** | 1e-3 | SGD initial LR |
| **WEIGHT_DECAY** | 1e-4 | L2 regularization |
| **LABEL_SMOOTHING** | 0.1 | Cross-entropy label smoothing |
| **OPTIMIZER** | SGD(momentum=0.9) | Standard SGD with momentum |
| **SCHEDULER** | None | No LR scheduling |
| **NUM_WORKERS** | 8 | Parallel data loading (reduce to 2–4 on NFS) |
| **USE_ZSCORE** | True | Normalize using z-score (μ, σ) |
| **USE_SAR_DESPECKLE** | True | Apply median filtering to SAR |
| **USE_AUG** | True | Data augmentation (random flips/rotations/crops) |
| **USE_TORCH_COMPILE** | False | torch.compile disabled (experimental) |

**Super-Resolution (SR) Variants** (METHODS_MS / METHODS_SAR):

| Suffix | Name | Type | Scale |
|--------|------|------|-------|
| (empty) | `baseline1` | None | 1× (original 32×32) |
| (empty) | `baseline2` | None | 1× (original 32×32) |
| `_vdsr2x` | `vdsr2x` | VDSR | 2× |
| `_edsr2x` | `edsr2x` | EDSR | 2× |
| `_esrgan2x` | `esrgan2x` | ESRGAN | 2× |
| `_edsr4x` | `edsr4x` | EDSR | 4× |
| `_swinir2x` | `swinir2x` | SwinIR | 2× |
| `_vdsr3x` | `vdsr3x` | VDSR | 3× |
| `_bsrnet2x` | `bsrnet2x` | BSRNet | 2× |
| `_realesrgan4x` | `realesrgan4x` | Real-ESRGAN | 4× |

---

## Core Scripts

### train_teacher_fusion.py

**Purpose**: Master orchestrator that:
1. Loads LCZ42 train/test splits from `data/lcz42/tables_MS.mat`
2. Loads pre-computed TDA features from `TDA/data/`
3. Iterates over all modes (RAND, RANDRGB, SAR) and SR variants
4. For each combination, trains N ensemble members with different random band selections
5. Computes per-configuration sum-rule fusion
6. Computes global cross-mode fusion

**Output Structure** (results/ directory):

```
results/
├── RAND/
│   ├── rand_baseline1_member0_summary.json
│   ├── rand_baseline1_member0_history.csv
│   ├── rand_baseline1_member0_eval.h5
│   ├── ...
│   ├── rand_vdsr2x_member9_eval.h5
│   ├── RAND_sumrule_summary.json
│   └── RAND_sumrule_eval.h5
├── RANDRGB/
│   ├── randrgb_baseline1_member0_summary.json
│   ├── ...
│   └── RANDRGB_sumrule_eval.h5
├── SAR/
│   ├── sar_baseline1_member0_summary.json
│   ├── ...
│   └── SAR_sumrule_eval.h5
└── fusion/
    ├── fusion_densenet201_full_sumrule_summary.json
    └── fusion_densenet201_full_sumrule_eval.h5
```

### rand_fusion.py

**Purpose**: Contains the core training loop and dataset wrapper.

**FusionTDADataset** (lines 16–72):
- Wraps MS/SAR datasets with aligned TDA features
- Ensures index alignment: `TDA[dataset.table['Index']]`
- Selects bands dynamically per forward pass (e.g., `torch.index_select(img_ms, dim=0, ms_band_indices)`)
- Concatenates MS and optional SAR into single image tensor

**train_fusion_member(cfg)** (lines 75+):
- Instantiates `TDAFusionDenseNet201` model
- Creates FusionTDADataset with band selections per mode/member
- Sets up optimizer (SGD), loss (CrossEntropyLoss + label smoothing), scheduler (None)
- Mixed-precision training with `torch.cuda.amp` (GradScaler, autocast)
- Per-epoch validation on test set
- Saves `.pth` checkpoint, `.csv` history, `.json` summary, `.h5` eval file

**Band Selection Logic** (mode-dependent):

| Mode | Band Selection |
|------|---|
| **RAND** | RNG selects 3 random bands from all MS bands (deterministic per member_id) |
| **RANDRGB** | RNG selects 2 random MS bands + 1 fixed RGB band (B4, B3, B2) = 3 channels total |
| **SAR** | RNG selects 2 random MS bands + 1 SAR band (VV or VH) = 3 channels total |

### fusion_densenet201.py

**TDAFusionDenseNet201** class (lines 7–58):
- Inherits from `nn.Module`
- **Constructor parameters**:
  - `tda_input_dim=18000`: Dimensionality of pre-computed TDA features
  - `num_classes=17`: Number of LCZ classes
  - `dropout_rate=0.5`: Dropout probability

- **Key components**:
  - `self.features`: DenseNet201 feature extractor (ImageNet pretrained)
  - `self.cnn_dropout`: 0.25 dropout after CNN features
  - `self.tda_mlp`: 4-layer MLP for TDA feature processing (hidden dims: [1024, 512, 256, 128])
  - `self.fusion_norm`: LayerNorm over concatenated features
  - `self.fusion_dropout`: 0.5 dropout before classification
  - `self.classifier`: 2-layer classifier (256 hidden → 17 classes)

- **forward(image, tda)**:
  - Image path: `image → backbone.features → ReLU → AdaptiveAvgPool2d → Flatten → dropout`
  - TDA path: `tda → MLP → output (128 dims)`
  - Fusion: `Concat → LayerNorm → Dropout → Classifier → logits (17 classes)`

### dataset_reading.py

**DatasetReading** class (linked from parent repository):
- Loads multispectral and SAR patches from HDF5 (`data/lcz42/test_MS.h5`, `test_SAR.h5`)
- Applies z-score normalization using pre-computed μ/σ
- Supports optional augmentation (random flips, rotations, crops)
- Returns torch.Tensor (C, H, W) normalized to [~mean=0, ~std=1]

**Important**: TDA feature loading is separate (from `TDA/data/` subdirectory).

### utils_results.py

**save_h5_results(filename, y_true, probs, cm, classes)**:
- Saves evaluation results to HDF5 format for MATLAB compatibility
- Stores:
  - `y_true`: Ground-truth labels
  - `probs`: Predicted probabilities (softmax logits)
  - `confusion_matrix`: Per-class confusion
  - `classes`: Label names

---

## Running Instructions

### Prerequisites

1. **LCZ42 data**: `data/lcz42/tables_MS.mat` (train/test splits)
2. **TDA features**: Pre-computed in `tda/data/` (`tda_MS_features_test.h5`,`tda_MS_features_*.h5`,`labels.h5`, `labels_test.h5`)
3. **Super-resolution images**: Pre-computed SR variants in `data/lcz42/` (VDSR, EDSR, etc.)
4. **Python environment**:
   ```bash
   pip install -r requirements.txt
   ```

### Full Training (All Modes + SR Variants)

```bash
cd fusion_ensembles/scripts
python train_teacher_fusion.py
```

This will automatically run:
- 3 modes (RAND, RANDRGB, SAR)
- 10 SR variants each (~3 Baselines 2× upscaling methods + 1 No-SR baseline)
- 10 ensemble members per combination
- **Total**: 3 × 10 × 10 = 300 model trainings

**Expected runtime**: ~2-3 days on a single A40 GPU (depends on data I/O).

### Single Mode Training

To train only one mode:

```python
python train_teacher_fusion.py --ensemble_type Rand
```

---

## Output Files

### Per-Member Files (one per ensemble member, per SR variant, per mode)

For each training run, three files are saved:

1. **`<mode>_<sr_variant>_member<id>_summary.json`**

2. **`<mode>_<sr_variant>_member<id>_history.csv`**

3. **`<mode>_<sr_variant>_member<id>_eval.h5`**

### Ensemble Sum-Rule Files

Per mode (RAND, RANDRGB, SAR):

**`<mode>_sumrule_summary.json`**:
- Averages predictions across 10 ensemble members for that SR variant group
- Reports aggregated metrics (accuracy, F1, etc.)
- Lists `components`: ["rand_baseline1", "rand_vdsr2x", ..., "rand_realesrgan4x"] (10 variants)

**`<mode>_sumrule_eval.h5`**:
- Fused predictions + confusion matrix

### Global Fusion File

**`fusion_densenet201_full_sumrule_summary.json`**:
- Fuses all 3 modes × 10 variants = 30 component models
- Represents the best fusion strategy combining RAND + RANDRGB + SAR

---

## Key Technical Notes

1. **Index Alignment**:
   - Dataset indices are 1-based in MATLAB but converted to 0-based in Python.
   - TDA features are aligned: `tda_features[sample_index]` corresponds to the correct sample.

2. **SAR Band Selection**:
   - SAR mode always has 2 MS bands + 1 SAR band = 3 channels
   - TDA features are the same for MS and SAR modes (no separate TDA for SAR)

3. **Band Selection Reproducibility**:
   - Each ensemble member uses a deterministic RNG seeded by `base_seed + member_id`
   - Ensures same band selections across runs

4. **Memory Usage**:
   - Batch size set to 128 (not 512) due to fusion overhead
   - TDA features: 18000 dims per sample adds significant memory
   - Recommend GPU with ≥16GB VRAM

5. **Data Augmentation**:
   - Applied during training (if `USE_AUG=True`)
   - Includes: random horizontal/vertical flips, random rotations, random crops, cutout

6. **Mixed Precision Training**:
   - Enabled by default using `torch.cuda.amp`
   - GradScaler prevents gradient underflow

7. **NFS/NAS Considerations**:
   - If using network storage, reduce `NUM_WORKERS` to 2–4 to minimize I/O conflicts
   - Default 8 workers may cause timeouts on slow networks

---

## Hyperparameter Tuning Guide

If you need to adapt the configuration:

| Parameter | Effect | Suggested Range |
|-----------|--------|---|
| **EPOCHS** | Total training time & convergence | 10–20 |
| **BATCH_SIZE** | Memory usage & gradient estimation | 32–256 (watch VRAM) |
| **LEARNING_RATE** | Convergence speed & stability | 1e-4 to 1e-2 |
| **WEIGHT_DECAY** | Regularization strength | 1e-5 to 1e-3 |
| **LABEL_SMOOTHING** | Label soft targets | 0.05–0.2 |
| **dropout_rate** | Overfitting prevention | 0.2–0.7 |

---

## Common Issues & Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| **CUDA out of memory** | Batch size too large or TDA dims | Reduce BATCH_SIZE from 128 to 64 |
| **Slow data loading** | NUM_WORKERS too high on NFS | Reduce NUM_WORKERS to 2–4 |
| **Index mismatch error** | MS/SAR dataset indices misaligned | Check HDF5 table loading in dataset_reading.py |
| **TDA features not found** | Missing TDA precomputation | Run TDA feature extraction first |
| **Poor accuracy** | Incorrect z-score normalization | Verify μ/σ values in dataset_reading.py |

---

## Citation

If you use this module in your research, please cite:

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

This project is released under the MIT License.
See the LICENSE file for details.

---

## Author & Attribution

**Project:** Matteo Rambaldi

**Affiliation:** MSc Artificial Intelligence, University of Padua

**Supervision:** Prof. Loris Nanni

**Co-Supervision:** Eng. Cristian Garjitzky

**TDA Framework:** Giotto-TDA (Giansiracusa et al., 2021)