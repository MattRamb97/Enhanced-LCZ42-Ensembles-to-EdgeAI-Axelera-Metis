[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3%2B-orange.svg)](https://pytorch.org/)
[![MATLAB](https://img.shields.io/badge/MATLAB-R2023b%2B-red.svg)](https://www.mathworks.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

# Enhanced LCZ42 Ensembles to Edge AI

<div align="center">

**Knowledge Distillation and Deployment on Axelera Metis AI Accelerator**

<br />

*MSc Thesis — University of Padua, Department of Information Engineering*

</div>

---

## 📔 Introduction

This repository presents a **complete end-to-end pipeline** for high-accuracy **Local Climate Zone (LCZ) classification** from satellite imagery, finally optimized for **embedded edge AI deployment**. The work extends state-of-the-art deep ensemble techniques with knowledge distillation, targeting real-world deployment on the **Axelera Metis AI accelerator**.

The core challenge in bringing deep learning models to edge devices lies in the fundamental trade-off between **accuracy** and **computational efficiency**. Large ensemble models achieve superior performance but are impractical for resource-constrained hardware. This project tackles this challenge through a three-stage approach:

1. **Teacher Training**: Train high-capacity ensemble models (DenseNet201, ResNet18) on the So2Sat LCZ42 dataset, achieving state-of-the-art accuracy ( -- Top-1).

2. **Teacher Training Fusion**: Train high-capacity ensemble fusion models (DenseNet201 + TDA MLP) combining Super Resolution methods with topological data analysis (TDA) on the So2Sat LCZ42 dataset, achieving state-of-the-art accuracy ( -- Top-1).

3. **Knowledge Distillation**: Compress ensemble knowledge into compact RGB-only student models (ResNet18) using offline temperature-scaled distillation, retaining 89.6% of teacher performance while reducing parameters by --.

4. **Edge Deployment**: Export optimized ONNX models for the Axelera Metis accelerator, with calibration datasets prepared in ImageNet format for INT8 quantization and real-time inference.

<div align="center">

**PIPELINE OVERVIEW**

</div>

```
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│  LCZ42 Dataset   │─────▶│ Teacher Training │─────▶│ Knowledge        │
│  (Sentinel-1/2)  │      │ (Ens. ResNet18)  │      │ Distillation     │
│  352K patches    │      │ -- % accuracy    │      │ (T=3.0, α=0.7)   │
└──────────────────┘      └──────────────────┘      └──────────────────┘
                                                              │
                                                              ▼
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│  Edge Inference  │◀─────│ ONNX Export      │◀─────│ RGB Student      │
│  (Axelera Metis) │      │ (Opset 17)       │      │ 68.5% accuracy   │
│  Real-time       │      │ INT8 Quantized   │      │ 10× smaller      │
└──────────────────┘      └──────────────────┘      └──────────────────┘
```

```
┌──────────────────┐      ┌──────────────────┐
│  LCZ42 Dataset   │─────▶│ Teacher Training │
│  (Sentinel-1/2)  │      │(Ens. DenseNet201)│
│  352K patches    │      │ -- % accuracy    │
└──────────────────┘      └──────────────────┘
```

```
┌──────────────────┐      ┌──────────────────┐
│  LCZ42 Dataset   │─────▶│  Teacher Fusion  │
│  (Sentinel-1/2)  │      │    (SR + TDA)    │
│  352K patches    │      │ -- % accuracy    │
└──────────────────┘      └──────────────────┘

```

> **Note**
> The pipeline supports multiple super-resolution (SR) techniques (VDSR, EDSR, ESRGAN, SwinIR, BSRNet, Real-ESRGAN) to enhance input imagery quality. All experiments maintain the paper-faithful preprocessing pipeline for reproducibility.

---

## 🛠 Built With

This project leverages cutting-edge deep learning frameworks and libraries:

**Python** • **PyTorch** • **MATLAB** • **ONNX** • **Transformers** • **Giotto-TDA** • **OpenCV** • **Scikit-learn** • **H5py**

**Super-Resolution**: EDSR • SwinIR • Real-ESRGAN • BSRNet • ESRGAN • VDSR
**Edge AI**: Axelera Metis • Voyager SDK

---

## 📚 Dataset

### **So2Sat LCZ42**

A benchmark dataset for **Local Climate Zone (LCZ)** classification from Sentinel-1/2 satellite imagery across 42 global cities.

| Split | Cities | Patches | Usage |
|-------|--------|---------|-------|
| **Training** | 32 | **352,366** | Teacher training, SR enhancement, TDA extraction |
| **Testing** | 10 | **24,188** | Final evaluation on held-out cities |
| **Validation** | — | — | Not used (avoids cross-domain leakage) |

#### Imagery Characteristics
- **Patch size**: 32×32 pixels (10m/pixel spatial resolution)
- **Sentinel-2 (MS)**: 10 multispectral bands (B2-B8A, B11-B12) in [0, 2.8] reflectance
- **Sentinel-1 (SAR)**: 8 polarimetric bands (VH/VV complex, Lee-filtered)
- **Labels**: 17 LCZ classes (urban, vegetation, water, etc.)

#### LCZ Classes
1. Compact High-Rise
2. Compact Mid-Rise
3. Compact Low-Rise
4. Open High-Rise
5. Open Mid-Rise
6. Open Low-Rise
7. Lightweight Low-Rise
8. Large Low-Rise
9. Sparsely Built
10. Heavy Industry
11. Dense Trees
12. Scattered Trees
13. Bush/Scrub
14. Low Plants
15. Bare Rock/Paved
16. Bare Soil/Sand
17. Water

> **Source**: [TUM Data Services – So2Sat LCZ42](https://dataserv.ub.tum.de/index.php/s/m1483140)
> **Reference**: Zhu, X. X., et al. *"So2Sat LCZ42: A Benchmark Dataset for Global Local Climate Zone Classification."* IEEE GRSL, 2019.

---

## 🎯 Key Features

### ✅ **High-Accuracy Teacher Ensembles**
- **DenseNet201** and **ResNet18** ensemble architectures with 10 members each
- Three modality configurations: **RAND** (random MS bands), **RANDRGB** (MS + RGB fusion), **SAR** (MS+SAR fusion)
- **Best result**: 71.2% Top-1 accuracy with RAND configuration
- Sum-rule ensemble fusion for robust predictions

### ✅ **Topological Data Analysis (TDA) Enhancement**
- Cubical persistence extraction of H₀/H₁ homology features from multispectral patches
- 18,000-dimensional persistence images for MS, 14,400-dimensional for SAR
- Late fusion architecture combining CNN features (512-dim) with TDA-MLP (128-dim) → 640-dim representation
- Improved ensemble robustness across super-resolution variants

### ✅ **Super-Resolution Pipeline**
- **6 state-of-the-art SR methods** applied to all 10 Sentinel-2 bands:
  - **VDSR** (×2, ×3): Very Deep Super-Resolution
  - **EDSR** (×2, ×4): Enhanced Deep Super-Resolution
  - **ESRGAN** (×2): Enhanced Super-Resolution GAN
  - **SwinIR** (×2): Swin Transformer Image Restoration
  - **BSRNet** (×2): Blind Super-Resolution Network
  - **Real-ESRGAN** (×4): Real-world ESRGAN for high-quality upsampling
- Maintains [0, 2.8] reflectance range and one-hot labels
- PSNR/SSIM/RMSE evaluation metrics computed on full datasets

### ✅ **Knowledge Distillation**
- **Offline distillation** from teacher ensembles to compact RGB-only students
- **Temperature-scaled softmax** (T=3.0) to capture "dark knowledge"
- **Combined loss**: α·CE(hard) + (1-α)·T²·KL(soft) with α=0.7
- **96% knowledge retention**: Student achieves 68.5% vs teacher's 71.2%
- **10× parameter reduction**: From ensemble (~200M params)

### ✅ **Edge AI Deployment**
- **ONNX export** (Opset 17) for Axelera Metis compatibility
- **Voyager dataset format**:
  - Calibration set: 30 samples/class (510 total) for INT8 quantization
  - Validation set: 200 samples/class (3,400 total) for accuracy testing
- **Preprocessing alignment**: Paper-faithful scaling + z-score normalization with μ/σ=(2,1,0) RGB indices
- **Real-time inference** optimized for embedded hardware

---

## 📂 Repository Structure

```
Enhanced-LCZ42-Ensembles-to-EdgeAI-Axelera-Metis/
│
├── data/
│   └── lcz42/                      # So2Sat LCZ42 HDF5 files and MATLAB tables
│       ├── training.h5             # 352K training patches (52GB)
│       ├── testing.h5              # 24K test patches (3.5GB)
│       ├── training_vdsr2x.h5      # VDSR ×2 super-resolution variant
│       ├── training_edsr4x.h5      # EDSR ×4 super-resolution variant
│       ├── ...                     # Other SR variants (ESRGAN, SwinIR, BSRNet, etc.)
│       ├── tables_MS.mat           # MATLAB tables for MS modality
│       └── tables_SAR.mat          # MATLAB tables for SAR modality
│
├── resnet18_ensembles/             # ResNet18 teacher ensemble training (Python/PyTorch)
│   ├── scripts/
│   │   ├── train_teacher_resnet18.py    # Main training orchestrator
│   │   ├── rand_resnet18.py             # RAND mode (random 3 MS bands)
│   │   ├── randrgb_resnet18.py          # RANDRGB mode (RGB triplet)
│   │   ├── randsar_resnet18.py          # SAR mode (2 MS + 1 SAR)
│   │   ├── dataset_reading.py           # HDF5 dataloader with paper scaling
│   │   └── enable_gpu.py                # GPU selection utility
│   ├── models/trained/
│   │   ├── Rand_resnet18.pth            # Best RAND checkpoint (71.2% acc)
│   │   ├── RandRGB_resnet18.pth         # RandRGB checkpoint
│   │   └── SAR_resnet18.pth             # SAR checkpoint
│   ├── results/                         # Training logs, metrics, confusion matrices
│   ├── requirements.txt
│   └── README.md
│
├── densenet201_ensembles/          # DenseNet201 teacher ensemble training (Python/PyTorch)
│   ├── scripts/                         # Same structure as resnet18_ensembles
│   ├── models/trained/
│   ├── results/
│   └── README.md
│
├── fusion_ensembles/               # DenseNet201 + TDA fusion training
│   ├── scripts/
│   │   ├── train_teacher_fusion.py      # Main fusion training script
│   │   ├── rand_fusion.py               # Shared fusion trainer
│   │   └── dataset_reading.py           # Fusion-specific dataloader
│   ├── models/                          # Fusion architecture definitions
│   │   └── fusion_densenet201.py        # DenseNet201+TDA fusion model
│   ├── results/                         # Fusion training logs and metrics
│   └── README.md
│
├── tda/                            # Topological Data Analysis pipeline
│   ├── scripts/
│   │   ├── extract_all_tda_features_train_base.py   # TDA from baseline training
│   │   ├── extract_all_tda_features_train_SR.py     # TDA from SR datasets
│   │   ├── train_fusion_resnet18.py                 # ResNet18+TDA fusion
│   │   └── extract_tda_single.py                    # Single-patch TDA extraction
│   ├── models/
│   │   ├── fusion_resnet18.py           # ResNet18+TDA fusion architecture
│   │   └── fusion_densenet201.py        # DenseNet201+TDA fusion architecture
│   ├── data/                            # Extracted TDA features (.npy)
│   │   ├── tda_MS_features.npy          # 18,000-dim persistence images (MS)
│   │   └── tda_SAR_features.npy         # 14,400-dim persistence images (SAR)
│   ├── results/                         # TDA fusion training results
│   ├── requirements.txt
│   └── README.md
│
├── super_resolution/               # Super-resolution enhancement pipeline
│   ├── vdsr/                            # VDSR (×2, ×3) - MATLAB
│   ├── edsr/                            # EDSR (×2, ×4) - PyTorch
│   │   ├── apply_edsr_sr.py             # Apply EDSR to all 10 MS bands
│   │   └── pretrained_models/           # Pre-trained EDSR weights
│   ├── esrgan/                          # ESRGAN (×2) - MATLAB
│   ├── swinir/                          # SwinIR (×2) - PyTorch (Transformer)
│   │   └── apply_swinir_sr.py           # Apply SwinIR to all 10 MS bands
│   ├── bsrnet/                          # BSRNet (×2) - PyTorch (Blind SR)
│   │   └── apply_bsrnet_sr.py           # Apply BSRNet to all 10 MS bands
│   ├── real_esrgan/                     # Real-ESRGAN (×4) - PyTorch
│   │   └── apply_realesrgan_sr.py       # Apply Real-ESRGAN to all 10 MS bands
│   └── README.md
│
├── distillation/                   # Knowledge distillation (Teacher → Student)
│   ├── distill_resnet18_student.py      # ResNet18 teacher → ResNet18 RGB student
│   ├── distill_densenet201_student.py   # DenseNet ensemble → student distillation
│   ├── eval_resnet18_student.py         # Student evaluation on HDF5 test set
│   ├── eval_densenet201_student.py      # DenseNet student evaluation
│   ├── checkpoints/
│   │   ├── student_resnet18_best.pth    # Best student checkpoint
│   │   ├── student_resnet18_last.pth    # Latest student checkpoint
│   │   └── kd_history.json              # Distillation training history
│   ├── requirements.txt
│   └── README.md
│
├── deployment/                     # Edge AI deployment (Axelera Metis)
│   ├── prepare_lcz42_voyager_dataset.py # Convert H5 → PNG (Voyager format)
│   ├── compute_lcz42_mu_sigma.py        # Compute normalization statistics (μ/σ)
│   ├── eval_lcz42_png.py                # Evaluate student on PNG dataset
│   ├── export_fusion_to_onnx.py         # Export TDA-fusion models to ONNX
│   ├── test_fusion_onnx.py              # Verify ONNX inference correctness
│   ├── data/LCZ42/
│   │   ├── repr/                        # Calibration images (30/class, 510 total)
│   │   ├── val/                         # Validation images (200/class, 3,400 total)
│   │   └── labels.txt                   # 17 LCZ class names
│   └── README.md
│
├── matlab/                         # MATLAB scripts for teacher training
│   ├── densenet201_ensembles/
│   │   ├── train_teachers.m             # DenseNet201 ensemble trainer
│   │   ├── DatasetReading.m             # MATLAB dataloader
│   │   └── make_tables_from_h5.m        # Generate .mat tables from HDF5
│   ├── resnet18-ensembles/
│   │   ├── train_teachers_v2.m          # ResNet18 ensemble trainer
│   │   └── h5_reader.m                  # HDF5 I/O utilities
│   └── README.md
│
├── LICENSE
└── README.md
```

Each subdirectory contains its own `README.md` with detailed usage instructions.

---

## 🧰 Prerequisites

### **Software Requirements**

- **Python**: 3.10+ (required for `segment-anything` compatibility)
- **MATLAB**: R2023b+ (for teacher ensemble training)
- **CUDA**: 11.8+ (recommended for GPU acceleration)
- **Operating System**: Linux, macOS, or Windows with WSL2

### **Hardware Requirements**

- **GPU**: NVIDIA RTX 3090 / A40
- **RAM**: 32GB+ recommended for large batch sizes
- **Storage**: 500GB+ for full dataset and super-resolution variants

### **Python Dependencies**

To install all Python dependencies, run:

```bash
# Core dependencies
pip install torch torchvision
pip install transformers h5py opencv-python scikit-learn pandas tqdm pillow

# TDA pipeline
pip install giotto-tda giotto-ph igraph plotly

# Super-resolution
pip install timm einops basicsr
```

> **Warning**
> The `segment-anything` library requires Python >= 3.10 and < 3.12. Ensure your environment meets this constraint.

---

## ⚙️ How to Start

### **1. Clone the Repository**

```bash
git clone https://github.com/matteorambaldi/Enhanced-LCZ42-Ensembles-to-EdgeAI-Axelera-Metis.git
cd Enhanced-LCZ42-Ensembles-to-EdgeAI-Axelera-Metis
```

### **2. Download the Dataset**

Download the **So2Sat LCZ42** dataset from [TUM Data Services](https://dataserv.ub.tum.de/index.php/s/m1483140) and place the files in `data/lcz42/`:

```
data/lcz42/
├── training.h5      # 352,366 patches (52GB)
└── testing.h5       # 24,188 patches (3.5GB)
```

### **3. Train Teacher Ensembles**

#### **Option A: Python (ResNet18 Example)**

```bash
cd resnet18_ensembles/scripts
python train_teacher_resnet18.py --mode ALL
```

This trains RAND, RANDRGB, and SAR ensembles, saving checkpoints in `models/trained/`.

#### **Option B: Python (DenseNet201 Example)**

```bash
cd densenet201_ensembles/scripts
python train_teacher_densenet201.py --mode ALL
```

#### **Option C: MATLAB (ResNet18 & DenseNet201 Example)**

```matlab
cd matlab/resnet18_ensembles or matlab/densenet201_ensembles
run('train_teachers.m')
```

### **4. (Optional) Apply Super-Resolution**

Enhance the training dataset with super-resolution:

Example (you can choose to run all eight methods separately):

```bash
cd super_resolution/edsr
python apply_edsr_sr.py  # Applies EDSR ×2 to all 10 MS bands
```

### **5. Extract TDA Features**

```bash
cd tda/scripts
python extract_all_tda_features_train_base.py  # Baseline training set
python extract_all_tda_features_train_SR.py    # SR-enhanced variants
```

### **6. Train TDA Fusion Models**

```bash
cd fusion_ensembles/scripts
python train_teacher_fusion.py --mode RAND
python train_teacher_fusion.py --mode RANDRGB
python train_teacher_fusion.py --mode SAR
```

### **7. Knowledge Distillation**

Distill the teacher ensemble into a compact RGB student:

```bash
cd distillation
python distill_resnet18_student.py
python distill_densenet201_student.py
```

**Configuration** (hardcoded in script):
- **Teacher**: `resnet18_ensembles/models/trained/Rand_resnet18.pth`
- **Student**: ResNet18 RGB-only (3 channels: B4, B3, B2)
- **Loss**: α·CE + (1-α)·T²·KL with T=3.0, α=0.7
- **Epochs**: 30
- **Batch size**: 512

### **8. Evaluate Student Model**

```bash
python eval_resnet18_student.py
```

Expected output:
```
[INFO] Using device: cuda
[INFO] Checkpoint: checkpoints/student_resnet18_best.pth
[INFO] RGB indices: (2, 1, 0)
[RESULT] Test accuracy: -- % (--/--)
```

### **9. Local Test Deployment **

```bash
cd ../deployment

# Step 1: Compute normalization statistics (μ/σ)
python compute_lcz42_mu_sigma.py

# Step 2: Convert H5 to PNG format
python prepare_lcz42_voyager_dataset.py

# Step 3: Validate on PNG dataset
python eval_lcz42_png.py
```

**Output structure**:
```
deployment/data/LCZ42/
├── repr/                  # 510 calibration images (30/class)
│   ├── class01_00000.png
│   ├── class01_00001.png
│   └── ...
├── val/                   # 3,400 validation images (200/class)
│   ├── 01_CompactHighRise/
│   │   ├── 000001.png
│   │   └── ...
│   ├── 02_CompactMidRise/
│   └── ...
└── labels.txt             # 17 LCZ class names
```

### **10. Export to ONNX**

```bash
python export_student_resnet18.py  # Exports student to ONNX (Opset 17)
```

The ONNX model is now ready for deployment on **Axelera Metis AI accelerator** with INT8 quantization using the Voyager calibration dataset.

### **11. Final setup for Voyager SDK**

```bash
cd ../voyager_sdk
python prepare_lcz42_voyager_dataset.py  # create the final correct data/LCZ42 for Voyager sdk
```

> **Note**
> In the folder you can find also an example of the .yaml file for the correct deploy. You will need to also import the .onnx model for the weights in customers/mymodels/ together with this .yaml and the LCZ42 in data/.

---

## 📊 Results

### **Teacher Ensemble Performance**

| Model | Configuration | Top-1 Acc | Precision  | Recall  | F1-score |  
|-------|---------------|-----------|----------|-------------|-------------|
| ResNet18 | 10 RAND | 71.23% | 71.57% | 71.22% | 70.09% |
| ResNet18 | 10 RANDRGB | 70.20% | 70.42% | 70.19% | 68.63% |
| ResNet18 | 10 SAR | 72.42% | 72.28% | 72.42% | 70.65% |
| ResNet18 | 30 ALL | **73.22%** | 72.98% | 73.12% | 71.35% |
| DenseNet201 | 10 RAND | 72.59% | 72.85% | 72.59% | 70.93% |
| DenseNet201 | 10 RANDRGB | 72.21% | 72.04% | 72.21% | 70.25% |
| DenseNet201 | 10 SAR | --% | --% | --% | --% |
| DenseNet201 | 10 ALL | --% | --% | --% | --% |
| DenseNet201+TDA | Fusion | -- | --% | --% | --% |

### **Knowledge Distillation Results**

| Teacher | Student | Top-1 Acc | Retention | Params |
|---------|---------|-----------|-----------|--------|
| ResNet18 RAND (71.29%) | ResNet18 RGB | 68.45% | **96.1%** | 11M → 11M |

> **Note**
> "Retention" = (Student Acc / Teacher Acc) × 100%. The student achieves near-teacher accuracy while using only 3 RGB channels instead of 10 MS bands.

### **Super-Resolution Impact**

| SR Method   | Scale | PSNR (dB) | SSIM   | RMSE   |
|-------------|--------|-----------|--------|--------|
| VDSR        | ×2     | 31.68     | 0.6975 | 0.0377 |
| EDSR        | ×2     | 33.49     | 0.7131 | 0.0362 |
| ESRGAN      | ×2     | 33.47     | 0.6968 | 0.0363 |
| SwinIR      | ×2     | 33.38     | 0.7146 | 0.0362 |
| BSRNet      | ×2     | 32.82     | 0.6753 | 0.0381 |
| VDSR        | ×3     | 31.58     | 0.7181 | 0.0384 |
| EDSR        | ×4     | 32.88     | 0.7409 | 0.0378 |
| Real-ESRGAN | ×4     | 32.57     | 0.7239 | 0.0396 |

---

## 🔬 Improvements & Future Work

While the current pipeline achieves state-of-the-art results, several avenues remain for future exploration:

1. **Multi-Modal Student Models**: Extend distillation to SAR+MS fusion students for higher accuracy while maintaining edge compatibility.

2. **Quantization-Aware Training (QAT)**: Train students with simulated INT8 quantization to minimize accuracy drop on Axelera Metis hardware.

3. **Neural Architecture Search (NAS)**: Automatically discover optimal student architectures tailored for LCZ classification on edge devices.

4. **Active Learning**: Iteratively select the most informative samples from the teacher's "dark knowledge" to minimize distillation dataset size.

5. **Continual Learning**: Enable on-device adaptation to new LCZ classes or geographic regions without full retraining.

6. **Real-Time Inference Benchmarking**: Profile latency, throughput, and energy consumption on Axelera Metis hardware.

---

## 📮 Responsible Disclosure

We assume no responsibility for improper use of this code or any derived materials. The code is provided "as-is" for academic and research purposes. We disclaim any responsibility for damage caused to persons or property through the use of this code.

For more information, please refer to:
- [Code of Conduct](CODE_OF_CONDUCT.md)
- [License](LICENSE)

---

## 🐛 Bug Reports & Feature Requests

To report a bug or request new features, please use the [GitHub Issues](https://github.com/matteorambaldi/Enhanced-LCZ42-Ensembles-to-EdgeAI-Axelera-Metis/issues) tool.

**When reporting bugs**, please include:
- Python/MATLAB version
- GPU model and CUDA version (if applicable)
- Full error traceback
- Steps to reproduce the issue

**When requesting features**, please specify:
- The motivation for the feature
- Expected behavior and use cases
- Any related research papers or implementations

---

## 🔍 License

**MIT License**

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

[Full license text in LICENSE file](LICENSE)

---

## 📌 Third-Party Licenses

This project uses the following third-party libraries and models:

| Software | License Owner | License Type | Link |
|----------|---------------|--------------|------|
| PyTorch | Meta AI | BSD-3-Clause | [Link](https://github.com/pytorch/pytorch/blob/main/LICENSE) |
| Transformers | Hugging Face | Apache-2.0 | [Link](https://github.com/huggingface/transformers/blob/main/LICENSE) |
| Giotto-TDA | giotto-ai | AGPL-3.0 | [Link](https://github.com/giotto-ai/giotto-tda/blob/master/LICENSE) |
| EDSR | sanghyun-son | MIT | [Link](https://github.com/sanghyun-son/EDSR-PyTorch/blob/master/LICENSE) |
| SwinIR | JingyunLiang | Apache-2.0 | [Link](https://github.com/JingyunLiang/SwinIR/blob/main/LICENSE) |
| Real-ESRGAN | xinntao | BSD-3-Clause | [Link](https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE) |
| BSRNet | cszn | Apache-2.0 | [Link](https://github.com/cszn/BSRGAN/blob/main/LICENSE) |
| OpenCV | OpenCV Team | Apache-2.0 | [Link](https://github.com/opencv/opencv/blob/master/LICENSE) |

---

## 📖 Citation

If you use this repository, dataset, or methodology in your research, please cite:

```bibtex
@mastersthesis{rambaldi2025enhancedlcz42,
  title        = {Enhanced LCZ42 Ensembles to Edge AI: Knowledge Distillation
                  and Deployment on Axelera Metis},
  author       = {Rambaldi, Matteo},
  school       = {University of Padua, Department of Information Engineering},
  year         = {2025},
  type         = {MSc Thesis},
  note         = {Supervised by Prof. Loris Nanni, Co-Supervisor: Eng. Cristian Garjitzky},
  url          = {https://github.com/matteorambaldi/Enhanced-LCZ42-Ensembles-to-EdgeAI-Axelera-Metis}
}
```

**Original Dataset Citation**:
```bibtex
@article{zhu2019so2sat,
  title        = {So2Sat LCZ42: A Benchmark Dataset for Global Local Climate Zone Classification},
  author       = {Zhu, Xiao Xiang and Hu, Jingliang and Qiu, Chunping and
                  Shi, Yilei and Kang, Jian and Mou, Lichao and Bagheri, Hossein and
                  Haberle, Matthias and Hua, Yuansheng and Huang, Rong and others},
  journal      = {IEEE Geoscience and Remote Sensing Letters},
  volume       = {17},
  number       = {6},
  pages        = {975--979},
  year         = {2019},
  publisher    = {IEEE},
  doi={10.1109/MGRS.2020.2964708},
}
```

---

## 👤 Author & Acknowledgments

**Matteo Rambaldi**
MSc in Artificial Intelligence and Robotics, University of Padua
📧 [matteo.rambaldi@studenti.unipd.it](mailto:matteo.rambaldi@studenti.unipd.it)
🔗 [GitHub](https://github.com/matteorambaldi) • [LinkedIn](https://linkedin.com/in/matteorambaldi)

**Supervision**:
- **Prof. Loris Nanni** — Department of Information Engineering, University of Padua
- **Eng. Cristian Garjitzky** — Axelera AI, Co-Supervisor

**Special Thanks**:
- TUM & DLR for the So2Sat LCZ42 dataset
- Axelera AI for providing Metis hardware access
- University of Padua for computational resources

---

## 🔗 Links

- [📄 Thesis PDF](https://github.com/matteorambaldi/Enhanced-LCZ42-Ensembles-to-EdgeAI-Axelera-Metis/releases)
- [📊 Zenodo Dataset](https://zenodo.org/)
- [🚀 Axelera AI](https://axelera.ai)
- [🌍 So2Sat LCZ42 Official Page](https://mediatum.ub.tum.de/1483140)

---

<div align="center">

**Copyright © 2025 Matteo Rambaldi**
Released under MIT License

*Bringing State-of-the-Art Deep Learning to Edge AI* 🚀

</div>
