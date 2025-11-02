# Enhanced-LCZ42-Ensembles-to-EdgeAI-Axelera-Metis

> Improved deep ensembles for Local Climate Zone mapping with knowledge distillation and deployment on Axelera Metis Edge AI accelerator.

## Project Overview
This repository extends the work of [Nanni & Brahnam, 2025] on **deep CNN ensembles for LCZ classification** (So2Sat LCZ42 dataset).  
We make two main contributions:
1. **Improving the ensemble baseline** with optimized training and fusion strategies for higher accuracy and robustness.  
2. **Compressing ensembles into efficient student models** via knowledge distillation, quantization, and pruning, enabling deployment on **Axelera Metis AI accelerator** for embedded applications.

## Dataset

**So2Sat LCZ42** — Sentinel-1/2 imagery paired with Local Climate Zone (LCZ) labels across 42 global cities.

| Split | Cities | Patches | Description |
|--------|---------|----------|--------------|
| Training | 32 | 352,366 | Used for teacher and SR model training |
| Testing | 10 | 24,188 | Held-out cities |
| Validation | — | — | Not used (would cause data leakage) |

**Source:** [TUM Data Services – So2Sat LCZ42 Dataset](https://dataserv.ub.tum.de/index.php/s/m1483140)  
**Reference:** Zhu, X. X., et al. *“So2Sat LCZ42: A Benchmark Dataset for Global Local Climate Zone Classification.”* IEEE GRSL, 2019.

## Repository Structure

```bash
Enhanced-LCZ42-Ensembles-to-EdgeAI-Axelera-Metis/
│
├── data/
│   └── lcz42/                 # So2Sat LCZ42 HDF5 files and .mat tables
│
├── densenet201_ensembles/     # Deep ensemble training (DenseNet201)
│
├── deployment/                # ONNX export and Metis EdgeAI integration
│
├── distillation/              # Knowledge distillation (ResNet18 students)
│
├── fusion_ensembles/          # DenseNet201 + TDA fusion (MS/SR/SAR)
│
├── matlab/                    # MATLAB scripts for preprocessing & teacher training
│   ├── densenet201_ensembles/
│   └── resnet18-ensembles/
│
├── resnet18_ensembles/        # Baseline ResNet18 teacher models (Rand/RGB/SAR)
│
├── super_resolution/          # VDSR, EDSR, ESRGAN, SwinIR, Real-ESRGAN, BSRNet
│   ├── bsrnet/
│   ├── edsr/
│   ├── esrgan/
│   ├── real_esrgan/
│   ├── swinir/
│   └── vdsr/
│
├── tda/                       # TDA feature extraction and fusion training
│   ├── data/
│   ├── models/
│   ├── scripts/
│   ├── results/
│   └── npy-matlab/
│
├── LICENSE
└── README.md
```

Each subfolder includes its own `README.md` describing internal scripts and usage.

## Reproducibility

1. Download LCZ42 data and place it under `data/lcz42/`
2. Reproduce teacher ensembles:
   - `densenet201_ensembles/` for baseline DenseNet teachers (Rand, RandRGB, SAR + full sum rule)
   - `resnet18_ensembles/` for comparable ResNet teachers
   - `fusion_ensembles/` for DenseNet201 + TDA fusion across baseline/SR inputs
3. (Optional) generate super-resolution upscales in `super_resolution/` and TDA descriptors in `tda/`
4. Distil/compress models (`distillation/`) and export to Metis (`deployment/`)

## Citation

If you use this repository or derived datasets in your research, please cite:

```bibtex
@mastersthesis{rambaldi2025enhancedlcz42,
  title        = {Enhanced LCZ42 Ensembles to Edge AI: Knowledge Distillation and Deployment on Axelera Metis},
  author       = {Rambaldi, Matteo},
  school       = {University of Padua},
  year         = {2025},
  note         = {GitHub Repository: https://github.com/matteorambaldi/Enhanced-LCZ42-Ensembles-to-EdgeAI-Axelera-Metis}
}
```

## License

This project is released under the MIT License.
See the LICENSE file for details.

## Author & Supervision

Matteo Rambaldi — MSc Artificial Intelligence, University of Padua\
Supervised by Prof. Loris Nanni\
Co-Supervisor: Eng. Cristian Garjitzky

## Links
- [Project Thesis (PDF)](https://github.com/matteorambaldi/Enhanced-LCZ42-Ensembles-to-EdgeAI-Axelera-Metis/releases)
- [Zenodo Dataset](https://zenodo.org/)
- [Axelera AI](https://axelera.ai)
