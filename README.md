# Enhanced-LCZ42-Ensembles-to-EdgeAI-Axelera-Metis

> Improved deep ensembles for Local Climate Zone mapping with knowledge distillation and deployment on Axelera Metis Edge AI accelerator.

---

## Project Overview
This repository extends the work of [Nanni & Brahnam, 2025] on **deep CNN ensembles for LCZ classification** (So2Sat LCZ42 dataset).  
We make two main contributions:
1. **Improving the ensemble baseline** with optimized training and fusion strategies for higher accuracy and robustness.  
2. **Compressing ensembles into efficient student models** via knowledge distillation, quantization, and pruning, enabling deployment on **Axelera Metis AI accelerator** for embedded applications.

---

## Dataset

**So2Sat LCZ42** — Sentinel-1/2 imagery paired with Local Climate Zone (LCZ) labels across 42 global cities.

| Split | Cities | Patches | Description |
|--------|---------|----------|--------------|
| Training | 32 | 352,366 | Used for teacher and SR model training |
| Testing | 10 | 24,188 | Held-out cities |
| Validation | — | — | Not used (would cause data leakage) |

**Source:** [TUM Data Services – So2Sat LCZ42 Dataset](https://dataserv.ub.tum.de/index.php/s/m1483140)  
**Reference:** Zhu, X. X., et al. *“So2Sat LCZ42: A Benchmark Dataset for Global Local Climate Zone Classification.”* IEEE GRSL, 2019.

---

## Repository Structure

Enhanced-LCZ42-Ensembles-to-EdgeAI-Axelera-Metis/
│
├── data/                # Placeholder for LCZ42 datasets (empty; linked via Zenodo)
├── deployment/          # ONNX export and Metis EdgeAI integration
├── distillation/        # Knowledge distillation and student model training
├── matlab/              # MATLAB preprocessing and ensemble creation
├── super-resolution/    # VDSR, EDSR, ESRGAN, SwinIR, BSRNet, Real-ESRGAN pipelines
├── TDA/                 # Topological Data Analysis
│
├── LICENSE              # MIT License
└── README.md            # (This file)

Each subfolder includes its own `README.md` describing internal scripts and usage.

---

## Reproducibility

1. Download LCZ42 data and place it under `data/lcz42/`:
   ```bash
   mkdir -p data/lcz42
2.	Run the preprocessing pipelines (MATLAB → Super-Resolution → Distillation).
3.	Export and deploy trained models using scripts in deployment/.

A full guide to reproducing results and exporting ONNX models is provided in
deployment/README.md.

---

## Citation

If you use this repository or derived datasets in your research, please cite:

@mastersthesis{rambaldi2025enhancedlcz42,
  title        = {Enhanced LCZ42 Ensembles to Edge AI: Knowledge Distillation and Deployment on Axelera Metis},
  author       = {Rambaldi, Matteo},
  school       = {University of Padua},
  year         = {2025},
  note         = {GitHub Repository: https://github.com/matteorambaldi/Enhanced-LCZ42-Ensembles-to-EdgeAI-Axelera-Metis}
}

---

## License

This project is released under the MIT License.
See the LICENSE file for details.

---

## Author & Supervision

Matteo Rambaldi — MSc Artificial Intelligence, University of Padua
Supervised by Prof. Loris Nanni
Co-Supervisor: Eng. Cristian Garjitzky

---

## Links
	•	[Project Thesis (PDF)] (https://github.com/matteorambaldi/Enhanced-LCZ42-Ensembles-to-EdgeAI-Axelera-Metis/releases)
	•	[Zenodo Dataset] (https://zenodo.org/)
	•	[Axelera AI] (https://axelera.ai)
