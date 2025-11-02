# ResNet18 Ensembles — LCZ42 Classification

This module contains training, evaluation, and result inspection code for **ResNet-18 ensemble models** applied to the So2Sat LCZ42 dataset using different Sentinel configurations.

---

## Configurations

Each ensemble is trained on a distinct input setting:

| Configuration | Input Channels | Description |
|---------------|----------------|-------------|
| `rand`        | Sentinel-2 (10 bands) | Multispectral only random 3 channels |
| `randrgb`     | Sentinel-2 (10 bands) | Multispectral RGB 3 channels |
| `sar`         | Sentinel-2 (10 bands) + Sentinel-1 (8 bands) | Multispectral 2 random channels + 1 SAR random channel |


---

## Directory Structure

```bash
resnet18_ensembles/
├── scripts/                  # Training, evaluation, and utilities
├── models/
│   └── trained/              # Saved model weights (.pth)
├── results/                  # Evaluation metrics, confusion matrices, plots
├── requirements.txt          # Dependencies for this module
└── README.md                 # (This file)


---

## Scripts Overview (`scripts/`)

| Script | Purpose |
|--------|----------|
| `rand_resnet18.py` | Train ensemble on Sentinel-2 (RAND) |
| `randrgb_resnet18.py` | Train ensemble on Sentinel-2 RGB (RANDRGB) |
| `randsar_resnet18.py` | Train ensemble on Sentinel-2 and Sentinel-1 (SAR) |
| `train_teacher_resnet18.py` | Launch training for any configuration (shared logic) |
| `dataset_reading.py` | Load HDF5 patches and labels |
| `h5_reader.py` | Utility for HDF5 I/O operations |
| `enable_gpu.py` | GPU selection helper |
| `inspect_results.py` | Plot confusion matrices and training curves |
| `inspect_tables.py` | View class distribution and patch metadata |
| `make_tables_from_h5.py` | Re-generate class tables from HDF5 files |
| `utils_results.py` | Result logging and metric calculation utilities |

## Trained Models (`models/trained/`)

| File | Input | Description |
|------|--------|-------------|
| `Rand_resnet18.pth` | Sentinel-2 | Trained 10-member ensemble on random 3 channels (/sen2) |
| `RandRGB_resnet18.pth` | Sentinel-2 | Trained 10-member ensemble on 2 random + 1 RGB channel (/sen2) |
| `SAR_resnet18.pth` | Sentinel-2 + Sentinel-1 | Trained 10-member ensemble on 2 random /sen2 + 1 random /sen1 |

---

## Results (`results/`)

The `results/` directory contains both visual and tabular performance summaries for each ensemble:

- **Figures:**  
  - Training accuracy/loss curves  
  - Confusion matrices  
- **Evaluation Files:**  
  - `*_history.csv` → Training logs (loss, accuracy per epoch)  
  - `*_summary.json` → Aggregated evaluation metrics (accuracy, F1, etc.)  
  - `*_eval_TEST.h5` → Per-sample test predictions  
  - `*_members.csv` → Logits of individual ensemble members  
- **Sum-rule fusion:**  
  - `results/<mode>/fusion_resnet18_<mode>_sumrule_*` ⟶ ensemble-average scores for that mode  
  - `results/fusion/resnet18_sumrule_*` ⟶ global fusion across RAND + RANDRGB + SAR  

---

## Requirements

Install dependencies inside your active virtual environment:

```bash
pip install -r requirements.txt
```

---

## Reproducibility
1. Place preprocessed `.h5` datasets under `data/lcz42/` (see root README)
2. Launch teachers (single configuration or ALL):

```bash
python scripts/train_teacher_resnet18.py --mode ALL
```
3. Inspect metrics and confusion matrices:

```bash
python scripts/inspect_results.py
```

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
