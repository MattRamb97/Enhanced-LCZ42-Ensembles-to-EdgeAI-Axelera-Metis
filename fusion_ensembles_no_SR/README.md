# Fusion DenseNet201 + TDA — LCZ42 Classification

This module trains **DenseNet201 + TDA fusion ensembles** on the So2Sat LCZ42 dataset.  
Each model combines multispectral (MS) patches with pre-computed topological descriptors (TDA) to improve LCZ recognition, and supports:

- `RAND` : three Sentinel‑2 bands picked at random
- `RANDRGB` : two random Sentinel‑2 bands + one RGB band
- `SAR` : two random Sentinel‑2 bands + one Sentinel‑1 band (MS + SAR fusion)

For each configuration we evaluate:

1. Every super-resolution variant (baseline1, baseline2, VDSR/EDSR/ESRGAN/SwinIR/Real-ESRGAN/BSRNet)
2. The ensemble sum rule across all members for that configuration
3. A global sum rule `fusion_densenet201_full_sumrule` spanning RAND + RANDRGB + SAR

---

## Directory Layout

```bash
fusion_ensembles/
├── scripts/
│   ├── fusion_densenet201.py      # DenseNet201 + TDA fusion backbone
│   ├── rand_fusion.py             # Shared trainer for RAND / RANDRGB / SAR
│   ├── train_teacher_fusion.py    # Orchestrates all configurations & sum rules
│   ├── dataset_reading.py         # HDF5 reader with paper-style scaling
│   └── utils_results.py           # HDF5 logging utility
├── models/
│   └── trained/                   # Saved weights (.pth)
├── results/                       # Evaluation metrics (per config + fusion)
└── README.md
```

---

## Running the Teacher Ensembles

1. Ensure the LCZ42 `.mat` tables live in `data/lcz42/` and the TDA descriptors in `TDA/data/`
2. Activate your environment and install requirements (`pip install -r requirements.txt`)
3. Launch training:

```bash
python scripts/train_teacher_fusion.py --mode ALL
```

The script will:

- iterate over all MS baselines & SR variants for RAND and RANDRGB
- iterate over baseline Sentinel-1 tables for SAR
- train a DenseNet201+TDA model per variant
- save metrics/curves in `results/<mode>/`
- emit ensemble sum-rule reports in `results/<mode>/<mode>_sumrule_*`
- emit the global fusion report in `results/fusion/fusion_densenet201_full_sumrule_*`

To run a single configuration (`RAND`, `RANDRGB`, or `SAR`):

```bash
python scripts/train_teacher_fusion.py --mode RANDRGB
```

---

## Outputs

For each trained model you will find in `results/<mode>/`:

- `*.pth` — saved weights (under `models/trained/`)
- `*_history.csv` — per-epoch loss/accuracy
- `*_summary.json` — Top‑1, accuracy, macro/weighted precision/recall/F1, selected bands
- `*_eval.h5` — predictions, confusion matrix, and metadata (compatible with MATLAB tooling)

The sum-rule evaluation files follow the same structure, with `components` listing the contributing members.

---

## Notes

- The trainer automatically ensures MS/SAR samples remain index-aligned with the TDA descriptors.
- SAR mode reuses the MS TDA features; only the raster input switches to a mixed MS+SAR tensor.
- Batch size defaults to 512; adjust `BATCH_SIZE` in `train_teacher_fusion.py` if required by GPU memory.

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
