# LCZ42 Dataset

## Overview

This directory contains the **So2Sat LCZ42** dataset and its **super-resolution enhanced variants** used in this project for Local Climate Zone (LCZ) classification.

**So2Sat LCZ42** is a large-scale global benchmark containing co-registered **Sentinel-1** (SAR) and **Sentinel-2** (multispectral) patches from 42 cities worldwide.

**Reference:**
> Zhu, X. X., et al. *"So2Sat LCZ42: A Benchmark Dataset for Global Local Climate Zone Classification."* IEEE Geoscience and Remote Sensing Letters, 2019.
> [DOI: 10.1109/LGRS.2019.2919262](https://doi.org/10.1109/LGRS.2019.2919262)
>
> Official website: [http://www.so2sat.eu/](http://www.so2sat.eu/)

---

## Dataset Specifications

Each patch in the dataset contains:

| Property | Details |
|:---------|:--------|
| **Spatial Size** | 32 × 32 pixels |
| **Resolution** | 10 m × 10 m per pixel |
| **Sentinel-2 Channels** | 10 bands (multispectral) |
| **Sentinel-1 Channels** | 8 bands (SAR) |
| **Labels** | 17-class one-hot encoded Local Climate Zone |
| **Data Type** | float16 (reflectance-like scale [0, 2.8]) |

---

## Directory Structure

```
data/lcz42/
├── training.h5                 # Baseline training patches (352K samples)
├── testing.h5                  # Baseline test patches (24K samples)
├── tables_MS.mat               # Multispectral patch metadata
├── tables_SAR.mat              # SAR patch metadata
│
├── training_vdsr2x.h5          # Super-resolved variants (×2 upscaling)
├── training_edsr2x.h5
├── training_edsr4x.h5          # Super-resolved variants (×4 upscaling)
├── training_swinir2x.h5
├── training_bsrnet2x.h5
├── training_esrgan2x.h5
├── training_vdsr3x.h5          # Super-resolved variants (×3 upscaling)
├── training_realesrgan4x.h5
│
├── testing_vdsr2x.h5           # Corresponding test sets (all SR variants)
├── testing_edsr2x.h5
├── testing_edsr4x.h5
├── ...
│
└── README.md                   # This file
```

**Note:** All files are excluded from version control (`.gitignore`) due to large storage requirements (~527.1 GB total).

---

## Sentinel-1 (SAR) Bands

The Sentinel-1 data includes 8 bands representing SAR measurements:

1. Real part of VH complex signal
2. Imaginary part of VH complex signal
3. Real part of VV complex signal
4. Imaginary part of VV complex signal
5. Intensity of lee-filtered VH signal
6. Intensity of lee-filtered VV signal
7. Real part of lee-filtered PolSAR covariance off-diagonal element
8. Imaginary part of lee-filtered PolSAR covariance off-diagonal element

---

## Sentinel-2 (Multispectral) Bands

The Sentinel-2 data includes 10 bands covering visible, NIR, and SWIR regions:

| Index | Band Name | Wavelength | Purpose |
|:-----:|:----------|:-----------|:--------|
| 1 | B2 (Blue) | 490 nm | Visible light |
| 2 | B3 (Green) | 560 nm | Visible light |
| 3 | B4 (Red) | 665 nm | Visible light |
| 4 | B5 (Red Edge 1) | 705 nm | Vegetation analysis |
| 5 | B6 (Red Edge 2) | 740 nm | Vegetation analysis |
| 6 | B7 (Red Edge 3) | 783 nm | Vegetation analysis |
| 7 | B8 (NIR) | 842 nm | Vegetation measurement |
| 8 | B8A (NIR narrow) | 865 nm | Vegetation measurement |
| 9 | B11 (SWIR 1) | 1610 nm | Moisture/water content |
| 10 | B12 (SWIR 2) | 2190 nm | Moisture/water content |

For detailed information, see the [Sentinel-2 MSI User Guide](https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi/overview).

---

## Super-Resolution Enhanced Variants

This project introduces 8 super-resolution (SR) enhanced variants of the So2Sat LCZ42 dataset, designed to improve spectral–spatial detail before model training.

Each SR method is applied to **Sentinel-2 multispectral channels (10 bands)** while preserving the original LCZ labels.

### SR Methods Summary

| Method | Scale | Architecture | Framework | Training Data |
|:-------|:-----:|:------------|:----------|:--------------|
| Baseline | ×1 | — | — | `training.h5` / `testing.h5` |
| **VDSR** | ×2 | CNN | MATLAB | `training_vdsr2x.h5` / `testing_vdsr2x.h5` |
| **EDSR** | ×2 | CNN | PyTorch | `training_edsr2x.h5` / `testing_edsr2x.h5` |
| **ESRGAN** | ×2 | GAN | MATLAB | `training_esrgan2x.h5` / `testing_esrgan2x.h5` |
| **EDSR** | ×4 | CNN | PyTorch | `training_edsr4x.h5` / `testing_edsr4x.h5` |
| **SwinIR** | ×2 | Transformer | PyTorch | `training_swinir2x.h5` / `testing_swinir2x.h5` |
| **VDSR** | ×3 | CNN | MATLAB | `training_vdsr3x.h5` / `testing_vdsr3x.h5` |
| **BSRNet** | ×2 | Blind SR (CNN) | PyTorch | `training_bsrnet2x.h5` / `testing_bsrnet2x.h5` |
| **Real-ESRGAN** | ×4 | GAN | PyTorch | `training_realesrgan4x.h5` / `testing_realesrgan4x.h5` |

### HDF5 File Structure

All `.h5` files (baseline) follow this structure:

```python
# Reading example
import h5py

with h5py.File('training.h5', 'r') as f:
    labels = f['/label'][:]           # Shape: (352000, 17) - one-hot encoded LCZ
    sen2_data = f['/sen2'][:]         # Shape: (352000, 32, 32, 10) - Sentinel-2 bands
    sen1_data = f['/sen1'][:]         # Shape: (352000, 32, 32, 8) - Sentinel-1 bands
```

All `.h5` files (Super Resolution) follow this structure:

```python
# Reading example
import h5py

with h5py.File('training.h5', 'r') as f:
    labels = f['/label'][:]           # Shape: (352000, 17) - one-hot encoded LCZ
    sen2_data = f['/sen2'][:]         # Shape: (352000, 32, 32, 10) - Sentinel-2 bands
```

**Data Specifications:**
- `/label`: LCZ labels as one-hot encoded vectors (17 classes), `float64` for baseline, `float16` for SR
- `/sen2`: Sentinel-2 multispectral data in **[0, 2.8]** reflectance-like scale, `float64` for baseline, `float16` for SR
- `/sen1`: Sentinel-1 SAR data in **[-0.5, 0.5]** normalized scale, `float64`

---

## Data Access & Availability

Datasets are hosted on **Zenodo** for open access and reproducibility:

🔗 [Dataset on Zenodo](https://doi.org/10.xxxx/zenodo.xxxxxx) (link to be updated)

---

## Citation

If you use this dataset or any super-resolution variants in your research, please cite the original work:

```bibtex
@article{zhu2019so2sat,
  title={So2Sat LCZ42: A Benchmark Dataset for Global Local Climate Zone Classification},
  author={Zhu, Xiaoxiang and Hu, Jingliang and Qiu, Chunping and Shi, Yilei and Kang, Jian and
          Mou, Lichao and Bagheri, Hossein and Haeberle, Matthias and Hua, Yuansheng and
          Huang, Rong and others},
  journal={IEEE Geoscience and Remote Sensing Letters},
  volume={17},
  number={9},
  pages={1655--1659},
  year={2019},
  publisher={IEEE},
  doi={10.1109/MGRS.2020.2964708}
}
```

If you use the super-resolution enhanced variants, please also cite this thesis:

```bibtex
@mastersthesis{rambaldi2025enhancedlcz42,
  title={Enhanced LCZ42 Ensembles to Edge AI: Knowledge Distillation and Deployment on Axelera Metis},
  author={Rambaldi, Matteo},
  school={University of Padua},
  year={2025},
  note={GitHub Repository: https://github.com/matteorambaldi/Enhanced-LCZ42-Ensembles-to-EdgeAI-Axelera-Metis}
}
```

---

## License

**So2Sat LCZ42 Original Dataset:** © Technical University of Munich (TUM) / DLR (German Aerospace Center)

All super-resolution enhanced derivatives are released for **non-commercial research use only**. Refer to the original dataset's [terms of use](https://www.so2sat.eu/terms) for complete details.

---

## Author & Attribution

**Dataset Enhancement & Project:** Matteo Rambaldi

**Affiliation:** MSc Artificial Intelligence, University of Padua

**Supervision:** Prof. Loris Nanni

**Co-Supervision:** Eng. Cristian Garjitzky

**Original Dataset Authors:** Xiaoxiang Zhu, Jingliang Hu, Chunping Qiu, Yilei Shi, and others

**Original Dataset Affiliation:** Technical University of Munich (SiPEO), DLR