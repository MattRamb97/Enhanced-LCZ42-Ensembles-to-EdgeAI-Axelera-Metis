# LCZ42 Dataset Structure

This directory is reserved for the **So2Sat LCZ42** dataset, tables and its **super-resolution enhanced variants** used in this project.

## Directory Structure

```bash
data/
└── lcz42/
    ├── training.h5                 # Baseline Sentinel‑1/2 training patches + LCZ labels
    ├── testing.h5                  # Baseline Sentinel‑1/2 test patches + LCZ labels
    ├── tables_MS.mat               # Patch metadata (multispectral)
    ├── tables_SAR.mat              # Patch metadata (SAR)
    │
    ├── tables_*.h5                 # Patch metadata SR‑enhanced datasets
    │
    ├── training_*.h5               # SR‑enhanced datasets (various models/scales)
    │
    ├── testing_*.h5                # Corresponding SR‑enhanced test
    │
    └── README.md                   # (This file)
```
All files in this directory are excluded from version control (`.gitignore`) due to large storage requirements.



## Dataset Description

**So2Sat LCZ42** is a large-scale global benchmark for Local Climate Zone (LCZ) classification, containing co-registered **Sentinel-1** (SAR) and **Sentinel-2** (multispectral) patches from 42 cities worldwide.

Each patch:
- Size: **32×32 pixels**
- Sentinel-2 channels: **10 bands**
- Sentinel-1 channels: **8 bands**
- Label: **17-class one-hot encoded LCZ vector**
- Resolution: **10 m × 10 m per pixel**

**Reference:**
> Zhu, X. X., et al. *“So2Sat LCZ42: A Benchmark Dataset for Global Local Climate Zone Classification.”* IEEE Geoscience and Remote Sensing Letters, 2019.  
> [DOI: 10.1109/LGRS.2019.2919262](https://doi.org/10.1109/LGRS.2019.2919262)

Official project website: [http://www.so2sat.eu/](http://www.so2sat.eu/)

## Sentinel-1 (SEN1) Bands

1. Real part of VH complex signal  
2. Imaginary part of VH complex signal  
3. Real part of VV complex signal  
4. Imaginary part of VV complex signal  
5. Intensity of lee-filtered VH signal  
6. Intensity of lee-filtered VV signal  
7. Real part of lee-filtered PolSAR covariance off-diagonal element  
8. Imaginary part of lee-filtered PolSAR covariance off-diagonal element  

## Sentinel-2 (SEN2) Bands

1. B2 (Blue) 
2. B3 (Green) 
3. B4 (Red) 
4. B5 (Red Edge 1)  
5. B6 (Red Edge 2) 
6. B7 (Red Edge 3) 
7. B8 (NIR) 
8. B8A (NIR narrow)  
9. B11 (SWIR 1) 
10. B12 (SWIR 2)

Details: [Sentinel-2 MSI User Guide](https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi/overview)


## Super-Resolution Datasets

This project introduces several **super-resolution (SR) enhanced variants** of the So2Sat LCZ42 dataset, designed to improve spectral–spatial detail before model training.

Each SR method is applied to the **Sentinel-2 multispectral channels (10 bands)** while preserving the one-hot encoded LCZ labels.

### Summary of SR-enhanced datasets

| Method | Scale | Model Type | Implementation | Output File |
|:--------|:------|:------------|:----------------|:--------------------------------------|
| Baseline (No SR) | ×1 | — | — | `training.h5`, `testing.h5` |
| VDSR | ×2 | CNN | MATLAB | `training_vdsr2x.h5`, `testing_vdsr2x.h5` |
| EDSR | ×2 | CNN | PyTorch | `training_edsr2x.h5`, `testing_edsr2x.h5` |
| ESRGAN (Triplet-based Semantic) | ×2 | GAN | MATLAB | `training_esrgan2x.h5`, `testing_esrgan2x.h5` |
| EDSR | ×4 | CNN | PyTorch | `training_edsr4x.h5`,  `testing_edsr4x.h5` |
| SwinIR | ×2 | Transformer | PyTorch | `training_swinir2x.h5`, `testing_swinir2x.h5` |
| VDSR | ×3 | CNN | MATLAB | `training_vdsr3x.h5`, `testing_vdsr3x.h5` |
| BSRNet | ×2 | Blind SR (CNN) | PyTorch | `training_bsrnet2x.h5`, `testing_bsrnet2x.h5` |
| Real-ESRGAN | ×4 | GAN (High Magnification) | PyTorch | `training_realesrgan4x.h5`, `testing_realesrgan4x.h5` |

### Data structure

Each SR-enhanced dataset maintains the same structure as the original LCZ42 `.h5` files:

- `/label`: original LCZ **one-hot encoded** vectors (17 classes)
- `/sen2`: super-resolved Sentinel-2 multispectral data in **[0, 2.8]** reflectance-like scale, `float16`

## Access

Public datasets, including the super-resolution enhanced versions, are hosted on **Zenodo** for open access and reproducibility:
[https://doi.org/10.xxxx/zenodo.xxxxxx](https://doi.org/10.xxxx/zenodo.xxxxxx)

## Citation (original dataset)

**Authors**: Xiaoxiang Zhu, Jingliang Hu, Chunping Qiu, Yilei Shi, Jian Kang, Lichao Mou, Hossein Bagheri, Matthias Haeberle, Yuansheng Hua, Rong Huang, Lloyd Hughes, Hao Li, Yao Sun, Guichen Zhang, Shiyao Han, Michael Schmitt, Yuanyuan Wang

**Affiliations**: Technical University of Munich (SiPEO), DLR (German Aerospace Center)

**Funding**: ERC Starting Grant “So2Sat: Big Data for 4D Global Urban Mapping”

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

## License Original Dataset So2Sat LCZ42

The original So2Sat LCZ42 dataset is © TUM/DLR and distributed under their specified terms.  
All super-resolved derivatives included here are released for **non-commercial research use** only.

Refer to the original dataset license and publication for attribution.