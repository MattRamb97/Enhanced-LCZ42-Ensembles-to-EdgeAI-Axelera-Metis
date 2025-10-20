# LCZ42 Dataset Structure

This directory is reserved for the **So2Sat LCZ42** dataset and its **super-resolution enhanced variants** used in this project.



## Directory Structure

```bash
data/
└── lcz42/
├── training.h5                 # Sentinel-1/2 training data + one-hot encoded LCZ labels
├── testing.h5                  # Sentinel-1/2 test data + one-hot encoded LCZ labels
├── tables_MS.mat               # Metadata for Sentinel-2 multispectral bands
├── tables_SAR.mat              # Metadata for Sentinel-1 SAR channels
├── training_vdsr2x.h5          # SR-enhanced dataset (VDSR ×2)
├── training_vdsr3x.h5          # SR-enhanced dataset (VDSR ×3)
├── training_edsr2x.h5          # SR-enhanced dataset (EDSR ×2)
├── training_edsr4x.h5          # SR-enhanced dataset (EDSR ×4)
├── training_esrgan2x.h5        # SR-enhanced dataset (ESRGAN ×2, triplet semantic)
├── training_swinir2x.h5        # SR-enhanced dataset (SwinIR ×2, Transformer)
├── training_bsrnet2x.h5        # SR-enhanced dataset (BSRNet ×2, blind SR)
├── training_realesrgan4x.h5    # SR-enhanced dataset (Real-ESRGAN ×4)
└── README.md                   # Dataset documentation (this file)
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
> Zhu, X. X., et al.  
> *“So2Sat LCZ42: A Benchmark Dataset for Global Local Climate Zone Classification.”*  
> IEEE Geoscience and Remote Sensing Letters, 2019.  
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

---

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

| ID | Method | Scale | Model Type | Implementation | Output File |
|:--:|:--------|:------|:------------|:----------------|:--------------------------------------|
| **1–2** | Baseline (No SR) | ×1 | — | — | `training.h5`, `testing.h5` |
| **3** | VDSR | ×2 | CNN | MATLAB | `training_vdsr2x.h5` |
| **4** | EDSR | ×2 | CNN | PyTorch | `training_edsr2x.h5` |
| **5** | ESRGAN (Triplet Semantic) | ×2 | GAN | MATLAB | `training_esrgan2x.h5` |
| **6** | EDSR | ×4 | CNN | PyTorch | `training_edsr4x.h5` |
| **7** | SwinIR | ×2 | Transformer | PyTorch | `training_swinir2x.h5` |
| **8** | VDSR | ×3 | CNN | MATLAB | `training_vdsr3x.h5` |
| **9** | BSRNet | ×2 | Blind SR (CNN) | PyTorch | `training_bsrnet2x.h5` |
| **10** | Real-ESRGAN | ×4 | GAN (High Magnification) | PyTorch | `training_realesrgan4x.h5` |

### Data structure

Each SR-enhanced dataset maintains the same structure as the original LCZ42 `.h5` files:

- `/sen2`: super-resolved Sentinel-2 multispectral data in **[0, 2.8]**, `float32`  
- `/label`: original LCZ **one-hot encoded** vectors (17 classes)

## Access

Public datasets, including the super-resolution enhanced versions, are hosted on **Zenodo** for open access and reproducibility:
[https://doi.org/10.xxxx/zenodo.xxxxxx](https://doi.org/10.xxxx/zenodo.xxxxxx)

## Citation (original dataset)

**Authors**: Xiaoxiang Zhu, Jingliang Hu, Chunping Qiu, Yilei Shi, Jian Kang, Lichao Mou, Hossein Bagheri, Matthias Haeberle, Yuansheng Hua, Rong Huang, Lloyd Hughes, Hao Li, Yao Sun, Guichen Zhang, Shiyao Han, Michael Schmitt, Yuanyuan Wang

**Affiliations**: Technical University of Munich (SiPEO), DLR (German Aerospace Center)

**Funding**: ERC Starting Grant “So2Sat: Big Data for 4D Global Urban Mapping”

## Maintainer

**Matteo Rambaldi** — University of Padua  •  MSc Artificial Intelligence and Robotics (2025)