"""
Compute per-channel mean/std for LCZ42 Sentinel-2 patches after paper scaling.

This mirrors the logic used inside resnet18_ensembles/scripts/dataset_reading.py
so that we can verify the μ/σ being used during training/distillation.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

# --------------------------------------------------------------------------------------
# Configuration (hardcoded for reproducibility)
# --------------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR.parent / "data" / "lcz42"
H5_FILES = [DATA_ROOT / "training.h5"]  # Compute on training set only (same as DatasetReading)
# NOTE: This ensures μ/σ match what's computed during distillation training
BAND_INDICES = (2, 1, 0)  # Sentinel-2 B4/B3/B2 (R/G/B)
RESIZE_TARGET = 224


def apply_paper_scaling_ms(patch: np.ndarray) -> np.ndarray:
    """Same scaling as DatasetReading (MS modality)."""
    patch = patch.astype(np.float32)
    patch = patch / (2.8 / 255.0)
    return np.clip(patch, 0.0, 255.0)


def compute_stats(files: list[Path], band_indices: tuple[int, int, int], size: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute μ/σ BEFORE resizing to match DatasetReading pipeline exactly."""
    acc_mu = None
    acc_sq = None
    total_pixels = 0

    for h5_path in files:
        with h5py.File(h5_path, "r") as f:
            sen2 = f["/sen2"]
            for patch in tqdm(sen2, desc=f"Processing {h5_path.name}"):
                # Extract bands and apply paper scaling (32x32, not resized)
                subset = patch[..., band_indices]  # RGB extraction from 10 bands
                subset = apply_paper_scaling_ms(subset)
                # NOTE: Do NOT resize here - compute stats on original 32x32 size
                # This matches DatasetReading which computes stats on inputSize=(32,32)
                flat = subset.reshape(-1, subset.shape[-1]).astype(np.float64)

                if acc_mu is None:
                    acc_mu = np.zeros(flat.shape[1], dtype=np.float64)
                    acc_sq = np.zeros(flat.shape[1], dtype=np.float64)

                acc_mu += flat.sum(axis=0)
                acc_sq += (flat ** 2).sum(axis=0)
                total_pixels += flat.shape[0]

    mu = acc_mu / total_pixels
    sigma = np.sqrt(np.maximum(acc_sq / total_pixels - mu ** 2, 1e-12))
    return mu.astype(np.float32), sigma.astype(np.float32)


def main() -> None:
    files = [f.resolve() for f in H5_FILES]
    mu, sigma = compute_stats(files, BAND_INDICES, RESIZE_TARGET)
    print("[INFO] Files          :", ", ".join(str(f.name) for f in files))
    print("[INFO] Band indices   :", BAND_INDICES)
    print("[INFO] Resize target  :", RESIZE_TARGET)
    print("[RESULT] μ :", mu.tolist())
    print("[RESULT] σ :", sigma.tolist())


if __name__ == "__main__":
    main()
