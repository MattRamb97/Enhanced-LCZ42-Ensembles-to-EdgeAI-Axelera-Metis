"""
Compute CORRECT μ/σ for LCZ42 after paper scaling AND resize to 224x224.

The old compute_lcz42_mu_sigma.py had a bug: it computed statistics on 32x32
data, but DatasetReading with inputSize=(224, 224) computes on 224x224 data.

This script computes μ/σ the CORRECT way:
  1. Load 32x32 HDF5
  2. Apply paper scaling
  3. Resize to 224x224 (BEFORE computing stats)
  4. Compute μ/σ on 224x224 data

This matches exactly what happens during distillation training.
"""

from __future__ import annotations

from pathlib import Path
import sys

import cv2
import h5py
import numpy as np
from tqdm import tqdm
from scipy.io import loadmat
import pandas as pd

# Add resnet18_ensembles scripts to path
REPO_ROOT = Path(__file__).resolve().parents[1]
RESNET_SCRIPTS = REPO_ROOT / "resnet18_ensembles" / "scripts"
sys.path.insert(0, str(RESNET_SCRIPTS))

from dataset_reading import apply_paper_scaling, _read_h5_patch  # type: ignore

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
DATA_ROOT = REPO_ROOT / "data" / "lcz42"
TABLE_PATH = DATA_ROOT / "tables_MS.mat"
BAND_INDICES = (2, 1, 0)  # RGB order (B4/B3/B2)
RESIZE_TARGET = 224


def _matlab_to_scalar(value):
    """Convert MATLAB scalar to Python int."""
    if isinstance(value, np.ndarray):
        return int(value.reshape(-1)[0])
    return int(value)


def _matlab_to_string(value):
    """Convert MATLAB string to Python str."""
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"U", "S"}:
            return "".join(value.astype(str).reshape(-1))
        if value.dtype == object:
            return "".join(str(v) for v in value.reshape(-1))
    return str(value)


def load_training_table(table_path: Path) -> pd.DataFrame:
    """Load MATLAB training table."""
    mat_data = loadmat(table_path, simplify_cells=False)
    train_table_raw = mat_data["train_MS"]

    rows = []
    for row in train_table_raw:
        path = _matlab_to_string(row[0])
        label = _matlab_to_scalar(row[1])
        index = _matlab_to_scalar(row[2]) - 1  # MATLAB is 1-indexed
        modality = _matlab_to_string(row[3]).upper()

        # Resolve path relative to resnet18_ensembles/scripts
        if not Path(path).is_absolute():
            path = str((RESNET_SCRIPTS / path).resolve())

        rows.append({"Path": path, "Label": label, "Index": index, "Modality": modality})

    return pd.DataFrame(rows)


def compute_stats_224(train_table: pd.DataFrame, resize_size: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute μ/σ on 224x224 resized data (after paper scaling).

    This matches exactly what DatasetReading does when called with inputSize=(224, 224):
      1. Load 32x32 HDF5
      2. Apply paper scaling
      3. Resize to 224x224
      4. Compute statistics
    """
    acc_mu = None
    acc_sq = None
    n_pix = 0

    print(f"[INFO] Computing μ/σ on {resize_size}x{resize_size} data (AFTER resize)")
    print("[INFO] Pipeline: load 32x32 → paper_scale → resize_224x224 → compute_stats")

    for i, row in tqdm(enumerate(train_table.itertuples()), total=len(train_table), desc="Processing training data"):
        # Step 1: Load 32x32 HDF5
        patch = _read_h5_patch(row.Path, int(row.Index), row.Modality)
        X = patch.astype(np.float32)

        # Step 2: Apply paper scaling
        X = apply_paper_scaling(X, row.Modality)

        # Step 3: Extract RGB bands
        X = X[..., BAND_INDICES]

        # Step 4: Resize to 224x224 (using INTER_NEAREST like training)
        X_resized = cv2.resize(X, (resize_size, resize_size), interpolation=cv2.INTER_NEAREST)

        # Step 5: Accumulate statistics on the resized data
        flat = X_resized.reshape(-1, X_resized.shape[-1]).astype(np.float64)

        if acc_mu is None:
            acc_mu = np.zeros(flat.shape[1], dtype=np.float64)
            acc_sq = np.zeros(flat.shape[1], dtype=np.float64)

        acc_mu += flat.sum(axis=0)
        acc_sq += (flat ** 2).sum(axis=0)
        n_pix += flat.shape[0]

    # Compute final statistics
    mu = acc_mu / n_pix
    sigma = np.sqrt(np.maximum(acc_sq / n_pix - mu ** 2, 1e-12))

    return mu.astype(np.float32), sigma.astype(np.float32)


def main() -> None:
    print(f"[INFO] Repo root: {REPO_ROOT}")
    print(f"[INFO] Data root: {DATA_ROOT}")
    print(f"[INFO] Table path: {TABLE_PATH}")

    if not TABLE_PATH.exists():
        print(f"[ERROR] Table file not found: {TABLE_PATH}")
        return

    print(f"\n[INFO] Loading training table...")
    train_table = load_training_table(TABLE_PATH)
    print(f"[INFO] Training samples: {len(train_table)}")

    mu_224, sigma_224 = compute_stats_224(train_table, RESIZE_TARGET)

    print("\n" + "="*80)
    print("[RESULT] CORRECT μ/σ for 224x224 data (after paper scaling + resize):")
    print("="*80)
    print(f"\nμ (mean):")
    print(f"  {mu_224.tolist()}")
    print(f"\nσ (std):")
    print(f"  {sigma_224.tolist()}")

    # Print as YAML format
    print("\n[YAML FORMAT - Copy this to resnet18-imagenet-onnx.yaml]:")
    print(f"      mean: {', '.join(f'{v:.3f}' for v in mu_224)}")
    print(f"      std: {', '.join(f'{v:.3f}' for v in sigma_224)}")

    print("\n[INFO] This is the CORRECT approach because:")
    print("  1. Statistics are computed on 224x224 data (matching inputSize=(224,224))")
    print("  2. Paper scaling is applied BEFORE resize")
    print("  3. Resize uses INTER_NEAREST (matching training)")
    print("  4. Statistics are computed on the RESULT")


if __name__ == "__main__":
    main()
