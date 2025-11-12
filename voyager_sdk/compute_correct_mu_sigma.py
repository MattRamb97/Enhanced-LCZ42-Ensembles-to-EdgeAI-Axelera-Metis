"""
Compute correct μ/σ for Voyager SDK post-resize normalization.

PROBLEM: Training pipeline applies normalization BEFORE resize:
  1. Load 32x32 HDF5
  2. Apply paper scaling → [0, 255]
  3. Apply z-score: (X - μ_32x32) / σ_32x32  (on 32x32 data!)
  4. Resize to 224x224

But Voyager SDK applies normalization AFTER resize:
  1. Load PNG 32x32 [0, 255]
  2. Resize to 224x224
  3. Apply z-score: (X - μ_224x224) / σ_224x224  (on 224x224 data)

SOLUTION: Simulate the EXACT training pipeline on training data:
  1. Load + paper scale + normalize 32x32
  2. Resize to 224x224
  3. Compute μ/σ on the RESULT
  These new μ/σ values can be used in post-resize normalization to match training.
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

# These are the current μ/σ computed on 32x32 pre-resize data (from compute_band_stats)
# We need to get the ACTUAL values from the training code
MU_32x32 = np.array([9.206003189086914, 9.952054977416992, 11.270723342895508], dtype=np.float32)
SIGMA_32x32 = np.array([6.044061660766602, 4.3512349128723145, 3.605332136154175], dtype=np.float32)


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


def compute_correct_stats(train_table: pd.DataFrame, mu_32x32: np.ndarray, sigma_32x32: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate the exact training pipeline:
    1. Load 32x32 HDF5
    2. Apply paper scaling
    3. Apply z-score normalization (32x32)
    4. Resize to 224x224
    5. Compute μ/σ on the normalized+resized data
    """
    acc_mu = None
    acc_sq = None
    n_pix = 0

    print("[INFO] Simulating training pipeline: load → paper_scale → normalize_32x32 → resize_224x224 → compute_stats")

    for i, row in tqdm(enumerate(train_table.itertuples()), total=len(train_table), desc="Processing training data"):
        # Step 1: Load 32x32 HDF5
        patch = _read_h5_patch(row.Path, int(row.Index), row.Modality)
        X = patch.astype(np.float32)

        # Step 2: Apply paper scaling
        X = apply_paper_scaling(X, row.Modality)

        # Step 3: Extract RGB bands
        X = X[..., BAND_INDICES]

        # Step 4: Apply z-score normalization on 32x32 data (as done in training)
        X = (X - mu_32x32) / (sigma_32x32 + 1e-6)

        # Step 5: Resize to 224x224 (using INTER_NEAREST like training)
        X_resized = cv2.resize(X, (224, 224), interpolation=cv2.INTER_NEAREST)

        # Step 6: Accumulate statistics on the resized, normalized data
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

    print(f"\n[INFO] Using pre-computed 32x32 μ/σ:")
    print(f"  μ_32x32: {MU_32x32.tolist()}")
    print(f"  σ_32x32: {SIGMA_32x32.tolist()}")

    mu_224x224, sigma_224x224 = compute_correct_stats(train_table, MU_32x32, SIGMA_32x32)

    print("\n" + "="*80)
    print("[RESULT] Correct μ/σ for POST-RESIZE normalization in Voyager SDK:")
    print("="*80)
    print(f"\nμ (mean) on normalized+resized 224x224 data:")
    print(f"  {mu_224x224.tolist()}")
    print(f"\nσ (std) on normalized+resized 224x224 data:")
    print(f"  {sigma_224x224.tolist()}")

    # Print as YAML format
    print("\n[YAML FORMAT - Copy this to resnet18-imagenet-onnx.yaml]:")
    print(f"      mean: {', '.join(f'{v:.3f}' for v in mu_224x224)}")
    print(f"      std: {', '.join(f'{v:.3f}' for v in sigma_224x224)}")

    print("\n[INFO] This accounts for:")
    print("  1. Z-score normalization applied on 32x32 data")
    print("  2. Resizing to 224x224 with INTER_NEAREST")
    print("  3. The combined effect of these operations")
    print("\nUsing these μ/σ in post-resize normalization will match the training pipeline!")


if __name__ == "__main__":
    main()
