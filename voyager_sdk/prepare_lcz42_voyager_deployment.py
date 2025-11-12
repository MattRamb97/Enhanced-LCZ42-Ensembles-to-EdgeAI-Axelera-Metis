"""
Convert LCZ42 training/testing H5 files into Voyager SDK-friendly PNG folders.

DIFFERENCE from research eval_lcz42_png.py:
- Research eval: PNGs contain raw reflectance [0, 65535] uint16
  → preprocess_png() does: [0,1] → paper_scale → resize → z_score
- Voyager SDK: PNGs contain paper-scaled values [0, 255] uint8
  → Voyager SDK normalize does: (img - μ) / σ directly on [0, 255] values

This script pre-applies paper scaling to PNGs so Voyager SDK's normalize
step works correctly with the correct LCZ42 μ/σ values.
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import h5py
import numpy as np

# --------------------------------------------------------------------------------------
# Configuration (hardcoded for reproducibility)
# --------------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR.parent / "data" / "lcz42"
TRAIN_H5 = DATA_ROOT / "training.h5"
TEST_H5 = DATA_ROOT / "testing.h5"
OUTPUT_ROOT = BASE_DIR / "data" / "LCZ42"

RGB_IDX = (2, 1, 0)  # Sentinel-2 bands B4, B3, B2 (R,G,B)
REPR_PER_CLASS = 30  # Calibration images per class for Axelera
VAL_PER_CLASS = None  # None = export ALL validation images, not a random subset
SEED = 42

CLS_NAMES = [
    "01_CompactHighRise",
    "02_CompactMidRise",
    "03_CompactLowRise",
    "04_OpenHighRise",
    "05_OpenMidRise",
    "06_OpenLowRise",
    "07_LightweightLowRise",
    "08_LargeLowRise",
    "09_SparselyBuilt",
    "10_HeavyIndustry",
    "11_DenseTrees",
    "12_ScatteredTree",
    "13_BushScrub",
    "14_LowPlants",
    "15_BareRockOrPaved",
    "16_BareSoilOrSand",
    "17_Water",
]


def apply_paper_scaling(img: np.ndarray) -> np.ndarray:
    """Apply paper scaling formula: x / (2.8 / 255.0)

    Transforms reflectances from [0, 2.8] → [0, 255] range.
    """
    img = img.astype(np.float32)
    img = img / (2.8 / 255.0)
    return np.clip(img, 0.0, 255.0)


def load_split(h5_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        sen2 = f["/sen2"][:]
        labels = f["/label"][:]
    return sen2, labels


def save_rgb_voyager(image: np.ndarray, dest: Path) -> None:
    """Save as 8-bit PNG with paper scaling already applied.

    Input: raw reflectance values in [0, 1] range
    Output: 8-bit PNG at ORIGINAL 32x32 resolution (no resize)

    NOTE: Paper scaling is applied DURING save, so PNG values are [0, 255].
    Voyager SDK's normalize step will work directly on these paper-scaled values
    using the correct LCZ42 μ/σ.
    """
    img = np.clip(image, 0.0, 1.0)

    # Apply paper scaling: [0, 1] → [0, 255] in float
    img_scaled = apply_paper_scaling(img)

    # Convert to uint8 for 8-bit PNG encoding
    img_uint8 = img_scaled.astype(np.uint8)

    # Convert RGB to BGR for OpenCV
    img_bgr = img_uint8[..., ::-1]

    # Save as 8-bit PNG WITHOUT resizing (at original 32x32 resolution)
    cv2.imwrite(str(dest), img_bgr)


def select_indices(labels: np.ndarray, per_class: int | None, seed: int) -> dict[int, list[int]]:
    """Select indices for each class.

    Args:
        labels: One-hot encoded class labels
        per_class: Number of samples per class (None = all samples)
        seed: Random seed for shuffling (only used if per_class is not None)

    Returns:
        Dictionary mapping class index to list of sample indices
    """
    assignments = {cls: [] for cls in range(len(CLS_NAMES))}

    # If per_class is None, select all indices for each class
    if per_class is None:
        for idx in range(len(labels)):
            cls = int(np.argmax(labels[idx]))
            assignments[cls].append(idx)
    else:
        # Otherwise, randomly select per_class samples per class
        rng = random.Random(seed)
        idxs = list(range(len(labels)))
        rng.shuffle(idxs)
        for idx in idxs:
            cls = int(np.argmax(labels[idx]))
            if len(assignments[cls]) >= per_class:
                continue
            assignments[cls].append(idx)
            if all(len(v) >= per_class for v in assignments.values()):
                break

    return assignments


def export_repr(data: np.ndarray, assignments: dict[int, list[int]], dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    counter = 0
    for cls, idx_list in assignments.items():
        for idx in idx_list:
            rgb = data[idx][..., RGB_IDX]
            # Paper scaling is applied INSIDE save_rgb_voyager()
            save_rgb_voyager(rgb, dest_dir / f"class{cls+1:02d}_{counter:05d}.png")
            counter += 1


def export_val(data: np.ndarray, assignments: dict[int, list[int]], dest_dir: Path) -> None:
    for cls_idx, cls_name in enumerate(CLS_NAMES):
        cls_dir = dest_dir / cls_name
        cls_dir.mkdir(parents=True, exist_ok=True)
        for idx in assignments[cls_idx]:
            rgb = data[idx][..., RGB_IDX]
            # Paper scaling is applied INSIDE save_rgb_voyager()
            save_rgb_voyager(rgb, cls_dir / f"{idx:06d}.png")


def write_labels_file(path: Path) -> None:
    with open(path, "w") as f:
        for name in CLS_NAMES:
            f.write(f"{name}\n")


def main() -> None:
    output_root = OUTPUT_ROOT.resolve()
    repr_dir = output_root / "repr"
    val_dir = output_root / "val"
    repr_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading {TRAIN_H5.name}...")
    train_data, train_labels = load_split(TRAIN_H5)
    print(f"[INFO] Loading {TEST_H5.name}...")
    test_data, test_labels = load_split(TEST_H5)

    print(f"[INFO] Selecting {REPR_PER_CLASS} calibration images per class...")
    repr_assign = select_indices(train_labels, REPR_PER_CLASS, SEED)
    print(f"[INFO] Selecting {VAL_PER_CLASS} validation images per class...")
    val_assign = select_indices(test_labels, VAL_PER_CLASS, SEED + 1)

    print(f"[INFO] Exporting calibration images to {repr_dir}...")
    print(f"[INFO] (Paper scaling applied during PNG encoding)")
    export_repr(train_data, repr_assign, repr_dir)
    print(f"[INFO] Exporting validation images to {val_dir}...")
    print(f"[INFO] (Paper scaling applied during PNG encoding)")
    export_val(test_data, val_assign, val_dir)
    write_labels_file(output_root / "labels.txt")

    print(f"\n[DONE] Calibration images: {repr_dir}")
    print(f"[DONE] Validation images: {val_dir}")
    print(f"[DONE] Labels file: {output_root / 'labels.txt'}")
    print(f"\n[INFO] PNG values are [0, 255] uint8 with paper scaling pre-applied")
    print(f"[INFO] Ready for Voyager SDK deployment with correct LCZ42 μ/σ normalization")


if __name__ == "__main__":
    main()
