"""
Convert LCZ42 training/testing H5 files into Voyager-friendly folders.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

RGB_IDX = (2, 1, 0)  # B4, B3, B2

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
    img = img.astype(np.float32)
    img = img / (2.8 / 255.0)
    return np.clip(img, 0.0, 255.0)


def load_split(h5_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        sen2 = f["/sen2"][:]
        labels = f["/label"][:]
    return sen2, labels


def save_rgb(image: np.ndarray, dest: Path) -> None:
    img = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8), mode="RGB")
    img = img.resize((224, 224), Image.BICUBIC)
    img.save(dest, quality=95)


def select_indices(labels: np.ndarray, per_class: int, seed: int) -> dict[int, list[int]]:
    rng = random.Random(seed)
    idxs = list(range(len(labels)))
    rng.shuffle(idxs)
    assignments = {cls: [] for cls in range(len(CLS_NAMES))}
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
            rgb = apply_paper_scaling(rgb)
            save_rgb(rgb, dest_dir / f"class{cls+1:02d}_{counter:05d}.png")
            counter += 1


def export_val(data: np.ndarray, assignments: dict[int, list[int]], dest_dir: Path) -> None:
    for cls_idx, cls_name in enumerate(CLS_NAMES):
        cls_dir = dest_dir / cls_name
        cls_dir.mkdir(parents=True, exist_ok=True)
        for idx in assignments[cls_idx]:
            rgb = data[idx][..., RGB_IDX]
            rgb = apply_paper_scaling(rgb)
            save_rgb(rgb, cls_dir / f"{idx:06d}.png")


def write_labels_file(path: Path) -> None:
    with open(path, "w") as f:
        for name in CLS_NAMES:
            f.write(f"{name}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LCZ42 dataset for Voyager SDK.")
    parser.add_argument("--train-h5", type=Path, default=Path("../data/lcz42/training.h5"))
    parser.add_argument("--test-h5", type=Path, default=Path("../data/lcz42/testing.h5"))
    parser.add_argument("--output-root", type=Path, default=Path("data/LCZ42"))
    parser.add_argument("--repr-per-class", type=int, default=30)
    parser.add_argument("--val-per-class", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    repr_dir = output_root / "repr"
    val_dir = output_root / "val"
    repr_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    train_data, train_labels = load_split(args.train_h5)
    test_data, test_labels = load_split(args.test_h5)

    repr_assign = select_indices(train_labels, args.repr_per_class, args.seed)
    val_assign = select_indices(test_labels, args.val_per_class, args.seed + 1)

    export_repr(train_data, repr_assign, repr_dir)
    export_val(test_data, val_assign, val_dir)
    write_labels_file(output_root / "labels.txt")

    print(f"[INFO] Calibration images at {repr_dir}")
    print(f"[INFO] Validation images at {val_dir}")
    print(f"[INFO] Labels file at {output_root / 'labels.txt'}")


if __name__ == "__main__":
    main()
