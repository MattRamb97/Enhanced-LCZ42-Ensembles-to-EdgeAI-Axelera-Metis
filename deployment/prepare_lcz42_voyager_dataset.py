"""
Convert LCZ42 training/testing H5 files into Voyager-friendly folders.

Output tree (under --output-root):
  repr/                 -> calibration images (flattened)
  val/<class_id>/       -> validation images grouped per class (1..17)
  labels.txt            -> simple class names ("LCZ_1", ...)
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

# Dataset order: [B2, B3, B4, ...] -> RGB = [B4, B3, B2]
RGB_IDX = (2, 1, 0)
LCZ_CLASSES = 17


def apply_paper_scaling(img: np.ndarray) -> np.ndarray:
    """Identical to DatasetReading applyPaperScaling for Sentinel-2."""
    img = img.astype(np.float32)
    img = img / (2.8 / 255.0)
    return np.clip(img, 0.0, 255.0)


def load_split(h5_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        sen2 = f["/sen2"][:]         # (N, H, W, 10)
        labels = f["/label"][:]     # (N, 17) one-hot
    return sen2, labels


def save_rgb(image: np.ndarray, dest: Path) -> None:
    img = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8), mode="RGB")
    img = img.resize((224, 224), Image.BICUBIC)
    img.save(dest, quality=95)


def select_indices(labels: np.ndarray, per_class: int, seed: int) -> list[int]:
    idxs = list(range(len(labels)))
    random.Random(seed).shuffle(idxs)
    counts = [0] * LCZ_CLASSES
    output = []
    for idx in idxs:
        cls = int(np.argmax(labels[idx]))
        if counts[cls] >= per_class:
            continue
        counts[cls] += 1
        output.append(idx)
        if all(c >= per_class for c in counts):
            break
    return output


def export_split(
    data: np.ndarray,
    labels: np.ndarray,
    indices: list[int],
    dest_dir: Path,
    grouped: bool,
) -> None:
    if grouped:
        for cls_id in range(1, LCZ_CLASSES + 1):
            (dest_dir / str(cls_id)).mkdir(parents=True, exist_ok=True)
    else:
        dest_dir.mkdir(parents=True, exist_ok=True)

    for i, idx in enumerate(indices):
        cls = int(np.argmax(labels[idx])) + 1  # 1-based
        rgb = data[idx][..., RGB_IDX]
        rgb = apply_paper_scaling(rgb)
        target = dest_dir / (f"class{cls:02d}_{i:05d}.png" if not grouped else f"{i:06d}.png")
        if grouped:
            target = dest_dir / str(cls) / f"{idx:06d}.png"
        save_rgb(rgb, target)


def write_labels_file(path: Path) -> None:
    with open(path, "w") as f:
        for cls in range(1, LCZ_CLASSES + 1):
            f.write(f"LCZ_{cls}\n")


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

    repr_indices = select_indices(train_labels, args.repr_per_class, args.seed)
    val_indices = select_indices(test_labels, args.val_per_class, args.seed + 1)

    export_split(train_data, train_labels, repr_indices, repr_dir, grouped=False)
    export_split(test_data, test_labels, val_indices, val_dir, grouped=True)
    write_labels_file(output_root / "labels.txt")

    print(f"[INFO] Calibration images written to {repr_dir}")
    print(f"[INFO] Validation images written to {val_dir}")
    print(f"[INFO] Labels file at {output_root / 'labels.txt'}")


if __name__ == "__main__":
    main()
