#!/usr/bin/env python3
"""
Create a 2×9 Sentinel-2 super-resolution comparison figure (full view + zoom).

The script replicates the exact visualization pipeline tested in
`data/lcz42/test.py`: extract the Sentinel-2 true-color bands (B4, B3, B2),
clip to the physical reflectance range [0, 2.8], apply a global percentile
stretch computed from the original LR patch, and finally apply gamma
correction (default γ=2.2). An optional grayscale flag lets you convert the
normalized RGB preview to luminance-only panels for tonal inspection.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# Sentinel-2 band indices inside LCZ42 patches (H×W×10).
RGB_BANDS: Tuple[int, int, int] = (3, 2, 1)  # B4 (red), B3 (green), B2 (blue)
PHYSICAL_RANGE: Tuple[float, float] = (0.0, 2.8)
GRAYSCALE_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)
CROP_SIZE_LR = 12  # pixels on the original 32×32 grid
ZOOM_DISPLAY_SIZE: Tuple[int, int] = (160, 160)
FIGURE_DPI = 300
FONT_SIZE = 8
DEFAULT_OUTPUT = "super_resolution_comparison.pdf"
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DATA_ROOT = REPO_ROOT / "data" / "lcz42"


@dataclass
class VariantSpec:
    arg_name: str
    title: str
    scale: int
    default_path: Path
    interp: int = cv2.INTER_CUBIC


VARIANTS: List[VariantSpec] = [
    VariantSpec(
        "original",
        "Original (32×32)",
        1,
        DATA_ROOT / "training.h5",
        cv2.INTER_NEAREST,
    ),
    VariantSpec(
        "vdsr2x",
        "VDSR ×2",
        2,
        DATA_ROOT / "training_vdsr2x.h5",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a publication-ready Sentinel-2 SR comparison figure."
    )
    for variant in VARIANTS:
        parser.add_argument(
            f"--{variant.arg_name}",
            type=Path,
            default=variant.default_path,
            help=f"HDF5 file containing the {variant.title} patch set.",
        )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="sen2",
        help="Dataset name inside each HDF5 file (default: sen2).",
    )
    parser.add_argument(
        "--label-dataset",
        type=str,
        default="label",
        help="Dataset name containing LCZ one-hot labels (default: label).",
    )
    parser.add_argument(
        "--target-labels",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Preferred LCZ class IDs (1–17). Used when --auto-select is enabled.",
    )
    parser.add_argument(
        "--auto-select",
        dest="auto_select",
        action="store_true",
        help="Automatically choose the first patch whose LCZ label is in target-labels.",
    )
    parser.add_argument(
        "--manual-select",
        dest="auto_select",
        action="store_false",
        help="Disable auto-selection and use --patch-index directly.",
    )
    parser.set_defaults(auto_select=True)
    parser.add_argument(
        "--patch-index",
        type=int,
        default=0,
        help="Patch index to visualize (same index is taken from every file).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(DEFAULT_OUTPUT),
        help="Output PDF filename.",
    )
    parser.add_argument(
        "--min-value",
        type=float,
        default=PHYSICAL_RANGE[0],
        help="Lower bound for reflectance clipping before normalization.",
    )
    parser.add_argument(
        "--max-value",
        type=float,
        default=PHYSICAL_RANGE[1],
        help="Upper bound for reflectance clipping before normalization.",
    )
    parser.add_argument(
        "--contrast-low",
        type=float,
        default=0.5,
        help="Lower percentile (0–100) for the global stretch (default 1).",
    )
    parser.add_argument(
        "--contrast-high",
        type=float,
        default=99.0,
        help="Upper percentile (0–100) for the global stretch (default 99).",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.2,
        help="Gamma correction applied after stretching (set ≤0 to disable).",
    )
    parser.add_argument(
        "--grayscale",
        action="store_true",
        help="Convert normalized RGB panels to grayscale after gamma.",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=CROP_SIZE_LR,
        help="Crop size defined on the original 32×32 LR grid.",
    )
    args = parser.parse_args()
    if args.patch_index < 0:
        parser.error("patch-index must be non-negative.")
    if args.contrast_high <= args.contrast_low:
        parser.error("contrast-high must be greater than contrast-low.")
    if args.max_value <= args.min_value:
        parser.error("max-value must be greater than min-value.")
    for variant in VARIANTS:
        path: Path = getattr(args, variant.arg_name)
        resolved = path.expanduser()
        if not resolved.exists():
            parser.error(f"File not found for --{variant.arg_name}: {resolved}")
        setattr(args, variant.arg_name, resolved)
    args.target_labels = normalize_target_labels(args.target_labels)
    return args


def load_patch(path: Path, dataset: str, index: int) -> np.ndarray:
    """Load a multispectral patch from an LCZ42-style HDF5 file."""
    path = path.expanduser()
    if not path.exists():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as h5f:
        if dataset not in h5f:
            raise KeyError(f"Dataset '{dataset}' not found in {path}")
        ds = h5f[dataset]
        if index >= ds.shape[0]:
            raise IndexError(
                f"Index {index} out of range for dataset '{dataset}' ({ds.shape[0]} patches)."
            )
        patch = np.asarray(ds[index], dtype=np.float32)

    if patch.ndim != 3:
        raise ValueError(f"Expected H×W×C patch, got {patch.shape} in {path}")
    if patch.shape[-1] in (3, 10):
        return patch
    if patch.shape[0] in (3, 10):  # channel-first → convert to channel-last
        return np.transpose(patch, (1, 2, 0))
    raise ValueError(f"Cannot infer channel dimension for patch shape {patch.shape} in {path}")


def extract_rgb(
    cube: np.ndarray,
    clip_range: Tuple[float, float],
) -> np.ndarray:
    """Extract Sentinel-2 B4/B3/B2, clip to range, and scale to [0, 1]."""
    if cube.shape[-1] <= max(RGB_BANDS):
        raise ValueError("Cube does not contain enough bands for RGB extraction.")
    rgb = cube[..., RGB_BANDS]
    vmin, vmax = clip_range
    rgb = np.clip(rgb, vmin, vmax)
    rgb = (rgb - vmin) / (vmax - vmin + 1e-6)
    return np.clip(rgb, 0.0, 1.0).astype(np.float32, copy=False)


def stretch_rgb(
    rgb: np.ndarray,
    low: float,
    high: float,
) -> np.ndarray:
    """Apply global percentile stretch (single low/high pair)."""
    rgb = np.clip(rgb, low, high)
    rgb = (rgb - low) / (high - low + 1e-6)
    return np.clip(rgb, 0.0, 1.0).astype(np.float32, copy=False)


def apply_gamma(rgb: np.ndarray, gamma: float) -> np.ndarray:
    """Apply gamma correction to enhance mid-tones."""
    if gamma is None or gamma <= 0:
        return rgb
    return np.clip(rgb, 0.0, 1.0) ** (1.0 / gamma)


def to_grayscale(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB image to luminance grayscale (keeps 3 channels for plotting)."""
    luminance = np.tensordot(rgb, GRAYSCALE_WEIGHTS, axes=([-1], [0]))
    return np.repeat(luminance[..., None], 3, axis=-1).astype(np.float32, copy=False)


def format_patch(
    cube: np.ndarray,
    clip_range: Tuple[float, float],
    stretch_low: float,
    stretch_high: float,
    gamma: float,
    grayscale: bool,
) -> np.ndarray:
    """Full pipeline used for every patch."""
    rgb = extract_rgb(cube, clip_range)
    rgb = stretch_rgb(rgb, stretch_low, stretch_high)
    rgb = apply_gamma(rgb, gamma)
    if grayscale:
        rgb = to_grayscale(rgb)
    return rgb


def resize_image(image: np.ndarray, size_hw: Tuple[int, int], interpolation: int) -> np.ndarray:
    height, width = size_hw
    return cv2.resize(image, (width, height), interpolation=interpolation)


def extract_center_crop(image: np.ndarray, crop_size: int) -> np.ndarray:
    h, w = image.shape[:2]
    crop = min(crop_size, h, w)
    top = (h - crop) // 2
    left = (w - crop) // 2
    return image[top : top + crop, left : left + crop]


def normalize_target_labels(labels: List[int]) -> List[int]:
    """Convert LCZ IDs (1–17) to zero-based class indices."""
    normalized: List[int] = []
    for label in labels:
        if label < 0:
            continue
        if 1 <= label <= 17:
            normalized.append(label - 1)
        else:
            normalized.append(label)
    return list(sorted(set(normalized)))


def auto_select_patch(
    path: Path,
    label_dataset: str,
    target_classes: List[int],
    batch_size: int = 2048,
) -> int:
    """Return the first patch index whose LCZ label is among target_classes."""
    if not target_classes:
        raise ValueError("target_classes list is empty.")
    with h5py.File(path, "r") as h5f:
        if label_dataset not in h5f:
            raise KeyError(f"Dataset '{label_dataset}' not found in {path}")
        labels = h5f[label_dataset]
        total = labels.shape[0]
        for start in range(0, total, batch_size):
            batch = np.asarray(labels[start : start + batch_size])
            classes = np.argmax(batch, axis=1)
            mask = np.isin(classes, target_classes)
            if np.any(mask):
                first = np.flatnonzero(mask)[0]
                return int(start + first)
    raise ValueError(
        f"No patch found in {path} for labels {target_classes}. "
        f"Consider relaxing --target-labels or disabling --auto-select."
    )


def prepare_entries(
    args: argparse.Namespace,
    clip_range: Tuple[float, float],
    stretch_low: float,
    stretch_high: float,
) -> List[Dict]:
    entries: List[Dict] = []
    for variant in VARIANTS:
        cube = load_patch(getattr(args, variant.arg_name), args.dataset_name, args.patch_index)
        rgb = format_patch(
            cube,
            clip_range,
            stretch_low,
            stretch_high,
            args.gamma,
            args.grayscale,
        )
        entries.append(
            {
                "title": variant.title,
                "rgb": rgb,
                "interp": variant.interp,
                "scale": variant.scale,
                "native_size": rgb.shape[:2],
            }
        )

    max_h = max(entry["rgb"].shape[0] for entry in entries)
    max_w = max(entry["rgb"].shape[1] for entry in entries)
    target_size = (max_h, max_w)

    for entry in entries:
        entry["full"] = resize_image(entry["rgb"], target_size, entry["interp"])

    return entries


def render_figure(entries: Iterable[Dict], output_path: Path) -> None:
    entries = list(entries)
    rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "DejaVu Sans", "Arial"],
            "font.size": FONT_SIZE,
        }
    )

    fig, axes = plt.subplots(
        1,
        len(entries),
        figsize=(len(entries) * 1.6, 1.8),
        constrained_layout=False,
    )
    axes = np.atleast_1d(axes)
    plt.subplots_adjust(wspace=0.08, hspace=0.0)

    for ax, entry in zip(axes, entries):
        ax.imshow(entry["full"])
        ax.set_title(entry["title"], pad=4)
        ax.axis("off")

    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    clip_range = (args.min_value, args.max_value)

    if args.auto_select:
        selected = auto_select_patch(
            args.original,
            args.label_dataset,
            args.target_labels,
        )
        args.patch_index = selected
        printable_labels = ", ".join(str(lbl + 1) for lbl in args.target_labels)
        print(
            f"[INFO] Auto-selected patch index {selected} "
            f"(LCZ in [{printable_labels}])."
        )

    # Compute global stretch from the original LR patch (same as test.py)
    base_cube = load_patch(
        getattr(args, VARIANTS[0].arg_name),
        args.dataset_name,
        args.patch_index,
    )
    base_rgb = extract_rgb(base_cube, clip_range)
    stretch_low, stretch_high = np.percentile(
        base_rgb,
        [args.contrast_low, args.contrast_high],
    )
    if not np.isfinite(stretch_low) or not np.isfinite(stretch_high) or stretch_high <= stretch_low:
        stretch_low, stretch_high = 0.0, 1.0

    print(
        f"[INFO] Stretch percentiles ({args.contrast_low:.1f}, {args.contrast_high:.1f}) "
        f"→ [{stretch_low:.4f}, {stretch_high:.4f}]"
    )
    if args.gamma > 0:
        print(f"[INFO] Gamma correction: γ = {args.gamma}")
    if args.grayscale:
        print("[INFO] Grayscale mode enabled (post-gamma luminance).")

    entries = prepare_entries(args, clip_range, float(stretch_low), float(stretch_high))
    render_figure(entries, args.output)
    print(f"[INFO] Saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
