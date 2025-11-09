#!/usr/bin/env python3
"""
Generate representative persistence images per LCZ class for LaTeX figures.

The script samples Sentinel-2 patches for the requested LCZ classes, computes
Cubical Persistence (H0/H1) on a selected spectral band, converts persistence
diagrams into persistence images via a Gaussian kernel, and saves a 2×N panel
figure.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
from gtda.homology import CubicalPersistence


# --------------------------------------------------------------------------- #
# LCZ metadata
# --------------------------------------------------------------------------- #

LCZ_CLASS_INFO: Dict[int, Tuple[str, str]] = {
    1: ("LCZ 1", "Compact High-Rise"),
    2: ("LCZ 2", "Compact Mid-Rise"),
    3: ("LCZ 3", "Compact Low-Rise"),
    4: ("LCZ 4", "Open High-Rise"),
    5: ("LCZ 5", "Open Mid-Rise"),
    6: ("LCZ 6", "Open Low-Rise"),
    7: ("LCZ 7", "Lightweight Low-Rise"),
    8: ("LCZ 8", "Large Low-Rise"),
    9: ("LCZ 9", "Sparsely Built"),
    10: ("LCZ 10", "Heavy Industry"),
    11: ("LCZ A", "Dense Trees"),
    12: ("LCZ B", "Scattered Trees"),
    13: ("LCZ C", "Bush / Scrub"),
    14: ("LCZ D", "Low Plants"),
    15: ("LCZ E", "Bare Rock or Paved"),
    16: ("LCZ F", "Bare Soil or Sand"),
    17: ("LCZ G", "Water"),
}

LETTER_TO_ID = {
    "A": 11,
    "B": 12,
    "C": 13,
    "D": 14,
    "E": 15,
    "F": 16,
    "G": 17,
}


# --------------------------------------------------------------------------- #
# Argument parsing and helpers
# --------------------------------------------------------------------------- #

def parse_class_ids(raw_values: Sequence[str]) -> List[int]:
    """Allow numeric IDs (1-17), LCZ_ prefixed strings, or letters A-G."""
    class_ids: List[int] = []
    for raw in raw_values:
        val = raw.strip().upper()
        if val.startswith("LCZ_"):
            val = val.replace("LCZ_", "")
        if val in LETTER_TO_ID:
            class_ids.append(LETTER_TO_ID[val])
        else:
            try:
                class_ids.append(int(val))
            except ValueError as exc:
                raise ValueError(f"Unrecognised LCZ class identifier: {raw}") from exc
    for cid in class_ids:
        if cid not in LCZ_CLASS_INFO:
            raise ValueError(f"LCZ class {cid} is outside the supported range [1, 17].")
    return class_ids


def class_label(class_id: int) -> str:
    short, full = LCZ_CLASS_INFO[class_id]
    return f"{short} — {full}"


# --------------------------------------------------------------------------- #
# Persistence image utilities
# --------------------------------------------------------------------------- #

def compute_birth_persistence_ranges(
    diagrams: Iterable[np.ndarray],
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Return global birth and persistence ranges covering all diagrams."""
    births: List[float] = []
    persistences: List[float] = []
    for diag in diagrams:
        if diag.size == 0:
            continue
        birth = diag[:, 0]
        death = diag[:, 1]
        valid = np.isfinite(birth) & np.isfinite(death) & (death > birth)
        if not np.any(valid):
            continue
        births.extend(birth[valid])
        persistences.extend((death - birth)[valid])

    if not births:
        return (0.0, 1.0), (0.0, 1.0)

    birth_min = float(np.min(births))
    birth_max = float(np.max(births))
    pers_max = float(np.max(persistences))

    # Add a few percent margin so Gaussians near the border remain visible.
    margin_b = max(1e-6, 0.03 * (birth_max - birth_min))
    margin_p = max(1e-6, 0.03 * pers_max)

    return (birth_min - margin_b, birth_max + margin_b), (0.0, pers_max + margin_p)


def gaussian_persistence_image(
    diagrams: Sequence[np.ndarray],
    birth_range: Tuple[float, float],
    persistence_range: Tuple[float, float],
    resolution: int,
    sigma: float,
    weight_persistence: float,
) -> np.ndarray:
    """Compute the average persistence image for provided diagrams."""
    if resolution <= 0:
        raise ValueError("resolution must be positive.")
    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    x = np.linspace(birth_range[0], birth_range[1], resolution)
    y = np.linspace(persistence_range[0], persistence_range[1], resolution)
    xx, yy = np.meshgrid(x, y)
    accumulator = np.zeros_like(xx, dtype=np.float64)
    eps = 1e-12

    for diag in diagrams:
        if diag.size == 0:
            continue
        birth_all = diag[:, 0]
        death_all = diag[:, 1]
        valid = np.isfinite(birth_all) & np.isfinite(death_all) & (death_all > birth_all)
        if not np.any(valid):
            continue
        birth = birth_all[valid]
        death = death_all[valid]
        persistence = death - birth

        for b, p in zip(birth, persistence):
            weight = math.pow(p, weight_persistence) if weight_persistence != 0 else 1.0
            gaussian = np.exp(-((xx - b) ** 2 + (yy - p) ** 2) / (2.0 * sigma**2))
            accumulator += weight * gaussian

    if diagrams:
        accumulator /= max(len(diagrams), 1)

    accumulator = accumulator / (np.max(accumulator) + eps)
    return accumulator.astype(np.float32)


# --------------------------------------------------------------------------- #
# Data loading and diagram generation
# --------------------------------------------------------------------------- #

def sample_indices_per_class(
    labels: np.ndarray,
    class_ids: Sequence[int],
    samples_per_class: int,
    seed: int,
) -> Dict[int, np.ndarray]:
    """Randomly sample indices for each LCZ class."""
    rng = np.random.default_rng(seed)
    sampled: Dict[int, np.ndarray] = {}
    y = labels.argmax(axis=1) + 1  # convert one-hot to 1-17 indices
    for cid in class_ids:
        indices = np.flatnonzero(y == cid)
        if len(indices) == 0:
            raise RuntimeError(f"No samples found for LCZ class {cid}.")
        count = min(samples_per_class, len(indices))
        sampled[cid] = rng.choice(indices, size=count, replace=False)
    return sampled


def compute_diagrams_for_band(
    dataset: h5py.Dataset,
    indices: np.ndarray,
    band_index: int,
    cp: CubicalPersistence,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Compute persistence diagrams (H0/H1) for a selected spectral band."""
    sorted_indices = np.sort(indices)
    patches = dataset[sorted_indices, :, :, band_index].astype(np.float64)
    # Bring values to 0-255 to match extraction scripts.
    patches = (patches / (2.8 / 255.0)).clip(0.0, 255.0)

    diagrams = cp.fit_transform(patches)
    h0_list: List[np.ndarray] = []
    h1_list: List[np.ndarray] = []
    for diag in diagrams:
        if diag.size == 0:
            h0_list.append(np.empty((0, 2), dtype=np.float64))
            h1_list.append(np.empty((0, 2), dtype=np.float64))
            continue
        dim = diag[:, 2].astype(np.int64)
        h0_points = diag[dim == 0, :2]
        h1_points = diag[dim == 1, :2]
        h0_list.append(h0_points)
        h1_list.append(h1_points)
    return h0_list, h1_list


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def plot_panel(
    images_h0: Dict[int, np.ndarray],
    images_h1: Dict[int, np.ndarray],
    birth_ranges_h0: Dict[int, Tuple[float, float]],
    persistence_ranges_h0: Dict[int, Tuple[float, float]],
    birth_ranges_h1: Dict[int, Tuple[float, float]],
    persistence_ranges_h1: Dict[int, Tuple[float, float]],
    class_ids: Sequence[int],
    cmap: str,
    output_path: Path,
):
    cols = len(class_ids)
    fig, axes = plt.subplots(2, cols, figsize=(3.0 * cols, 6))

    if cols == 1:
        axes = np.array([[axes[0]], [axes[1]]])  # ensure 2×N indexing

    for j, cid in enumerate(class_ids):
        # H0 row
        ax0 = axes[0, j]
        img0 = images_h0[cid]
        br0 = birth_ranges_h0[cid]
        pr0 = persistence_ranges_h0[cid]
        im0 = ax0.imshow(
            img0,
            cmap=cmap,
            origin="lower",
            extent=[br0[0], br0[1], pr0[0], pr0[1]],
            aspect="auto",
        )
        ax0.set_title(class_label(cid), fontsize=11)
        if j == 0:
            ax0.set_ylabel("$H_0$ — Connected components")
        ax0.set_xticks([])
        ax0.set_yticks([])

        # H1 row
        ax1 = axes[1, j]
        img1 = images_h1[cid]
        br1 = birth_ranges_h1[cid]
        pr1 = persistence_ranges_h1[cid]
        im1 = ax1.imshow(
            img1,
            cmap=cmap,
            origin="lower",
            extent=[br1[0], br1[1], pr1[0], pr1[1]],
            aspect="auto",
        )
        if j == 0:
            ax1.set_ylabel("$H_1$ — Loops / voids")
        ax1.set_xlabel("Birth")
        ax1.set_yticks([])

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main routine
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute representative persistence images for LCZ classes."
    )
    parser.add_argument(
        "--h5",
        default="data/lcz42/training.h5",
        help="Path to the LCZ42 HDF5 file containing /sen2 and /label datasets.",
    )
    parser.add_argument(
        "--band",
        type=int,
        default=2,
        help="Zero-based Sentinel-2 band index to analyse (default: 2 ≈ Red band B4).",
    )
    parser.add_argument(
        "--class-ids",
        nargs="+",
        default=["1", "5", "8", "10", "A", "D", "G"],
        help="List of LCZ classes to visualise (accepts numbers, letters A-G, or LCZ_*).",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=64,
        help="Maximum number of patches sampled per class to build the persistence image.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=64,
        help="Resolution (pixels per axis) of the persistence images.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=3.0,
        help="Gaussian kernel width used to rasterise persistence diagrams.",
    )
    parser.add_argument(
        "--weight-persistence",
        type=float,
        default=1.0,
        help="Exponent applied to persistence when weighting Gaussian kernels.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed controlling class-wise sampling.",
    )
    parser.add_argument(
        "--cmap",
        default="magma",
        help="Matplotlib colormap for persistence images.",
    )
    parser.add_argument(
        "--output",
        default="figures/chapter7/tda_persistence_images.pdf",
        help="Destination path for the generated figure.",
    )

    args = parser.parse_args()

    class_ids = parse_class_ids(args.class_ids)
    output_path = Path(args.output)

    cp = CubicalPersistence(homology_dimensions=(0, 1))

    h5_path = Path(args.h5)
    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        if "sen2" not in f or "label" not in f:
            raise KeyError("Expected datasets '/sen2' and '/label' in the HDF5 file.")

        num_bands = f["sen2"].shape[-1]
        if not 0 <= args.band < num_bands:
            raise ValueError(
                f"Band index {args.band} is out of range for Sentinel-2 dataset "
                f"with {num_bands} bands."
            )

        labels = f["label"][:]
        sampled_indices = sample_indices_per_class(
            labels, class_ids, args.samples_per_class, args.seed
        )

        images_h0: Dict[int, np.ndarray] = {}
        images_h1: Dict[int, np.ndarray] = {}
        birth_ranges_h0: Dict[int, Tuple[float, float]] = {}
        persistence_ranges_h0: Dict[int, Tuple[float, float]] = {}
        birth_ranges_h1: Dict[int, Tuple[float, float]] = {}
        persistence_ranges_h1: Dict[int, Tuple[float, float]] = {}

        for cid in class_ids:
            indices = sampled_indices[cid]
            h0_diag, h1_diag = compute_diagrams_for_band(f["sen2"], indices, args.band, cp)

            birth_ranges_h0[cid], persistence_ranges_h0[cid] = compute_birth_persistence_ranges(
                h0_diag
            )
            birth_ranges_h1[cid], persistence_ranges_h1[cid] = compute_birth_persistence_ranges(
                h1_diag
            )

            images_h0[cid] = gaussian_persistence_image(
                h0_diag,
                birth_ranges_h0[cid],
                persistence_ranges_h0[cid],
                args.resolution,
                args.sigma,
                args.weight_persistence,
            )
            images_h1[cid] = gaussian_persistence_image(
                h1_diag,
                birth_ranges_h1[cid],
                persistence_ranges_h1[cid],
                args.resolution,
                args.sigma,
                args.weight_persistence,
            )

    plot_panel(
        images_h0,
        images_h1,
        birth_ranges_h0,
        persistence_ranges_h0,
        birth_ranges_h1,
        persistence_ranges_h1,
        class_ids,
        args.cmap,
        output_path,
    )

    print(f"[✓] Persistence figure saved to: {output_path}")


if __name__ == "__main__":
    main()
