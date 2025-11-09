"""
Quick evaluation of a distilled ResNet18 student on the LCZ42 validation split.
Reuses the same dataset utilities defined for training to ensure identical preprocessing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18

from distill_resnet18_student import (  # type: ignore
    DatasetReading,
    KDPairedDataset,
    _resolve_table_paths,
    load_table_mat,
    _select_device,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RESNET_SCRIPTS = REPO_ROOT / "resnet18_ensembles" / "scripts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate distilled ResNet18 student on LCZ42.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(Path("checkpoints") / "student_resnet18_best.pth"),
        help="Path to the student checkpoint (.pth).",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(REPO_ROOT / "data" / "lcz42"),
        help="Directory containing tables_MS.mat and HDF5 files.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=512, help="Evaluation batch size (default: 512)."
    )
    parser.add_argument(
        "--num-workers", type=int, default=8, help="DataLoader worker count (default: 8)."
    )
    parser.add_argument(
        "--rgb-indices",
        type=int,
        nargs=3,
        default=(3, 2, 1),
        help="Zero-based Sentinel-2 indices to form the RGB composite.",
    )
    parser.add_argument(
        "--apply-rgb-zscore",
        action="store_true",
        help="Use the μ/σ stats from DatasetReading on the selected RGB channels.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device selection (default: auto).",
    )
    parser.add_argument(
        "--loader-workers",
        type=int,
        default=12,
        help="Worker count for DatasetReading internals (default: 12).",
    )
    return parser.parse_args()


def build_student(num_classes: int, checkpoint_path: Path) -> nn.Module:
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    return model


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    table_path = data_root / "tables_MS.mat"
    train_table, test_table = load_table_mat(table_path, "train_MS", "test_MS")
    train_table = _resolve_table_paths(train_table, RESNET_SCRIPTS)
    test_table = _resolve_table_paths(test_table, RESNET_SCRIPTS)

    torch.set_num_threads(max(1, args.loader_workers))
    _, ds_val, info = DatasetReading(
        dict(
            trainTable=train_table,
            testTable=test_table,
            useZscore=True,
            useSARdespeckle=False,
            useAugmentation=False,
            inputSize=(224, 224),
            numWorkers=args.loader_workers,
        )
    )

    rgb_mu = torch.tensor(info["mu"], dtype=torch.float32)[list(args.rgb_indices)]
    rgb_sigma = torch.tensor(info["sigma"], dtype=torch.float32)[list(args.rgb_indices)]

    val_dataset = KDPairedDataset(
        ds_val,
        args.rgb_indices,
        rgb_mu=rgb_mu if args.apply_rgb_zscore else None,
        rgb_sigma=rgb_sigma if args.apply_rgb_zscore else None,
        rescale_factor=1.0 / 255.0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    device = _select_device(args.device)
    print(f"[INFO] Using device: {device}")
    model = build_student(info["numClasses"], Path(args.checkpoint)).to(device)
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for rgb, _, labels in val_loader:
            rgb = rgb.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(rgb)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / max(total, 1)
    print(f"[RESULT] Validation accuracy: {acc:.4%} ({correct}/{total})")


if __name__ == "__main__":
    main()
