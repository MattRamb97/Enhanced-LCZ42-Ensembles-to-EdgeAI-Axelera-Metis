"""
Evaluate the ResNet18 student distilled from the DenseNet201 teacher.
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
    _select_device,
    load_table_mat,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DENSENET_SCRIPTS = REPO_ROOT / "densenet201_ensembles" / "scripts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ResNet18 student distilled from DenseNet201 teacher."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(
            Path("checkpoints") / "student_resnet18_from_densenet201_best.pth"
        ),
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(REPO_ROOT / "data" / "lcz42"),
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--rgb-indices", type=int, nargs=3, default=(3, 2, 1))
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
    )
    parser.add_argument("--loader-workers", type=int, default=12)
    return parser.parse_args()


def build_student(num_classes: int, checkpoint_path: Path) -> nn.Module:
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    return model


def main() -> None:
    args = parse_args()
    table_path = Path(args.data_root) / "tables_MS.mat"
    train_table, test_table = load_table_mat(table_path, "train_MS", "test_MS")
    train_table = _resolve_table_paths(train_table, DENSENET_SCRIPTS)
    test_table = _resolve_table_paths(test_table, DENSENET_SCRIPTS)

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

    val_dataset = KDPairedDataset(
        ds_val,
        args.rgb_indices,
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
            preds = model(rgb).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / max(total, 1)
    print(f"[RESULT] Test accuracy: {acc:.4%} ({correct}/{total})")


if __name__ == "__main__":
    main()