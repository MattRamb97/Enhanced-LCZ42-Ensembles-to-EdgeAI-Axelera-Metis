"""
Evaluation of the distilled ResNet18 student on the LCZ42 validation split.
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

# --------------------------------------------------------------------------------------
# Configuration (hardcoded for reproducibility)
# --------------------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
RESNET_SCRIPTS = REPO_ROOT / "resnet18_ensembles" / "scripts"
DATA_ROOT = REPO_ROOT / "data" / "lcz42"
CHECKPOINT_PATH = Path(__file__).resolve().parent / "checkpoints" / "resnet18_to_resnet18" / "student_resnet18_last.pth"

RGB_INDICES = (2, 1, 0)  # Sentinel-2 B4/B3/B2 (R/G/B)
BATCH_SIZE = 512
NUM_WORKERS = 8
DEVICE = "auto"  # auto, cpu, cuda, mps


def build_student(num_classes: int, checkpoint_path: Path) -> nn.Module:
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    return model


def main(checkpoint_path: Path | None = None) -> None:
    """Evaluate student model on test set.

    Args:
        checkpoint_path: Path to checkpoint. If None, uses default CHECKPOINT_PATH.
    """
    if checkpoint_path is None:
        checkpoint_path = CHECKPOINT_PATH
    table_path = DATA_ROOT / "tables_MS.mat"
    train_table, test_table = load_table_mat(table_path, "train_MS", "test_MS")
    train_table = _resolve_table_paths(train_table, RESNET_SCRIPTS)
    test_table = _resolve_table_paths(test_table, RESNET_SCRIPTS)

    _, ds_val, info = DatasetReading(
        dict(
            trainTable=train_table,
            testTable=test_table,
            useZscore=True,
            useSARdespeckle=False,
            useAugmentation=False,
            inputSize=(224, 224),
        )
    )

    # DEBUG: Print μ/σ computed by DatasetReading
    mu_actual = info.get("mu")
    sigma_actual = info.get("sigma")
    if mu_actual is not None:
        print(f"[DEBUG] DatasetReading computed μ: {mu_actual.tolist()}")
        print(f"[DEBUG] DatasetReading computed σ: {sigma_actual.tolist()}")

    val_dataset = KDPairedDataset(ds_val, RGB_INDICES)
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    device = _select_device(DEVICE)
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Checkpoint: {checkpoint_path}")
    print(f"[INFO] RGB indices: {RGB_INDICES}")

    model = build_student(info["numClasses"], checkpoint_path).to(device)
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
    print(f"[RESULT] Test accuracy: {acc:.4%} ({correct}/{total})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate distilled ResNet18 student on LCZ42 test set"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT_PATH,
        help="Path to student checkpoint (default: checkpoints/resnet18_to_resnet18/student_resnet18_last.pth)",
    )
    args = parser.parse_args()
    main(checkpoint_path=args.checkpoint)
