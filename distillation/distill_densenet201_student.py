"""
Offline knowledge distillation using the RAND DenseNet201 ensemble as teacher
and the same ResNet18 student architecture used for deployment.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

os.environ.setdefault("ACCELERATE_DISABLE_READONLY", "YES")
os.environ.setdefault("NPY_DISABLE_MAC_OSX_ACCELERATE", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.models import DenseNet201_Weights, ResNet18_Weights, densenet201, resnet18
from tqdm import tqdm

from distill_resnet18_student import (  # type: ignore
    DatasetReading,
    KDPairedDataset,
    _resolve_table_paths,
    _select_device,
    distillation_loss,
    evaluate,
    load_table_mat,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DENSENET_SCRIPTS = REPO_ROOT / "densenet201_ensembles" / "scripts"


class DenseNet201MS(nn.Module):
    """DenseNet201 backbone adapted to 3-band inputs."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        base = densenet201(weights=DenseNet201_Weights.DEFAULT)
        self.features = base.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base.classifier.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class RandDenseNet201Teacher(nn.Module):
    """Wrapper replicating the training-time ensemble (averaging logits)."""

    def __init__(self, num_members: int, num_classes: int, channels_per_member: int = 3) -> None:
        super().__init__()
        self.models = nn.ModuleList([DenseNet201MS(num_classes) for _ in range(num_members)])
        bands = torch.zeros(num_members, channels_per_member, dtype=torch.long)
        self.register_buffer("bands_matrix", bands)

    def forward(self, x: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        logits_sum = None
        for model, band_idx in zip(self.models, self.bands_matrix):
            subset = torch.index_select(x, dim=1, index=band_idx)
            logits = model(subset)
            logits_sum = logits if logits_sum is None else logits_sum + logits
        logits_avg = logits_sum / len(self.models)
        if return_logits:
            return logits_avg
        return torch.softmax(logits_avg, dim=1)


def build_teacher(checkpoint_path: Path, num_classes: int) -> RandDenseNet201Teacher:
    state = torch.load(checkpoint_path, map_location="cpu")
    if "bands_matrix" not in state:
        raise RuntimeError("Checkpoint missing bands_matrix buffer; unexpected format.")
    num_members = state["bands_matrix"].shape[0]
    teacher = RandDenseNet201Teacher(num_members, num_classes)
    teacher.load_state_dict(state, strict=True)
    return teacher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distill a ResNet18 student from the RAND DenseNet201 ensemble."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(REPO_ROOT / "data" / "lcz42"),
        help="Directory containing tables_MS.mat and HDF5 files.",
    )
    parser.add_argument(
        "--teacher-checkpoint",
        type=str,
        default=str(
            REPO_ROOT / "densenet201_ensembles" / "models" / "trained" / "Rand_densenet201.pth"
        ),
        help="Path to the RAND DenseNet201 ensemble checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path("distillation") / "checkpoints"),
        help="Directory where student checkpoints/history will be stored.",
    )
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--rgb-indices",
        type=int,
        nargs=3,
        default=(3, 2, 1),
        help="Zero-based Sentinel-2 indices for RGB composite (default B4/B3/B2).",
    )
    parser.add_argument(
        "--rescale-factor",
        type=float,
        default=1.0 / 255.0,
        help="Scalar applied to RGB tensor when z-score is not used.",
    )
    parser.add_argument("--apply-rgb-zscore", action="store_true")
    parser.add_argument("--student-pretrained", action="store_true")
    parser.add_argument("--use-sar-despeckle", action="store_true")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Computation device preference.",
    )
    parser.add_argument(
        "--teacher-device",
        type=str,
        default="cpu",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device for teacher forward pass (default: cpu).",
    )
    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_root = Path(args.data_root)
    table_path = data_root / "tables_MS.mat"
    train_table, test_table = load_table_mat(table_path, "train_MS", "test_MS")
    train_table = _resolve_table_paths(train_table, DENSENET_SCRIPTS)
    test_table = _resolve_table_paths(test_table, DENSENET_SCRIPTS)

    ds_train, ds_val, info = DatasetReading(
        dict(
            trainTable=train_table,
            testTable=test_table,
            useZscore=True,
            useSARdespeckle=args.use_sar_despeckle,
            useAugmentation=True,
            inputSize=(224, 224),
        )
    )

    rgb_mu = torch.tensor(info["mu"], dtype=torch.float32)[list(args.rgb_indices)]
    rgb_sigma = torch.tensor(info["sigma"], dtype=torch.float32)[list(args.rgb_indices)]
    train_dataset = KDPairedDataset(
        ds_train,
        args.rgb_indices,
        rgb_mu=rgb_mu if args.apply_rgb_zscore else None,
        rgb_sigma=rgb_sigma if args.apply_rgb_zscore else None,
        rescale_factor=args.rescale_factor,
    )
    val_dataset = KDPairedDataset(
        ds_val,
        args.rgb_indices,
        rgb_mu=rgb_mu if args.apply_rgb_zscore else None,
        rgb_sigma=rgb_sigma if args.apply_rgb_zscore else None,
        rescale_factor=args.rescale_factor,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    teacher = build_teacher(Path(args.teacher_checkpoint), info["numClasses"])
    student_device = _select_device(args.device)
    teacher_device = _select_device(args.teacher_device)
    print(f"[INFO] Student device: {student_device} | Teacher device: {teacher_device}")
    teacher.to(teacher_device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = resnet18(weights=ResNet18_Weights.DEFAULT if args.student_pretrained else None)
    student.fc = nn.Linear(student.fc.in_features, info["numClasses"])
    student.to(student_device)

    optimizer = optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    os.makedirs(args.output_dir, exist_ok=True)
    history = []

    for epoch in range(1, args.epochs + 1):
        student.train()
        running_loss = 0.0
        hard_loss = 0.0
        soft_loss = 0.0
        for rgb, full, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            rgb = rgb.to(student_device, non_blocking=True)
            labels = labels.to(student_device, non_blocking=True)

            with torch.no_grad():
                teacher_logits = teacher(
                    full.to(teacher_device, non_blocking=True), return_logits=True
                ).to(student_device)

            student_logits = student(rgb)
            loss, loss_parts = distillation_loss(
                student_logits,
                teacher_logits,
                labels,
                alpha=args.alpha,
                temperature=args.temperature,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            hard_loss += loss_parts["hard"] * batch_size
            soft_loss += loss_parts["soft"] * batch_size

        scheduler.step()

        train_loss = running_loss / len(train_dataset)
        train_hard = hard_loss / len(train_dataset)
        train_soft = soft_loss / len(train_dataset)
        val_acc = evaluate(student, val_loader, student_device)
        history.append(
            dict(
                epoch=epoch,
                train_loss=train_loss,
                hard=train_hard,
                soft=train_soft,
                val_acc=val_acc,
                lr=scheduler.get_last_lr()[0],
            )
        )

        print(
            f"[Epoch {epoch}] loss={train_loss:.4f} hard={train_hard:.4f} "
            f"soft={train_soft:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_path = Path(args.output_dir) / "student_resnet18_from_densenet201_best.pth"
            torch.save(student.state_dict(), best_path)
            print(f"[✓] Saved new best model → {best_path} (val_acc={val_acc:.4f})")

    history_path = Path(args.output_dir) / "kd_history_densenet201.json"
    with open(history_path, "w") as f:
        json.dump(dict(history=history, best_acc=best_acc), f, indent=2)
    print(f"[INFO] Training complete. Best val_acc={best_acc:.4f}")


if __name__ == "__main__":
    train(parse_args())
