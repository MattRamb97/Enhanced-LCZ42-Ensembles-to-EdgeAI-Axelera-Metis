"""
Offline knowledge distillation using the RAND ResNet18 ensemble as teacher and an RGB-only
ResNet18 student. The teacher consumes the full multispectral Sentinel-2 stack, while the student
receives a band-reduced RGB composite (b4-b3-b2). The script reuses the LCZ42 data tables and
pre-processing defined for the ensemble training code.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18
from tqdm import tqdm

# Add ensemble script directory to import DatasetReading utilities
REPO_ROOT = Path(__file__).resolve().parents[1]
RESNET_SCRIPTS = REPO_ROOT / "resnet18_ensembles" / "scripts"
import sys

sys.path.append(str(RESNET_SCRIPTS))
from dataset_reading import DatasetReading  # type: ignore


# --------------------------------------------------------------------------------------
# MATLAB table helpers (replicated from train_teacher_resnet18.py)
# --------------------------------------------------------------------------------------
def _matlab_to_scalar(value) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return int(value)
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return _matlab_to_scalar(value.reshape(-1)[0])
    return int(np.array(value).reshape(-1)[0])


def _matlab_to_string(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return _matlab_to_string(value.reshape(-1)[0])
        if value.dtype.kind in {"U", "S"}:
            return "".join(value.astype(str).reshape(-1))
        if value.dtype == object:
            return "".join(_matlab_to_string(v) for v in value.reshape(-1))
    return str(value)


def _matlab_table_to_df(table_array: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame(table_array, columns=["Path", "Label", "Index", "Modality"])
    df["Path"] = df["Path"].map(_matlab_to_string)
    df["Label"] = df["Label"].map(_matlab_to_scalar)
    df["Index"] = df["Index"].map(_matlab_to_scalar).astype(int) - 1
    df["Modality"] = df["Modality"].map(lambda x: _matlab_to_string(x).upper())
    return df


def load_table_mat(path: Path, train_key: str, test_key: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data = loadmat(path, simplify_cells=False)
    train = _matlab_table_to_df(data[train_key])
    test = _matlab_table_to_df(data[test_key])
    return train, test


def _resolve_table_paths(df: pd.DataFrame, base_dir: Path) -> pd.DataFrame:
    """Ensure MATLAB-relative HDF5 paths become absolute paths rooted at the repo."""
    base_dir = base_dir.resolve()
    df = df.copy()

    def _resolve(path_str: str) -> str:
        path_obj = Path(path_str)
        if path_obj.is_absolute():
            return str(path_obj)
        return str((base_dir / path_obj).resolve())

    df["Path"] = df["Path"].map(_resolve)
    return df


# --------------------------------------------------------------------------------------
# Teacher architecture (identical to training code but with logits option)
# --------------------------------------------------------------------------------------
class ResNet18MS(nn.Module):
    """ResNet18 backbone adapted to 3-band inputs."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        base = resnet18(weights=None)
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.fc = nn.Linear(base.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class RandResNet18Teacher(nn.Module):
    """Ensemble wrapper that returns either logits or averaged probabilities."""

    def __init__(self, num_members: int, num_classes: int, bands_per_member: int = 3) -> None:
        super().__init__()
        members = [ResNet18MS(num_classes) for _ in range(num_members)]
        self.models = nn.ModuleList(members)
        bands = torch.zeros(num_members, bands_per_member, dtype=torch.long)
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


def build_teacher(checkpoint_path: Path) -> RandResNet18Teacher:
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    member_ids = sorted(
        {int(key.split(".")[1]) for key in state_dict.keys() if key.startswith("models.")}
    )
    num_members = len(member_ids)
    fc_key = next(k for k in state_dict.keys() if k.endswith("fc.weight"))
    num_classes = state_dict[fc_key].shape[0]
    teacher = RandResNet18Teacher(num_members=num_members, num_classes=num_classes)
    teacher.load_state_dict(state_dict, strict=True)
    return teacher


# --------------------------------------------------------------------------------------
# Dataset wrapper
# --------------------------------------------------------------------------------------
class KDPairedDataset(Dataset):
    """
    Returns both the full multispectral tensor (for teacher) and the RGB subset (for student).
    """

    def __init__(
        self,
        base_dataset: Dataset,
        rgb_indices: Iterable[int],
        rgb_mu: torch.Tensor | None,
        rgb_sigma: torch.Tensor | None,
        rescale_factor: float,
    ) -> None:
        self.base = base_dataset
        self.rgb_idx = torch.tensor(list(rgb_indices), dtype=torch.long)
        self.rescale_factor = rescale_factor
        self.rgb_mu = rgb_mu
        self.rgb_sigma = rgb_sigma

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        full_tensor, label = self.base[idx]  # (C_full, H, W)
        rgb = torch.index_select(full_tensor, dim=0, index=self.rgb_idx)
        if self.rgb_mu is not None and self.rgb_sigma is not None:
            rgb = (rgb - self.rgb_mu[:, None, None]) / (self.rgb_sigma[:, None, None] + 1e-6)
        else:
            rgb = rgb * self.rescale_factor
        return rgb, full_tensor, label


# --------------------------------------------------------------------------------------
# Loss / metrics
# --------------------------------------------------------------------------------------
def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
    temperature: float,
) -> Tuple[torch.Tensor, dict]:
    hard_loss = F.cross_entropy(student_logits, labels)
    log_student = F.log_softmax(student_logits / temperature, dim=1)
    teacher_prob = F.softmax(teacher_logits / temperature, dim=1)
    soft_loss = F.kl_div(log_student, teacher_prob, reduction="batchmean") * (temperature**2)
    loss = alpha * hard_loss + (1 - alpha) * soft_loss
    return loss, {"hard": hard_loss.item(), "soft": soft_loss.item()}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for rgb, _, labels in loader:
        rgb = rgb.to(device)
        labels = labels.to(device)
        logits = model(rgb)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / max(total, 1)


# --------------------------------------------------------------------------------------
# Device helper
# --------------------------------------------------------------------------------------
def _select_device(request: str) -> torch.device:
    request = request.lower()
    supports_mps = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    if request == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if supports_mps:
            return torch.device("mps")
        return torch.device("cpu")
    if request == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA requested but not available.")
    if request == "mps":
        if supports_mps:
            return torch.device("mps")
        raise RuntimeError("MPS requested but not available.")
    if request == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unknown device option '{request}'.")


# --------------------------------------------------------------------------------------
# Training loop
# --------------------------------------------------------------------------------------
@dataclass
class KDConfig:
    temperature: float = 2.0
    alpha: float = 0.7
    lr: float = 2e-4
    weight_decay: float = 1e-4
    epochs: int = 30
    batch_size: int = 512
    num_workers: int = 10
    rgb_indices: Tuple[int, int, int] = (3, 2, 1)  # Sentinel-2 B4/B3/B2
    rescale_factor: float = 1.0 / 255.0


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_root = Path(args.data_root)
    table_path = data_root / "tables_MS.mat"
    train_table, test_table = load_table_mat(table_path, "train_MS", "test_MS")
    train_table = _resolve_table_paths(train_table, RESNET_SCRIPTS)
    test_table = _resolve_table_paths(test_table, RESNET_SCRIPTS)
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    teacher = build_teacher(Path(args.teacher_checkpoint))
    device = _select_device(args.device)
    print(f"[INFO] Using device: {device}")
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = resnet18(weights=ResNet18_Weights.DEFAULT if args.student_pretrained else None)
    student.fc = nn.Linear(student.fc.in_features, info["numClasses"])
    student.to(device)

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
            rgb = rgb.to(device, non_blocking=True)
            full = full.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.no_grad():
                teacher_logits = teacher(full, return_logits=True)

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
        val_acc = evaluate(student, val_loader, device)
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
            best_path = Path(args.output_dir) / "student_resnet18_best.pth"
            torch.save(student.state_dict(), best_path)
            print(f"[✓] Saved new best model to {best_path} (val_acc={val_acc:.4f})")

    history_path = Path(args.output_dir) / "kd_history.json"
    with open(history_path, "w") as f:
        json.dump(dict(history=history, best_acc=best_acc), f, indent=2)
    print(f"[INFO] Training complete. Best val_acc={best_acc:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline KD for ResNet18 student (RGB-only).")
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(REPO_ROOT / "data" / "lcz42"),
        help="Directory containing tables_MS.mat",
    )
    parser.add_argument(
        "--teacher-checkpoint",
        type=str,
        default=str(
            REPO_ROOT / "resnet18_ensembles" / "models" / "trained" / "Rand_resnet18.pth"
        ),
        help="Path to the RAND ensemble checkpoint (.pth).",
    )
    parser.add_argument("--output-dir", type=str, default=str(Path("distillation") / "checkpoints"))
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--rgb-indices",
        type=int,
        nargs=3,
        default=(3, 2, 1),
        help="Zero-based band indices used to build the RGB composite (default: Sentinel-2 B4/B3/B2).",
    )
    parser.add_argument(
        "--rescale-factor",
        type=float,
        default=1.0 / 255.0,
        help="Scalar applied to RGB tensor when z-score is not used.",
    )
    parser.add_argument("--apply-rgb-zscore", action="store_true", help="Use μ/σ from dataset info.")
    parser.add_argument(
        "--student-pretrained",
        action="store_true",
        help="Initialise the student from ImageNet pretrained weights.",
    )
    parser.add_argument(
        "--use-sar-despeckle",
        action="store_true",
        help="Enable SAR despeckling in DatasetReading (mirrors ensemble training).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Computation device preference (default: auto).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
