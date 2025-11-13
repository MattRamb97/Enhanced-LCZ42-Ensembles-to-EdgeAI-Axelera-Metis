"""
Offline knowledge distillation: RAND DenseNet201 ensemble (teacher) -> RGB ResNet18 student.

"""

from __future__ import annotations

import json
import os
import random
import sys
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
from torchvision.models import DenseNet201_Weights, ResNet18_Weights, densenet201, resnet18
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
DENSENET_SCRIPTS = REPO_ROOT / "densenet201_ensembles" / "scripts"
sys.path.append(str(DENSENET_SCRIPTS))
from dataset_reading import DatasetReading  # type: ignore


# --------------------------------------------------------------------------------------
# Constants / paths
# --------------------------------------------------------------------------------------
DATA_ROOT = REPO_ROOT / "data" / "lcz42"
TEACHER_CHECKPOINT = (
    REPO_ROOT / "densenet201_ensembles" / "models" / "trained" / "Rand_densenet201.pth"
)
OUTPUT_DIR = Path(__file__).resolve().parent / "checkpoints" / "densenet201_to_resnet18"
RGB_INDICES = (2, 1, 0)  # Sentinel-2 B4/B3/B2 in zero-based Python indexing


# --------------------------------------------------------------------------------------
# MATLAB table helpers
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
# Teacher ensemble
# --------------------------------------------------------------------------------------
class DenseNet201MS(nn.Module):
    """DenseNet201 backbone adapted to 3-band inputs."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        base = densenet201(weights=None)
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
    """RAND ensemble wrapper that averages member logits."""

    def __init__(self, num_members: int, num_classes: int, bands_per_member: int = 3) -> None:
        super().__init__()
        members = [DenseNet201MS(num_classes) for _ in range(num_members)]
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


def build_teacher(checkpoint_path: Path) -> RandDenseNet201Teacher:
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    member_ids = sorted({int(key.split(".")[1]) for key in state_dict if key.startswith("models.")})
    num_members = len(member_ids)
    fc_key = next(k for k in state_dict.keys() if k.endswith("fc.weight"))
    num_classes = state_dict[fc_key].shape[0]
    teacher = RandDenseNet201Teacher(num_members=num_members, num_classes=num_classes)
    teacher.load_state_dict(state_dict, strict=True)
    return teacher


# --------------------------------------------------------------------------------------
# Dataset wrapper
# --------------------------------------------------------------------------------------
class KDPairedDataset(Dataset):
    """
    Returns both the full multispectral tensor (teacher) and the RGB subset (student).
    """

    def __init__(
        self,
        base_dataset: Dataset,
        rgb_indices: Iterable[int],
    ) -> None:
        self.base = base_dataset
        self.rgb_idx = torch.tensor(list(rgb_indices), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        full_tensor, label = self.base[idx]  # (C_full, H, W)
        rgb = torch.index_select(full_tensor, dim=0, index=self.rgb_idx)
        return rgb, full_tensor, label


# --------------------------------------------------------------------------------------
# Loss / helpers
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
# Configuration
# --------------------------------------------------------------------------------------
@dataclass
class KDConfig:
    data_root: Path = DATA_ROOT
    teacher_checkpoint: Path = TEACHER_CHECKPOINT
    output_dir: Path = OUTPUT_DIR
    alpha: float = 0.7
    lr: float = 2e-4
    weight_decay: float = 1e-4
    epochs: int = 30
    batch_size: int = 512
    num_workers: int = 8
    seed: int = 42
    rgb_indices: Tuple[int, int, int] = RGB_INDICES
    student_pretrained: bool = False
    use_sar_despeckle: bool = False
    device: str = "auto"


# --------------------------------------------------------------------------------------
# Training loop
# --------------------------------------------------------------------------------------
def distill_student(cfg: KDConfig | None = None) -> None:
    cfg = cfg or KDConfig()

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    data_root = Path(cfg.data_root).resolve()
    teacher_ckpt = Path(cfg.teacher_checkpoint).resolve()
    output_dir = Path(cfg.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    table_path = data_root / "tables_MS.mat"
    train_table, test_table = load_table_mat(table_path, "train_MS", "test_MS")
    train_table = _resolve_table_paths(train_table, DENSENET_SCRIPTS)
    test_table = _resolve_table_paths(test_table, DENSENET_SCRIPTS)

    ds_train, _, info = DatasetReading(
        dict(
            trainTable=train_table,
            testTable=test_table,
            useZscore=True,
            useSARdespeckle=cfg.use_sar_despeckle,
            useAugmentation=True,
            inputSize=(224, 224),
        )
    )

    train_dataset = KDPairedDataset(ds_train, cfg.rgb_indices)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    device = _select_device(cfg.device)
    teacher = build_teacher(teacher_ckpt).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = resnet18(weights=ResNet18_Weights.DEFAULT if cfg.student_pretrained else None)
    student.fc = nn.Linear(student.fc.in_features, info["numClasses"])
    student.to(device)

    optimizer = optim.AdamW(student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    history = []
    best_loss = float("inf")
    best_path = output_dir / "student_resnet18_from_densenet201_best.pth"

    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Training samples: {len(train_dataset)}")

    for epoch in range(1, cfg.epochs + 1):
        student.train()
        running_loss = 0.0
        hard_loss = 0.0
        soft_loss = 0.0

        for rgb, full, labels in tqdm(train_loader, desc=f"[Epoch {epoch}/{cfg.epochs}]"):
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
                alpha=cfg.alpha,
                temperature=3.0,
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
        history.append(
            dict(
                epoch=epoch,
                train_loss=train_loss,
                hard=train_hard,
                soft=train_soft,
                lr=scheduler.get_last_lr()[0],
            )
        )

        print(
            f"[Epoch {epoch}] loss={train_loss:.4f} hard={train_hard:.4f} soft={train_soft:.4f}"
        )

        epoch_ckpt = output_dir / f"student_resnet18_from_densenet201_epoch{epoch:02d}.pth"
        torch.save(student.state_dict(), epoch_ckpt)
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(student.state_dict(), best_path)

    history_path = output_dir / "kd_history_densenet201_resnet18.json"
    with open(history_path, "w") as f:
        json.dump(
            dict(
                history=history,
                best_loss=best_loss,
                mu=info["mu"].tolist(),
                sigma=info["sigma"].tolist(),
                numClasses=info["numClasses"],
            ),
            f,
            indent=2,
        )

    latest_path = output_dir / "student_resnet18_from_densenet201_last.pth"
    torch.save(student.state_dict(), latest_path)
    print(f"[INFO] Training complete. Checkpoints saved to {output_dir}")
    print(f"[INFO] Best checkpoint → {best_path}")
    print(f"[INFO] Latest checkpoint → {latest_path}")


if __name__ == "__main__":
    os.environ.setdefault("ACCELERATE_DISABLE_READONLY", "YES")
    os.environ.setdefault("NPY_DISABLE_MAC_OSX_ACCELERATE", "1")
    distill_student()