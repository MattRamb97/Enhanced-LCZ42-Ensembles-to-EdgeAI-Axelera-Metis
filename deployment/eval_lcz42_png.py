"""
Evaluate the distilled ResNet18 student on the Voyager PNG dataset (LCZ42)
with preprocessing aligned to DatasetReading statistics.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18
from tqdm import tqdm

# --------------------------------------------------------------------------------------
# Configuration (hardcoded for reproducibility)
# --------------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
VAL_ROOT = BASE_DIR / "data" / "LCZ42" / "val"
CHECKPOINT_PATH = BASE_DIR.parent / "distillation" / "checkpoints" / "student_resnet18_last.pth"

# μ/σ from DatasetReading for RGB channels (after paper scaling on 224x224 resized data)
# DatasetReading outputs [μ₀, μ₁, μ₂, ...] for bands [0, 1, 2, ...]
# RGB tensor from index_select([2, 1, 0]) has channels: [original_ch2, original_ch1, original_ch0]
# So we need μ/σ reordered as: [μ₂, μ₁, μ₀]
# IMPORTANT: If you recompute with compute_lcz42_mu_sigma.py, update these values!
MU = torch.tensor([9.206003189086914, 9.952054977416992, 11.270723342895508], dtype=torch.float32)
SIGMA = torch.tensor([6.044061660766602, 4.3512349128723145, 3.605332136154175], dtype=torch.float32)

BATCH_SIZE = 512
NUM_WORKERS = 8
DEVICE = "auto"  # auto, cpu, cuda, mps
NUM_CLASSES = 17


def preprocess_png(path: str) -> torch.Tensor:
    """Load 16-bit PNG and apply paper scaling + resize + z-score normalization.

    Pipeline (matches DatasetReading):
    1. Load 32x32 uint16 PNG (raw reflectance values [0, 65535])
    2. Scale back to [0, 1] then apply paper_scaling
    3. Resize to 224x224 (MUST be before z-score, as μ/σ computed on 224x224)
    4. Apply z-score normalization (computed on 224x224 paper-scaled values)
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # Reads as uint16 if available
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    img = img[:, :, ::-1]  # BGR → RGB
    img = img.astype(np.float32)

    # Scale back from uint16 [0, 65535] to float32 [0, 1]
    # (inverse of 65535.0 scaling in prepare_lcz42_voyager_dataset.py)
    img = img / 65535.0

    # Apply paper scaling: x / (2.8 / 255.0) which is x * 255.0 / 2.8
    img = img / (2.8 / 255.0)
    img = np.clip(img, 0.0, 255.0)

    # Resize to 224x224 BEFORE z-score normalization
    # μ/σ are computed on 224x224 resized data, so resize must happen first
    img_resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_NEAREST)

    # Apply z-score normalization with pre-computed μ/σ (computed on 224x224 paper-scaled values)
    img_resized = (img_resized - MU.numpy()[None, None, :]) / (SIGMA.numpy()[None, None, :] + 1e-6)

    img = torch.from_numpy(img_resized.transpose(2, 0, 1))  # (H, W, C) → (C, H, W)
    return img


# ----------------------------------------------------------------------
# Custom dataset reading Voyager-style directory
# ----------------------------------------------------------------------
class PNGDataset(Dataset):
    def __init__(self, root_dir: str) -> None:
        self.samples = []
        root = Path(root_dir)
        for cls_idx, cls_name in enumerate(sorted(d.name for d in root.iterdir() if d.is_dir())):
            class_dir = root / cls_name
            for img_path in class_dir.glob("*.png"):
                self.samples.append((str(img_path), cls_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = preprocess_png(path)
        return img, label


# ----------------------------------------------------------------------
# Model and evaluation
# ----------------------------------------------------------------------
def build_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def select_device(request: str) -> torch.device:
    if request == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(request)


def main() -> None:
    device = select_device(DEVICE)
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Validation root: {VAL_ROOT}")
    print(f"[INFO] Checkpoint: {CHECKPOINT_PATH}")
    print(f"[INFO] μ/σ: {MU.tolist()} / {SIGMA.tolist()}")

    model = build_model(CHECKPOINT_PATH, device)
    dataset = PNGDataset(str(VAL_ROOT))
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="[Evaluating]"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / max(total, 1)
    print(f"\n[RESULT] Accuracy: {acc:.4%} ({correct}/{total})")


if __name__ == "__main__":
    main()
