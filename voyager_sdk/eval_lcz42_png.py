"""
Evaluate the distilled ResNet18 student on the Voyager PNG dataset (LCZ42).

IMPORTANT: voyager_sdk/data/LCZ42 contains 32x32 PNG images that are ALREADY paper-scaled [0, 255].
Pipeline:
1. Load 32x32 uint8 PNG (paper-scaled values [0, 255])
2. Resize to 224x224 (required by ResNet18 input)
3. Apply z-score normalization (μ/σ computed from training dataset)
4. Evaluate model
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18
from tqdm import tqdm

# --------------------------------------------------------------------------------------
# Configuration (hardcoded for reproducibility)
# --------------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
VAL_ROOT = BASE_DIR / "data" / "LCZ42" / "val"
CHECKPOINT_PATH = BASE_DIR.parent / "distillation" / "checkpoints" / "densenet201_to_resnet18" / "checkpoints" / "student_resnet18_from_densenet201_last.pth"

# μ/σ from DenseNet201→ResNet18 distillation training on RGB channels
# After paper scaling on 224x224 resized data
# RGB channels in order: [B4, B3, B2]
MU = torch.tensor([9.206003189086914, 9.952054977416992, 11.270723342895508], dtype=torch.float32)
SIGMA = torch.tensor([6.044061660766602, 4.3512349128723145, 3.605332136154175], dtype=torch.float32)

BATCH_SIZE = 512
NUM_WORKERS = 8
DEVICE = "auto"  # auto, cpu, cuda, mps
NUM_CLASSES = 17


def preprocess_png(path: str) -> torch.Tensor:
    """Load 32x32 paper-scaled PNG and prepare for ResNet18.

    Pipeline:
    1. Load 32x32 uint8 PNG (already paper-scaled [0, 255])
    2. Resize to 224x224 (required by ResNet18)
    3. Apply z-score normalization with pre-computed μ/σ
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")

    # Convert BGR → RGB
    img = img[:, :, ::-1]
    img = img.astype(np.float32)

    # Resize from 32x32 to 224x224 (BEFORE z-score normalization)
    # μ/σ are computed on 224x224 resized data
    img_resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_NEAREST)

    # Apply z-score normalization with pre-computed μ/σ
    img_resized = (img_resized - MU.numpy()[None, None, :]) / (SIGMA.numpy()[None, None, :] + 1e-6)

    # Convert (H, W, C) → (C, H, W) for PyTorch
    img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float()
    return img_tensor


# --------------------------------------------------------------------------------------
# Custom dataset reading Voyager-style directory
# --------------------------------------------------------------------------------------
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

    def __getitem__(self, idx: int) -> tuple:
        path, label = self.samples[idx]
        img = preprocess_png(path)
        return img, label


# --------------------------------------------------------------------------------------
# Model and evaluation
# --------------------------------------------------------------------------------------
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
    print(f"[INFO] μ (RGB): {MU.tolist()}")
    print(f"[INFO] σ (RGB): {SIGMA.tolist()}")

    if not CHECKPOINT_PATH.exists():
        print(f"[ERROR] Checkpoint not found: {CHECKPOINT_PATH}")
        return

    model = build_model(CHECKPOINT_PATH, device)
    dataset = PNGDataset(str(VAL_ROOT))
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

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
