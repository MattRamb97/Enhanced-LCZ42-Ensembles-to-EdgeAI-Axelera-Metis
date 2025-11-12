"""
Export the distilled ResNet18 student to ONNX for Voyager/Axelera Metis deployment.

Simple one-line execution: python export_student_resnet18.py
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import resnet18

# --------------------------------------------------------------------------------------
# Configuration (hardcoded for reproducibility)
# --------------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = BASE_DIR / "checkpoints" / "student_resnet18_last.pth"
OUTPUT_PATH = BASE_DIR / "checkpoints" / "student_resnet18_rgb.onnx"

NUM_CLASSES = 17
OPSET_VERSION = 17  # Axelera Metis requires opset 17
INPUT_SHAPE = (1, 3, 224, 224)  # (batch, channels, height, width)


def load_student(checkpoint_path: Path, num_classes: int) -> nn.Module:
    """Load distilled ResNet18 student from checkpoint."""
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def main() -> None:
    """Export student model to ONNX format for edge deployment."""
    print(f"[INFO] Loading checkpoint: {CHECKPOINT_PATH}")
    model = load_student(CHECKPOINT_PATH, NUM_CLASSES)

    print(f"[INFO] Creating dummy input with shape {INPUT_SHAPE}")
    dummy_input = torch.randn(*INPUT_SHAPE)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Exporting to ONNX (opset {OPSET_VERSION})...")
    torch.onnx.export(
        model,
        dummy_input,
        OUTPUT_PATH,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}},
        opset_version=OPSET_VERSION,
        do_constant_folding=True,
        dynamo=False,  # Use legacy exporter to avoid onnxscript compatibility issues
    )

    print(f"[OK] Successfully exported ONNX → {OUTPUT_PATH}")
    print(f"[OK] Model ready for Voyager/Metis deployment")


if __name__ == "__main__":
    main()
