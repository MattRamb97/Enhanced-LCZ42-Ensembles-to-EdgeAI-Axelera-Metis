"""
Export the distilled ResNet18 student to ONNX for Voyager/Axelera Metis deployment.

"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import resnet18

# --------------------------------------------------------------------------------------
# Configuration (hardcoded defaults, overridable via argparse)
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


def main(checkpoint_path: Path | None = None, output_path: Path | None = None) -> None:
    """Export student model to ONNX format for edge deployment.

    Args:
        checkpoint_path: Path to checkpoint. If None, uses default CHECKPOINT_PATH.
        output_path: Path to output ONNX file. If None, uses default OUTPUT_PATH.
    """
    if checkpoint_path is None:
        checkpoint_path = CHECKPOINT_PATH
    if output_path is None:
        output_path = OUTPUT_PATH

    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    model = load_student(checkpoint_path, NUM_CLASSES)

    print(f"[INFO] Creating dummy input with shape {INPUT_SHAPE}")
    dummy_input = torch.randn(*INPUT_SHAPE)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Exporting to ONNX (opset {OPSET_VERSION})...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}},
        opset_version=OPSET_VERSION,
        do_constant_folding=True,
        dynamo=False,  # Use legacy exporter to avoid onnxscript compatibility issues
    )

    print(f"[OK] Successfully exported ONNX → {output_path}")
    print(f"[OK] Model ready for Voyager/Metis deployment")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export distilled ResNet18 student to ONNX for deployment"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT_PATH,
        help="Path to student checkpoint (default: resnet18_to_resnet18/student_resnet18_last.pth)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Path to output ONNX file (default: checkpoints/student_resnet18_rgb.onnx)",
    )
    args = parser.parse_args()
    main(checkpoint_path=args.checkpoint, output_path=args.output)
