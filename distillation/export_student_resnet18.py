"""
Export the distilled ResNet18 student to ONNX for Voyager deployment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export distilled ResNet18 student to ONNX.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(Path("checkpoints") / "student_resnet18_best.pth"),
        help="Path to the student checkpoint (.pth).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path("checkpoints") / "student_resnet18_best.onnx"),
        help="Destination ONNX file.",
    )
    parser.add_argument(
        "--num-classes", type=int, default=17, help="Number of LCZ classes (default: 17)."
    )
    parser.add_argument(
        "--opset", type=int, default=17, help="ONNX opset version (default: 17)."
    )
    parser.add_argument(
        "--use-pretrained",
        action="store_true",
        help="Initialise backbone with ImageNet weights before loading the checkpoint.",
    )
    return parser.parse_args()


def load_student(num_classes: int, checkpoint_path: Path, use_pretrained: bool) -> nn.Module:
    weights = ResNet18_Weights.IMAGENET1K_V1 if use_pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    state = torch.load(checkpoint_path, map_location="cpu")
    missing = model.load_state_dict(state, strict=False)
    if missing.missing_keys or missing.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch — missing={missing.missing_keys} unexpected={missing.unexpected_keys}"
        )
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = load_student(args.num_classes, checkpoint_path, args.use_pretrained)
    dummy = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}},
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print(f"[OK] Exported ONNX → {output_path}")


if __name__ == "__main__":
    main()
