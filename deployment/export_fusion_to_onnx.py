"""
export_fusion_to_onnx.py
Converts trained TDA-Fusion ResNet18 models (.pth) to ONNX.
Author: Matteo Rambaldi — Thesis utilities
"""

import os
import sys
import torch

# ------------------------------------------------------------
# Add TDA/models to path so import works from deployment folder
# ------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TDA_DIR = os.path.join(ROOT_DIR, "TDA")
sys.path.append(TDA_DIR)

from models.fusion_resnet18 import TDAFusionResNet18

# ------------------ CONFIG ------------------
MODEL_DIR = os.path.join(TDA_DIR, "models")

MODELS = {
    "fusion_resnet18_rand.pth": ("fusion_resnet18_rand.onnx", 18000),
    "fusion_resnet18_randrgb.pth": ("fusion_resnet18_randrgb.onnx", 18000),
    "fusion_resnet18_ensemblesar.pth": ("fusion_resnet18_ensemblesar.onnx", 14400),
}

NUM_CLASSES = 17
OPSET_VERSION = 17  # MATLAB-friendly (use 17 for Axelera export later)

# Choose device automatically (MPS for Apple Silicon, CUDA if NVIDIA GPU, else CPU)
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"[INFO] Using device: {DEVICE}")

# ------------------ EXPORT LOOP ------------------
for pth_name, (onnx_name, tda_dim) in MODELS.items():
    pth_path = os.path.join(MODEL_DIR, pth_name)
    onnx_path = os.path.join(MODEL_DIR, onnx_name)

    if not os.path.isfile(pth_path):
        print(f"[WARN] File not found: {pth_path} — skipping.")
        continue

    print(f"\n[INFO] Exporting {pth_name} → {onnx_name}")

    # Load architecture and weights
    model = TDAFusionResNet18(tda_input_dim=tda_dim, num_classes=NUM_CLASSES)
    ckpt = torch.load(pth_path, map_location="cpu")
    model.load_state_dict(ckpt)
    model = model.to(DEVICE).eval()

    # Dummy inputs: batch 1 (3 × 224 × 224 image, tda vector)
    img_dummy = torch.randn(1, 3, 224, 224, device=DEVICE)
    tda_dummy = torch.randn(1, tda_dim, device=DEVICE)

    with torch.no_grad():
        torch.onnx.export(
            model,
            (img_dummy, tda_dummy),
            onnx_path,
            input_names=["image", "tda"],
            output_names=["logits"],
            opset_version=OPSET_VERSION,
            do_constant_folding=True,
            dynamo=False,             # <<< force legacy exporter
            dynamic_axes=None         # <<< avoid the dynamo warning path
        )

    print(f"[OK] Saved: {onnx_path}")

print("\n All exports complete.")