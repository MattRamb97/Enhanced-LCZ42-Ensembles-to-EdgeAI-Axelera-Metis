import os, sys
import torch
import numpy as np
import onnxruntime as ort

# --- fix import path ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(ROOT, "TDA"))
from models.fusion_resnet18 import TDAFusionResNet18

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "TDA", "models")
CHECKPOINTS = {
    "fusion_resnet18_rand": 18000,
    "fusion_resnet18_randrgb": 18000,
    "fusion_resnet18_ensemblesar": 14400,
}

device = "mps" if torch.backends.mps.is_available() else "cpu"

for name, tda_dim in CHECKPOINTS.items():
    print(f"\n[TEST] {name}")
    pth = os.path.join(MODEL_DIR, f"{name}.pth")
    onnx_path = os.path.join(MODEL_DIR, f"{name}.onnx")

    # Load PyTorch model
    model = TDAFusionResNet18(tda_input_dim=tda_dim, num_classes=17)
    model.load_state_dict(torch.load(pth, map_location="cpu"))
    model.eval()

    # Dummy inputs
    img = torch.randn(1, 3, 224, 224)
    tda = torch.randn(1, tda_dim)

    with torch.no_grad():
        out_torch = model(img, tda).numpy()

    # Run ONNX inference
    ort_sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    out_onnx = ort_sess.run(["logits"], {"image": img.numpy(), "tda": tda.numpy()})[0]

    # Compare
    diff = np.abs(out_torch - out_onnx).max()
    print(f"Max |Δ| = {diff:.6f}")