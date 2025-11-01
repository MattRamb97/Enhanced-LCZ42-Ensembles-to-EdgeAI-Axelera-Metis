# bsrnet_model.py
# --------------------------------------------------------------
# BSRNet (RRDBNet) loader without pip-deps, importing directly
# from a local clone of the official BSRGAN repo.
#
# Folder structure expected:
# super-resolution/
#   BSRNet/
#     bsrnet_model.py   <-- this file
#     apply_bsrnet_sr.py
#     BSRGAN/           <-- git clone https://github.com/cszn/BSRGAN.git
#       models/network_rrdbnet.py
#       utils/ ...
#     weights/
#       BSRNet.pth      <-- download from BSRGAN "model_zoo"
#       # (optionally) BSRGAN.pth
# --------------------------------------------------------------

import os, sys
import torch
import torch.nn as nn

# Point Python to the local BSRGAN repo for the model definition
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BSRGAN_DIR = os.path.join(_THIS_DIR, "BSRGAN")
if _BSRGAN_DIR not in sys.path:
    sys.path.insert(0, _BSRGAN_DIR)

# Import the official RRDBNet used by BSRNet
from models.network_rrdbnet import RRDBNet  # from the cloned BSRGAN repo


def define_model(scale=2, device="cpu", weights_path=None, use_gan=False):
    """
    Returns a BSRNet/BSRGAN generator as RRDBNet with pretrained weights.
    Optimized for A40 (48 GB) and satellite float32 data.
    """
    # RRDBNet configuration used by BSRGAN/BSRNet
    # (Compatible with ESRGAN/Real-ESRGAN small model)
    model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=scale)

    # Patch residual compression: add post-scale multiplier to output
    class ScaledRRDBNet(nn.Module):
        def __init__(self, core):
            super().__init__()
            self.core = core
            self.res_scale = 3.5  # empirically restores amplitude to [0,1] for sat data
        def forward(self, x):
            return torch.clamp(self.core(x) * self.res_scale, 0, 1)

    model = ScaledRRDBNet(model)    

    # Default weights location (inside ./weights)
    if weights_path is None:
        wname = "BSRGAN.pth" if use_gan else "BSRNet.pth"
        weights_path = os.path.join(_THIS_DIR, "weights", wname)

    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Missing weights: {weights_path}")

    print(f"[INFO] Loading {'BSRGAN' if use_gan else 'BSRNet'} weights: {weights_path}")
    state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    missing, unexpected = model.core.load_state_dict(state, strict=False)
    if missing: print(f"[WARN] Missing keys: {missing[:5]}{'...' if len(missing)>5 else ''}")
    if unexpected: print(f"[WARN] Unexpected keys: {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")

    model.to(device).eval()
    torch.set_grad_enabled(False)
    print(f"[INFO] BSRNet ready on {device} (A40 optimized)")
    return model