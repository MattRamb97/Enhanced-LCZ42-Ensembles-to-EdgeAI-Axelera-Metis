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

    Args:
        scale (int): upscaling factor (2 or 4). We use 2 per your requirement.
        device (str or torch.device): 'cuda' or 'cpu'
        weights_path (str|None): path to .pth. If None, defaults to weights/BSRNet.pth
        use_gan (bool): if True, load BSRGAN.pth (perceptual); else BSRNet.pth (PSNR)

    Returns:
        torch.nn.Module on the requested device, set to eval().
    """
    # RRDBNet configuration used by BSRGAN/BSRNet
    # (Compatible with ESRGAN/Real-ESRGAN small model)
    model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=scale)

    # Default weights location (inside ./weights)
    if weights_path is None:
        wname = "BSRGAN.pth" if use_gan else "BSRNet.pth"
        weights_path = os.path.join(_THIS_DIR, "weights", wname)

    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"Pretrained weights not found: {weights_path}\n"
            f"→ Download from the BSRGAN repo model_zoo and place it there."
        )

    print(f"[INFO] Loading {'BSRGAN' if use_gan else 'BSRNet'} weights: {weights_path}")
    ckpt = torch.load(weights_path, map_location="cpu")
    # Some checkpoints store directly the state_dict, others under a key
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    # Key names in KAIR/BSRGAN are usually already aligned to RRDBNet
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] Missing keys: {missing[:5]}{'...' if len(missing)>5 else ''}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")

    model.to(device).eval()
    print(f"[INFO] BSRNet model ready on {device}")
    return model