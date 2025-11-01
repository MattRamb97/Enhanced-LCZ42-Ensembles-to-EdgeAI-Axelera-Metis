import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "SwinIR")))

import torch
from models.network_swinir import SwinIR

def define_model(scale=2, device="cpu"):
    """
    Defines a SwinIR ×2 model (RGB, pretrained) for per-band super-resolution.
    Each Sentinel-2 band will be replicated into 3 channels before inference.
    """
    # --- Model definition (3-channel RGB pretrained)
    model = SwinIR(
        upscale=scale,
        in_chans=3,
        img_size=64,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,         # <- FIXED (was 60)
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="pixelshuffle",
        resi_connection="1conv",
    )

    # --- Load pretrained weights
    url = (
        "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/"
        "001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth"
    )
    print(f"[INFO] Downloading pretrained SwinIR ×{scale} weights...")
    checkpoint = torch.hub.load_state_dict_from_url(url, map_location="cpu")

    # Some checkpoints use "params_ema", others "params"
    state_dict = checkpoint.get("params_ema") or checkpoint.get("params")
    model.load_state_dict(state_dict, strict=True)

    print("[INFO] Pretrained SwinIR model loaded successfully.")
    return model.to(device)