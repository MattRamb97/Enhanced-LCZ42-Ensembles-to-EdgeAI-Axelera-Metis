import os
import gc
import h5py
import torch
import numpy as np
from tqdm import tqdm
from bsrnet_model import define_model

# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------
DATA_DIR = "../../data/lcz42"
SCALE = 2                     # BSRNet ×2
USE_GAN = False               # True -> BSRGAN (perceptual), False -> BSRNet (PSNR)
WEIGHTS = None                # e.g., "weights/BSRNet.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_FILES = [
    os.path.join(DATA_DIR, "training.h5"),
    os.path.join(DATA_DIR, "testing.h5")
]

BATCH_SIZE = 256               # Adjust for VRAM (A40: 128–256 works fine)

# ---------------------------------------------------------------
# Load pretrained model
# ---------------------------------------------------------------
print(f"[INFO] Loading BSR{'GAN' if USE_GAN else 'Net'} ×{SCALE} model on {DEVICE} ...")
model = define_model(scale=SCALE, device=DEVICE, weights_path=WEIGHTS, use_gan=USE_GAN)
model.eval()
print("[INFO] Model loaded successfully.")

# ---------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------
def prepare_rgb_batch(patch_batch):
    """Expand Sentinel-2 bands into 3-channel RGB tensors."""
    B, H, W, C = patch_batch.shape
    x = np.clip(patch_batch / 2.8, 0, 1).astype(np.float32)
    # (B, H, W, C) → (B*C, 3, H, W)
    bands = x.transpose(0, 3, 1, 2).reshape(-1, 1, H, W)
    bands_rgb = np.repeat(bands, 3, axis=1)
    return torch.from_numpy(bands_rgb)

def process_batch(batch_tensor):
    """Run BSRNet forward pass and return upscaled numpy batch."""
    with torch.no_grad():
        sr = model(batch_tensor.to(DEVICE))
    sr = sr.cpu().clamp(0, 1).numpy()  # (B*C,3,Hs,Ws)
    # Convert back to grayscale [0,2.8]
    sr_gray = sr[:, 0, :, :].astype(np.float32) * 2.8
    return sr_gray

# ---------------------------------------------------------------
# Process datasets (batched, CUDA-optimized)
# ---------------------------------------------------------------
for input_h5 in INPUT_FILES:
    base = os.path.splitext(os.path.basename(input_h5))[0]
    output_h5 = os.path.join(DATA_DIR, f"{base}_bsrnet{SCALE}x.h5")

    with h5py.File(input_h5, "r") as f_in:
        N, H, W, C = f_in["/sen2"].shape
        labels = f_in["/label"][:]
        print(f"\n[INFO] Processing {base}.h5 → {os.path.basename(output_h5)}")
        print(f"       Found {N} patches ({H}×{W}×{C})")

    with h5py.File(output_h5, "w") as f_out:
        f_out.create_dataset("sen2", shape=(N, H*SCALE, W*SCALE, C), dtype="float32")
        f_out.create_dataset("label", data=labels, dtype="uint8")

        with h5py.File(input_h5, "r") as f_in:
            for start in tqdm(range(0, N, BATCH_SIZE), desc=f"{base} ×{SCALE}"):
                end = min(start + BATCH_SIZE, N)
                patch_batch = f_in["/sen2"][start:end].astype(np.float32)  # (B,H,W,C)
                B = patch_batch.shape[0]

                # ---- Preprocess & inference
                x = prepare_rgb_batch(patch_batch)  # (B*C,3,H,W)
                sr_gray = process_batch(x)          # (B*C,Hs,Ws)
                sr_gray = sr_gray.reshape(B, C, H*SCALE, W*SCALE)

                # ---- Reorder back (B,Hs,Ws,C)
                sr_gray = sr_gray.transpose(0, 2, 3, 1)
                f_out["/sen2"][start:end] = sr_gray

                # ---- Cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

    print(f"[INFO] Saved {output_h5}")

print("\nBSRNet processing completed successfully for all datasets.")