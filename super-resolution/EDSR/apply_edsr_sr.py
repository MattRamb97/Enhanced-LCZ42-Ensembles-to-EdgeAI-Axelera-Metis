import h5py
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from super_image import EdsrModel, ImageLoader

# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------
MODE = "testing"  # "training" or "testing"
INPUT_H5  = f"../../data/lcz42/{MODE}.h5"
OUTPUT_H5 = f"../../data/lcz42/{MODE}_edsr2x.h5"
SCALE = 2
MODEL_ID = "eugenesiow/edsr-base"   # pretrained EDSR ×2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------
# Load model once
# ---------------------------------------------------------------
print(f"[INFO] Loading pretrained EDSR model ({MODEL_ID}) on {DEVICE}...")
model = EdsrModel.from_pretrained(MODEL_ID, scale=SCALE).to(DEVICE)
model.eval()

## ---------------------------------------------------------------
# READ INPUT FILE INFO
# ---------------------------------------------------------------
with h5py.File(INPUT_H5, "r") as f_in:
    N = f_in["/sen2"].shape[0]
    H, W, C = f_in["/sen2"].shape[1:]
    labels = f_in["/label"][:]  # one-hot (N,17)
    print(f"[INFO] Found {N} patches ({H}×{W}×{C})")

# ---------------------------------------------------------------
# CREATE OUTPUT FILE
# ---------------------------------------------------------------
with h5py.File(OUTPUT_H5, "w") as f_out:
    f_out.create_dataset("sen2", shape=(N, H*SCALE, W*SCALE, C), dtype="float32")
    f_out.create_dataset("label", data=labels, dtype="uint8")
print(f"[INFO] Created output file: {OUTPUT_H5}")

# ---------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------
with h5py.File(INPUT_H5, "r") as f_in, h5py.File(OUTPUT_H5, "a") as f_out:
    for i in tqdm(range(N), desc=f"Applying EDSR ×{SCALE}"):
        patch = f_in["/sen2"][i].astype(np.float32)  # shape (32,32,10)
        out_patch = np.zeros((H*SCALE, W*SCALE, C), dtype=np.float32)

        for c in range(C):
            # Normalize [0, 2.8] → [0, 1]
            band = np.clip(patch[..., c] / 2.8, 0.0, 1.0).astype(np.float32)
            band3 = np.repeat(band[..., None], 3, axis=-1)  # fake RGB
            tensor = (
                torch.from_numpy(band3.transpose(2, 0, 1))
                .unsqueeze(0)
                .to(DEVICE)
            )  # [1,3,H,W]

            with torch.no_grad():
                preds = model(tensor).clamp(0.0, 1.0)

            sr = preds.squeeze(0)[0].cpu().numpy() * 2.8  # take 1st chan, back to [0, 2.8]
            out_patch[..., c] = sr.astype(np.float32)

        f_out["/sen2"][i] = out_patch

print(f"Finished: saved EDSR×{SCALE} dataset to {OUTPUT_H5}")