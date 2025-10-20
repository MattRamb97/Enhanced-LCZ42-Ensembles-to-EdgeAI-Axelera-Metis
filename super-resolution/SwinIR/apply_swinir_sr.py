import h5py
import numpy as np
import torch
from tqdm import tqdm
from swinir_model import define_model
import os
import gc

# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------
DATA_DIR = "../../data/lcz42"
SCALE = 2  # SwinIR is ×2 (can adjust if needed)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_FILES = [
    os.path.join(DATA_DIR, "training.h5"),
    os.path.join(DATA_DIR, "testing.h5")
]

# ---------------------------------------------------------------
# Load pretrained model
# ---------------------------------------------------------------
print(f"[INFO] Loading SwinIR ×{SCALE} model on {DEVICE} ...")
model = define_model(scale=SCALE, device=DEVICE)
model.eval()
print("[INFO] SwinIR model loaded successfully.")

# ---------------------------------------------------------------
# Process datasets
# ---------------------------------------------------------------
for input_h5 in INPUT_FILES:
    base = os.path.splitext(os.path.basename(input_h5))[0]
    output_h5 = os.path.join(DATA_DIR, f"{base}_swinir{SCALE}x.h5")

    with h5py.File(input_h5, "r") as f_in:
        N, H, W, C = f_in["/sen2"].shape
        labels = f_in["/label"][:]
        print(f"\n[INFO] Processing {base}.h5 → {base}_swinir{SCALE}x.h5")
        print(f"       Found {N} patches ({H}×{W}×{C})")

    with h5py.File(output_h5, "w") as f_out:
        f_out.create_dataset("sen2", shape=(N, H * SCALE, W * SCALE, C), dtype="float32")
        f_out.create_dataset("label", data=labels, dtype="uint8")

        with h5py.File(input_h5, "r") as f_in:
            for i in tqdm(range(N), desc=f"{base} ×{SCALE}"):
                patch = f_in["/sen2"][i].astype(np.float32)
                out_patch = np.zeros((H * SCALE, W * SCALE, C), dtype=np.float32)

                for c in range(C):
                    # --- Normalize [0, 2.8] → [0, 1]
                    band = np.clip(patch[..., c] / 2.8, 0, 1)

                    # --- Fake 3-channel RGB image for model
                    band3 = np.repeat(band[..., None], 3, axis=-1)
                    img = torch.from_numpy(band3).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

                    # --- Super-resolve
                    with torch.no_grad():
                        sr = model(img)
                    sr = sr.squeeze(0).permute(1, 2, 0).cpu().numpy()

                    # --- Clamp & rescale back to [0, 2.8]
                    sr = np.clip(sr, 0, 1) * 2.8
                    out_patch[..., c] = sr[..., 0]  # take one channel (they're identical)

                f_out["/sen2"][i] = out_patch

                # Optional cleanup (safe to remove for A40/3090)
                if i % 500 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    print(f"[INFO] Saved {output_h5}")

print("\nSwinIR processing completed successfully for all datasets.")