import h5py
import numpy as np
import torch
from tqdm import tqdm
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
import os
import gc

# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------
DATA_DIR = "../../data/lcz42"
SCALE = 4  # can also set to 2
MODEL_PATH = f"RealESRGAN_x{SCALE}plus.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_FILES = [
    os.path.join(DATA_DIR, "training.h5"),
    os.path.join(DATA_DIR, "testing.h5")
]

# ---------------------------------------------------------------
# Load pretrained model
# ---------------------------------------------------------------
print(f"[INFO] Loading pretrained {MODEL_PATH} on {DEVICE} ...")
model = RealESRGANer(
    scale=SCALE,
    model_path=MODEL_PATH,
    model=RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=SCALE,
    ),
    half=False,
    device=DEVICE,
)
print("[INFO] Model loaded successfully.")

# ---------------------------------------------------------------
# Process datasets
# ---------------------------------------------------------------
for input_h5 in INPUT_FILES:
    base = os.path.splitext(os.path.basename(input_h5))[0]
    output_h5 = os.path.join(DATA_DIR, f"{base}_realesrgan{SCALE}x.h5")

    with h5py.File(input_h5, "r") as f_in:
        N, H, W, C = f_in["/sen2"].shape
        labels = f_in["/label"][:]
        print(f"\n[INFO] Processing {base}.h5 → {base}_realesrgan{SCALE}x.h5")
        print(f"       Found {N} patches ({H}×{W}×{C})")

    with h5py.File(output_h5, "w") as f_out:
        f_out.create_dataset("sen2", shape=(N, H*SCALE, W*SCALE, C), dtype="float32")
        f_out.create_dataset("label", data=labels, dtype="uint8")

        with h5py.File(input_h5, "r") as f_in:
            for i in tqdm(range(N), desc=f"{base} ×{SCALE}"):
                patch = f_in["/sen2"][i].astype(np.float32)
                out_patch = np.zeros((H*SCALE, W*SCALE, C), dtype=np.float32)

                for c in range(C):
                    # --- Normalize [0, 2.8] → [0, 1]
                    band = np.clip(patch[..., c] / 2.8, 0, 1)

                    # --- Fake 3-channel RGB image for model
                    band3 = np.repeat(band[..., None], 3, axis=-1)
                    img_bgr = (band3[:, :, ::-1] * 255).astype(np.uint8)  # RGB→BGR

                    # --- Super-resolve (Real-ESRGAN expects NumPy BGR)
                    with torch.no_grad():
                        sr_bgr, _ = model.enhance(img_bgr, outscale=SCALE)

                    # --- Convert back to single-channel float32 [0, 2.8]
                    sr_gray = sr_bgr[:, :, 0].astype(np.float32) / 255.0 * 2.8
                    out_patch[..., c] = sr_gray

                f_out["/sen2"][i] = out_patch

                if i % 500 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    print(f"Saved {output_h5}")

print("\nReal-ESRGAN processing completed successfully for all datasets.")