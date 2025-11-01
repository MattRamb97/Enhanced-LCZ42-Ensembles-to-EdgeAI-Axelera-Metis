import h5py
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------
DATA_DIR = "../../data/lcz42"
INPUT_H5 = f"{DATA_DIR}/training.h5"
SWINIR_H5 = f"{DATA_DIR}/training_swinir2x.h5"

PATCH_INDEX = 0  # which patch to visualize
N_BANDS = 10
SCALE = 2  # ×2 upscaling
CLIP_RANGE = (0, 2.8)

# ---------------------------------------------------------------
# Helper: contrast stretch (same logic as MATLAB version)
# ---------------------------------------------------------------
def contrast_stretch(band):
    low = np.percentile(band, 1)
    high = np.percentile(band, 99)
    stretched = (band - low) / (high - low + 1e-8)
    return np.clip(stretched, 0, 1)

# ---------------------------------------------------------------
# Load datasets
# ---------------------------------------------------------------
with h5py.File(INPUT_H5, "r") as f_in, h5py.File(SWINIR_H5, "r") as f_sr:
    lr_patch = np.array(f_in["sen2"][PATCH_INDEX])
    sr_patch = np.array(f_sr["sen2"][PATCH_INDEX])

    H_lr, W_lr, _ = lr_patch.shape
    H_sr, W_sr, _ = sr_patch.shape
    print(f"[INFO] Patch {PATCH_INDEX}: {H_lr}×{W_lr} → {H_sr}×{W_sr}")

    # Normalize to [0, 1] from reflectance [0, 2.8]
    lr_patch = np.clip(lr_patch / CLIP_RANGE[1], 0, 1)
    sr_patch = np.clip(sr_patch / CLIP_RANGE[1], 0, 1)

# ---------------------------------------------------------------
# Plot each channel comparison
# ---------------------------------------------------------------
fig, axes = plt.subplots(2, N_BANDS, figsize=(2.2 * N_BANDS, 4))
fig.suptitle(f"Patch {PATCH_INDEX}: Bicubic vs SwinIR ×{SCALE}", fontsize=14)

for c in range(N_BANDS):
    lr_band = contrast_stretch(lr_patch[..., c])
    sr_band = contrast_stretch(sr_patch[..., c])

    # Left row → Bicubic
    axes[0, c].imshow(lr_band, cmap="gray", vmin=0, vmax=1)
    axes[0, c].set_title(f"Ch {c+1}\nLR")
    axes[0, c].axis("off")

    # Right row → SwinIR
    axes[1, c].imshow(sr_band, cmap="gray", vmin=0, vmax=1)
    axes[1, c].set_title(f"Ch {c+1}\nSwinIR")
    axes[1, c].axis("off")

plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.show()