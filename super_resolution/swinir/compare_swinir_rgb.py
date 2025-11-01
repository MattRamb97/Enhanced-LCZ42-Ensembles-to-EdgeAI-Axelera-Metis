import h5py
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------
DATA_DIR = "../../data/lcz42"
INPUT_H5 = f"{DATA_DIR}/training.h5"
SWINIR_H5 = f"{DATA_DIR}/training_swinir2x.h5"

PATCH_INDEX = 0        # patch to visualize
BANDS_RGB = (3, 2, 1)  # Sentinel-2 RGB = B4,B3,B2
CLIP_RANGE = (0, 2.8)

# ---------------------------------------------------------------
# Helper
# ---------------------------------------------------------------
def contrast_stretch(img):
    """Percentile-based stretch per channel."""
    out = np.zeros_like(img)
    for c in range(img.shape[2]):
        ch = img[..., c]
        low, high = np.percentile(ch, (1, 99))
        out[..., c] = np.clip((ch - low) / (high - low + 1e-8), 0, 1)
    return out

# ---------------------------------------------------------------
# Load and process
# ---------------------------------------------------------------
with h5py.File(INPUT_H5, "r") as f_in, h5py.File(SWINIR_H5, "r") as f_sr:
    lr_patch = np.array(f_in["sen2"][PATCH_INDEX])
    sr_patch = np.array(f_sr["sen2"][PATCH_INDEX])

# Extract RGB channels and normalize
lr_rgb = np.clip(lr_patch[..., list(BANDS_RGB)] / CLIP_RANGE[1], 0, 1)
sr_rgb = np.clip(sr_patch[..., list(BANDS_RGB)] / CLIP_RANGE[1], 0, 1)

# Apply contrast stretch
lr_rgb = contrast_stretch(lr_rgb)
sr_rgb = contrast_stretch(sr_rgb)

# ---------------------------------------------------------------
# Plot
# ---------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(6, 3))
ax[0].imshow(lr_rgb)
ax[0].set_title(f"Bicubic RGB (idx={PATCH_INDEX})\n{lr_rgb.shape[0]}×{lr_rgb.shape[1]}")
ax[0].axis("off")

ax[1].imshow(sr_rgb)
ax[1].set_title(f"SwinIR ×2 RGB\n{sr_rgb.shape[0]}×{sr_rgb.shape[1]}")
ax[1].axis("off")

plt.tight_layout()
plt.show()