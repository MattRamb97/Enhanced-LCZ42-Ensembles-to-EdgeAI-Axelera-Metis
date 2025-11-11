import h5py
import numpy as np
import matplotlib.pyplot as plt

DATASETS = [
    "testing.h5",
    "testing_bsrnet2x.h5",
    "testing_edsr2x.h5",
    "testing_edsr4x.h5",
    "testing_esrgan2x.h5",
    "testing_realesrgan4x.h5",
    "testing_swinir2x.h5",
    "testing_vdsr2x.h5",
    "testing_vdsr3x.h5",
]

PATCH_INDEX = 4 # 638


def preprocess(patch):
    rgb = patch[..., [3, 2, 1]]  # B4,B3,B2
    rgb = np.clip(rgb / 2.8, 0, 1)
    low, high = np.percentile(rgb, (1, 99))
    rgb = np.clip((rgb - low) / (high - low), 0, 1)
    rgb = np.power(rgb, 1 / 2.2)  # gamma correction
    return rgb


images = []
for fname in DATASETS:
    with h5py.File(fname, "r") as f:
        patch = f["sen2"][PATCH_INDEX]
    images.append((preprocess(patch), fname))

rows = 2
cols = int(np.ceil(len(images) / rows))
fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 3))
axes = axes.flatten()
for ax, (img, title) in zip(axes, images):
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(title, fontsize=8)

for ax in axes[len(images) :]:
    ax.axis("off")

plt.tight_layout()
plt.show()
