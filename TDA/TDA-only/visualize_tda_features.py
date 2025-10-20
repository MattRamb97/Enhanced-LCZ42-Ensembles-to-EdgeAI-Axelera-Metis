import numpy as np
import matplotlib.pyplot as plt

# ---------- Load ----------
tda_file = "tda_features.npy"
features = np.load(tda_file)  # (samples, bands, homology_dims, n_bins, n_bins)
print("Loaded shape:", features.shape)

# Choose which sample to visualize
sample_idx = 0
sample = features[sample_idx]  # shape: (10, 2, n_bins, n_bins)

# ---------- Plot ----------
fig, axs = plt.subplots(nrows=10, ncols=2, figsize=(8, 20))
fig.suptitle(f"TDA Features — Sample {sample_idx}", fontsize=16)

for band in range(10):
    for h_dim in range(2):
        ax = axs[band, h_dim]
        ax.imshow(sample[band, h_dim], cmap="viridis")
        ax.set_title(f"Band {band} — H{h_dim}")
        ax.axis("off")

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.show()