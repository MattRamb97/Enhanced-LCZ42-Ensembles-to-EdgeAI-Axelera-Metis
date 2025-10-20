import h5py
import numpy as np
from tqdm import tqdm
from gtda.homology import CubicalPersistence
from gtda.diagrams import PersistenceImage

# ---------- Configuration ----------
h5_path = "../data/lcz42/training.h5"
n_bins = 30          # image resolution
sigma = 0.5          # Gaussian kernel width
output_ms_file = "tda_MS_features.npy"
output_sar_file = "tda_SAR_features.npy"

# ---------- TDA tools ----------
cp = CubicalPersistence(homology_dimensions=(0, 1))
pimg = PersistenceImage(sigma=sigma, n_bins=n_bins)

# ---------- Streaming patch-wise processing ----------
def process_streamed_modality(f, key, modality_name):
    dset = f[key]
    num_samples = dset.shape[0]
    num_bands = dset.shape[-1]
    tda_all = []

    print(f"[INFO] Processing {modality_name} — {num_samples} samples, {num_bands} bands")

    for i in tqdm(range(num_samples)):
        patch = dset[i]  # shape: (32, 32, C)
        patch = (patch / (2.8 / 255.0)).clip(0, 255).astype(np.float64)

        # Process per band, or optionally full patch here
        pi_list = []
        for band in range(num_bands):
            img = patch[..., band]  # shape: (32, 32)
            diagram = cp.fit_transform(img)  # shape: (1, n_points, 3)
            pi = pimg.fit_transform(diagram)                  # shape: (1, 2, n_bins, n_bins)
            pi_list.append(pi[0])  # drop batch dim

        pi_tensor = np.stack(pi_list, axis=0)  # (10, 2, n_bins, n_bins)
        tda_all.append(pi_tensor)

    tda_array = np.stack(tda_all)  # (N, 10, 2, n_bins, n_bins)
    tda_array_flat = tda_array.reshape(num_samples, -1)
    tda_array_flat = tda_array_flat.astype(np.float32)
    return tda_array_flat

# ---------- MAIN ----------
if __name__ == "__main__":
    with h5py.File(h5_path, "r") as f:
        tda_ms = process_streamed_modality(f, "sen2", "MS")
        np.save(output_ms_file, tda_ms)
        print(f"Saved MS features → {output_ms_file} — Shape: {tda_ms.shape}")

        tda_sar = process_streamed_modality(f, "sen1", "SAR")
        np.save(output_sar_file, tda_sar)
        print(f"Saved SAR features → {output_sar_file} — Shape: {tda_sar.shape}")