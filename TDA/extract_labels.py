import h5py
import numpy as np

# ---------- Path ----------
h5_path = "../data/lcz42/training.h5"
output_path = "labels.npy"

# ---------- Extract ----------
with h5py.File(h5_path, "r") as f:
    if "label" in f:
        labels = f["label"][:]  # (N, 17) one-hot
        print(f"Loaded {labels.shape[0]} labels with shape {labels.shape}.")
    else:
        raise KeyError("No 'label' key found in HDF5 file!")
    
# ---------- Convert one-hot → class indices ----------
labels = np.argmax(labels, axis=1).astype(np.int64) + 1
print("Converted to class indices:", np.unique(labels), "(unique classes)")

# ---------- Save ----------
np.save(output_path, labels)
print(f"Saved labels to {output_path}")