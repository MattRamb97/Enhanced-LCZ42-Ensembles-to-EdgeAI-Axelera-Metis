import h5py
import numpy as np
from pathlib import Path

# ---------- Configuration ----------
datasets = {
    "train": {
        "input": "../../data/lcz42/training.h5",
        "output": "../data/labels.h5",
    },
    "test": {
        "input": "../../data/lcz42/testing.h5",
        "output": "../data/labels_test.h5",
    },
}

# ---------- Function ----------
def extract_labels(h5_path: str, output_path: str):
    """Convert one-hot labels in an HDF5 file to integer class indices."""
    assert Path(h5_path).exists(), f"[ERROR] HDF5 file not found: {h5_path}"
    print(f"\n[INFO] Reading labels from: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        if "label" not in f:
            raise KeyError("No 'label' dataset found in HDF5 file!")
        labels = f["label"][:]  # shape (N, 17)
        print(f"[INFO] Loaded labels — shape: {labels.shape}, dtype: {labels.dtype}")

    # Convert one-hot → class indices (1–17)
    labels = np.argmax(labels, axis=1).astype(np.uint8) + 1
    print(f"[INFO] Converted to class indices in range [{labels.min()}, {labels.max()}]")

    # Save as compact HDF5
    with h5py.File(output_path, "w") as f_out:
        f_out.create_dataset("labels", data=labels, dtype=np.uint8)
    print(f"[✓] Saved labels → {output_path} — shape: {labels.shape}, dtype=uint8")

    # Verify
    with h5py.File(output_path, "r") as f_verify:
        check = f_verify["labels"][:]
        print(f"[INFO] Verified readback: {check.shape} elements | dtype={check.dtype}")
        print(f"[INFO] Unique values: {np.unique(check)}")


# ---------- Main ----------
if __name__ == "__main__":
    for name, paths in datasets.items():
        extract_labels(paths["input"], paths["output"])

    print("\n[INFO] Label extraction completed successfully for all datasets.")