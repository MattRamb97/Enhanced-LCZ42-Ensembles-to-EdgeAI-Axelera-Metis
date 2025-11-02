import h5py
import numpy as np

def h5_reader(path: str, index: int, modality: str) -> np.ndarray:
    modality = modality.upper()
    if modality == "MS":
        dataset = "/sen2"
    elif modality == "SAR":
        dataset = "/sen1"
    else:
        raise ValueError(f"Unknown modality: {modality}")

    with h5py.File(path, "r") as f:
        dset = f[dataset]
        # h5py uses (N, H, W, C)
        if index < 0 or index >= dset.shape[0]:
            raise IndexError(f"Index {index} out of range 0–{dset.shape[0]-1}")

        patch = dset[index]  # shape: (H, W, C)
        patch = np.array(patch, dtype=np.float32)
        return patch