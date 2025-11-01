import h5py
import numpy as np
from gtda.homology import CubicalPersistence
from gtda.diagrams import PersistenceImage
import multiprocessing as mp
from joblib import Parallel, delayed
from pathlib import Path

# ---------- Configuration ----------
DATA_DIR = Path("../../data/lcz42")
OUTPUT_DIR = Path("../data")

SR_DATASETS = {
    "bsrnet2x":  "training_bsrnet2x.h5",
    "edsr4x":     "training_edsr4x.h5",
    "realesrgan4x": "training_realesrgan4x.h5",
    "vdsr2x":     "training_vdsr2x.h5",
    "edsr2x":     "training_edsr2x.h5",
    "esrgan2x":   "training_esrgan2x.h5",
    "swinir2x":   "training_swinir2x.h5",
    "vdsr3x":     "training_vdsr3x.h5",
}

n_bins = 30          # persistence image resolution
sigma = 0.5          # Gaussian kernel width
dtype_out = np.float16  # reduces file size ~50%

# ---------- TDA tools ----------
cp = CubicalPersistence(homology_dimensions=(0, 1))
pimg = PersistenceImage(sigma=sigma, n_bins=n_bins)

# ---------- Per-sample processor ----------
def compute_tda_patch(patch, modality_name):
    patch = (patch / (2.8 / 255.0)).clip(0, 255).astype(np.float64)
    num_bands = patch.shape[-1]
    pi_list = []
    for b in range(num_bands):
        img = patch[..., b]
        diagram = cp.fit_transform(img)
        pi = pimg.fit_transform(diagram)
        pi_list.append(pi[0])
    pi_tensor = np.stack(pi_list, axis=0)
    return pi_tensor.reshape(-1).astype(dtype_out)

# ---------- Streamed processing ----------
def process_streamed_modality(f, key, modality_name):
    if key not in f.keys():
        print(f"[WARN] Dataset '{key}' not found in file → skipping {modality_name}")
        return None

    dset = f[key]
    num_samples = dset.shape[0]
    print(f"[INFO] Processing {modality_name}: {num_samples} samples × {dset.shape[-1]} bands")

    n_jobs = max(1, mp.cpu_count() - 2)  # leave 1–2 cores free
    results = Parallel(n_jobs=n_jobs, prefer="processes", verbose=10)(
        delayed(compute_tda_patch)(dset[i], modality_name) for i in range(num_samples)
    )

    tda_array = np.stack(results)
    tda_array_flat = tda_array.reshape(num_samples, -1)
    return tda_array_flat

# ---------- MAIN ----------
if __name__ == "__main__":
    for tag, filename in SR_DATASETS.items():
        h5_path = DATA_DIR / filename
        output_ms_file = OUTPUT_DIR / f"tda_MS_features_{tag}.h5"

        if not h5_path.exists():
            print(f"[SKIP] {h5_path} not found.")
            continue

        print(f"\n[INFO] Opening HDF5 file: {h5_path}")
        with h5py.File(h5_path, "r") as f:
            tda_ms = process_streamed_modality(f, "sen2", "MS")
            if tda_ms is not None:
                with h5py.File(output_ms_file, "w") as out_f:
                    out_f.create_dataset(f"tda_MS_features_{tag}", data=tda_ms)
                print(f"[✓] Saved MS features → {output_ms_file} — Shape: {tda_ms.shape}, dtype={tda_ms.dtype}")

        print(f"[INFO] {tag} completed successfully.")

    print("\n[INFO] All SR datasets processed successfully.")