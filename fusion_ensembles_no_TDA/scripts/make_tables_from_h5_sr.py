import os
import h5py
import numpy as np
import pandas as pd
from scipy.io import savemat

def build_table(h5_path, modality="MS"):
    with h5py.File(h5_path, "r") as f:
        n_samples = f["label"].shape[0]
        labels_1b = np.argmax(f["label"][:], axis=1) + 1  # 1-based
    print(f"[INFO] {modality} → Found {n_samples} samples in {h5_path}")

    df = pd.DataFrame({
        "Path":     [h5_path] * n_samples,
        "Label":    labels_1b,
        "Index":    np.arange(1, n_samples + 1),  # 1-based
        "Modality": [modality] * n_samples,
    })
    return df

def make_tables_MS_for_suffix(data_dir, suffix):
    train_h5 = os.path.join(data_dir, f"training{suffix}.h5")
    test_h5  = os.path.join(data_dir, f"testing{suffix}.h5")

    if not (os.path.exists(train_h5) and os.path.exists(test_h5)):
        print(f"[WARN] Missing {train_h5} or {test_h5} — skipping {suffix}")
        return

    print(f"[INFO] Building tables for MS (/sen2){suffix} ...")
    train_MS = build_table(train_h5, "MS")
    test_MS  = build_table(test_h5,  "MS")

    out_ms = os.path.join(data_dir, f"tables_MS{suffix}.mat")
    savemat(out_ms, {
        f"train_MS{suffix}": train_MS.values,
        f"test_MS{suffix}":  test_MS.values
    })
    print(f"[✓] Saved {out_ms}")

if __name__ == "__main__":
    data_dir = "../../data/lcz42"
    methods = [
        "_vdsr2x",
        "_edsr2x",
        "_edsr4x",
        "_esrgan2x",
        "_swinir2x",
        "_vdsr3x",
        "_bsrnet2x",
        "_realesrgan4x",
    ]
    for suf in methods:
        make_tables_MS_for_suffix(data_dir, suf)