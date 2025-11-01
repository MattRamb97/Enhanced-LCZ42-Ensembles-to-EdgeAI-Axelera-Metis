import h5py
import numpy as np
import pandas as pd
from scipy.io import savemat
import os

def make_tables_from_h5(data_dir):
    """
    Build tables_MS.mat and tables_SAR.mat from HDF5 dataset.
    Compatible with MATLAB make_tables_from_h5.m
    """
    train_h5 = os.path.join(data_dir, "training.h5")
    test_h5  = os.path.join(data_dir, "testing.h5")

    def build_table(h5_path, modality):
        with h5py.File(h5_path, "r") as f:
            n_samples = f["label"].shape[0]
            print(f"[INFO] {modality} → Found {n_samples} samples in {h5_path}")

        path_list = [h5_path] * n_samples
        label = np.argmax(h5py.File(h5_path, "r")["label"][:], axis=1) + 1  # 1-based
        index = np.arange(1, n_samples + 1)
        modality_list = [modality] * n_samples

        df = pd.DataFrame({
            "Path": path_list,
            "Label": label,
            "Index": index,
            "Modality": modality_list,
        })
        return df

    # --- Build tables ---
    print("[INFO] Building tables for MS (/sen2)...")
    train_MS = build_table(train_h5, "MS")
    test_MS  = build_table(test_h5,  "MS")

    print("[INFO] Building tables for SAR (/sen1)...")
    train_SAR = build_table(train_h5, "SAR")
    test_SAR  = build_table(test_h5,  "SAR")

    # --- Save as MATLAB .mat ---
    out_ms = os.path.join(data_dir, "tables_MS.mat")
    out_sar = os.path.join(data_dir, "tables_SAR.mat")

    savemat(out_ms, {
    "train_MS": train_MS.values,
    "test_MS": test_MS.values
    })
    savemat(out_sar, {
        "train_SAR": train_SAR.values,
        "test_SAR": test_SAR.values
    })

    print(f"Saved {out_ms} and {out_sar} (MATLAB-compatible)")

if __name__ == "__main__":
    make_tables_from_h5("../../data/lcz42")