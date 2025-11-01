import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
import h5py

DATA_DIR = "../../data/lcz42"

def load_table(mat_path, train_key, test_key):
    """Load .mat tables (NumPy arrays) into pandas DataFrames."""
    data = loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    def arr_to_df(arr):
        # Expect shape (N, 4): [Path, Label, Index, Modality]
        cols = ["Path", "Label", "Index", "Modality"]
        df = pd.DataFrame(arr, columns=cols)
        # Convert types
        df["Path"] = df["Path"].astype(str)
        df["Label"] = df["Label"].astype(int)
        df["Index"] = df["Index"].astype(int)
        df["Modality"] = df["Modality"].astype(str)
        return df

    train_df = arr_to_df(data[train_key])
    test_df = arr_to_df(data[test_key])
    return train_df, test_df


def inspect_table(df, name):
    """Print quick summary for one table."""
    print(f"\n{name} — {len(df)} samples")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Unique labels: {len(set(df['Label']))}")
    print(f"  Modalities: {set(df['Modality'])}")
    print(f"  Example rows:")
    print(df.head(3).to_string(index=False))


def check_h5(path):
    """Quick HDF5 inspection to confirm structure."""
    if not os.path.exists(path):
        print(f"  Missing file: {path}")
        return
    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        print(f"  HDF5 keys: {keys}")
        for key in keys:
            print(f"    {key}: shape={f[key].shape}, dtype={f[key].dtype}")


def main():
    ms_path = os.path.join(DATA_DIR, "tables_MS.mat")
    sar_path = os.path.join(DATA_DIR, "tables_SAR.mat")

    if not os.path.exists(ms_path) or not os.path.exists(sar_path):
        print("Missing .mat tables. Run make_tables_from_h5.py first.")
        return

    # Load tables
    train_MS, test_MS = load_table(ms_path, "train_MS", "test_MS")
    train_SAR, test_SAR = load_table(sar_path, "train_SAR", "test_SAR")

    # Print stats
    inspect_table(train_MS, "TRAIN_MS")
    inspect_table(test_MS, "TEST_MS")
    inspect_table(train_SAR, "TRAIN_SAR")
    inspect_table(test_SAR, "TEST_SAR")

    # Inspect the linked .h5 files
    print("\nChecking referenced HDF5 file structure:")
    example_path = train_MS["Path"].iloc[0]
    check_h5(example_path)

    print("\nTable inspection complete.")


if __name__ == "__main__":
    main()