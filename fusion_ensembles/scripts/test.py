import h5py, numpy as np
from dataset_reading import DatasetReading
from train_teacher_fusion import load_table_mat

train_table, test_table = load_table_mat("../../data/lcz42/tables_MS.mat", "train_MS", "test_MS")
dsTr, dsTe, info = DatasetReading(dict(trainTable=train_table, testTable=test_table, useZscore=True, useAugmentation=True))
indices = dsTr.table["Index"].to_numpy(dtype=np.int32)
with h5py.File("../../TDA/data/labels.h5") as f:
    labels_from_h5 = f["labels"][:]
mismatch = np.where(labels_from_h5[indices] != dsTr.table["Label"].to_numpy())[0]
print("mismatch count:", mismatch.size)
