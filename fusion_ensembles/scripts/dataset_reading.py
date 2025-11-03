import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import random
import pandas as pd
from tqdm import tqdm

# -----------------------------------------------------
# Helper functions
# -----------------------------------------------------

def apply_paper_scaling(x: np.ndarray, modality: str) -> np.ndarray:
    """Apply paper scaling identical to MATLAB's applyPaperScaling."""
    x = x.astype(np.float32)
    if modality.upper() == "SAR":
        x = np.clip(x, -0.5, 0.5)
        x = (x + 0.5) * 255.0
    else:  # MS
        x = x / (2.8 / 255.0)
    return x


def compute_band_stats(table: pd.DataFrame):
    """Compute per-channel mean/std after paper scaling (TRAIN only)."""
    acc_mu, acc_sq, n_pix, C = None, None, 0, None
    for i, row in tqdm(enumerate(table.itertuples()), total=len(table), desc="Computing μ/σ"):
        with h5py.File(row.Path, "r") as f:
            dset = f["/sen1" if row.Modality.upper() == "SAR" else "/sen2"]
            patch = dset[int(row.Index)]
        X = apply_paper_scaling(patch, row.Modality)
        C = X.shape[-1]
        x = X.reshape(-1, C).astype(np.float64)
        if acc_mu is None:
            acc_mu = np.zeros(C, dtype=np.float64)
            acc_sq = np.zeros(C, dtype=np.float64)
        acc_mu += x.sum(axis=0)
        acc_sq += (x ** 2).sum(axis=0)
        n_pix += x.shape[0]
    mu = acc_mu / n_pix
    sigma = np.sqrt(np.maximum(acc_sq / n_pix - mu ** 2, 1e-12))
    return mu.astype(np.float32), sigma.astype(np.float32), C


# -----------------------------------------------------
# Augmentation
# -----------------------------------------------------

def augment_aligned(img: np.ndarray) -> np.ndarray:
    """Affine warp + reflection + light cutout, identical to MATLAB randomAffine2d."""
    H, W, C = img.shape

    # Random horizontal reflection (50% chance)
    if random.random() < 0.5:
        img = np.ascontiguousarray(np.flip(img, axis=1))

    # Random affine (rotation ±8°, translation ±2px)
    M = cv2.getRotationMatrix2D((W / 2, H / 2), random.uniform(-8, 8), 1.0)
    M[0, 2] += random.uniform(-2, 2)
    M[1, 2] += random.uniform(-2, 2)

    warped = np.stack([
        cv2.warpAffine(
            img[:, :, c],
            M, (W, H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        ) for c in range(C)
    ], axis=-1)

    # Light cutout (same as MATLAB)
    if random.random() < 0.3:
        k = random.randint(3, 5)
        x = random.randint(0, max(0, W - k))
        y = random.randint(0, max(0, H - k))
        warped[y:y + k, x:x + k, :] = 0

    return warped


# -----------------------------------------------------
# Dataset Class
# -----------------------------------------------------

class So2SatDataset(Dataset):
    def __init__(
        self,
        table: pd.DataFrame,
        resize_to=None,
        use_zscore=False,
        mu=None,
        sigma=None,
        use_sar_despeckle=False,
        use_augmentation=False,
        to_gpu=False,
        random_seed=42,
    ):
        super().__init__()
        self.table = table.reset_index(drop=True)
        self.resize_to = resize_to
        self.use_zscore = use_zscore
        self.mu = mu
        self.sigma = sigma
        self.use_sar_despeckle = use_sar_despeckle
        self.use_augmentation = use_augmentation
        self.to_gpu = to_gpu
        random.seed(random_seed)
        np.random.seed(random_seed)

    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx):
        row = self.table.iloc[idx]
        modality = row["Modality"].upper()
        with h5py.File(row["Path"], "r") as f:
            X = f["/sen1" if modality == "SAR" else "/sen2"][int(row["Index"])]

        X = apply_paper_scaling(X, modality)

        # Optional despeckle (SAR only)
        if self.use_sar_despeckle and modality == "SAR":
            for c in range(X.shape[2]):
                X[:, :, c] = cv2.fastNlMeansDenoising(X[:, :, c].astype(np.uint8), None, 12, 7, 21)

        # Optional z-score (after paper scaling)
        if self.use_zscore and self.mu is not None and self.sigma is not None:
            X = (X - self.mu) / (self.sigma + 1e-6)

        # Augment
        if self.use_augmentation:
            X = augment_aligned(X)

        if self.resize_to is not None:
            target_w, target_h = self.resize_to
            if (X.shape[1], X.shape[0]) != (target_w, target_h):
                if target_w > X.shape[1] or target_h > X.shape[0]:
                    interp = cv2.INTER_CUBIC
                else:
                    interp = cv2.INTER_AREA
                X = cv2.resize(X, (target_w, target_h), interpolation=interp)

        # To tensor (C,H,W), float32 in [0,1]
        X = torch.tensor(X, dtype=torch.float32).permute(2, 0, 1)
        if self.to_gpu and torch.cuda.is_available():
            X = X.pin_memory().to("cuda", non_blocking=True)
        label = int(row["Label"]) - 1  # 0-based

        return X, label


# -----------------------------------------------------
# Top-level API (equivalent to DatasetReading.m)
# -----------------------------------------------------

def DatasetReading(cfg: dict):
    """
    Python equivalent of MATLAB DatasetReading.m
    cfg keys:
        trainTable, testTable  -> pandas DataFrames
        resizeTo, useZscore, useSARdespeckle, useAugmentation
        calibrationFrac, randomSeed
    """
    random_seed = cfg.get("randomSeed", 42)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    resize_to = cfg.get("resizeTo")
    use_zscore = cfg.get("useZscore", False)
    use_sar = cfg.get("useSARdespeckle", False)
    use_aug = cfg.get("useAugmentation", False)

    train_table = cfg["trainTable"]
    test_table = cfg["testTable"]

    print("[DatasetReading] Computing per-channel μ/σ on TRAIN…")
    mu, sigma, C = compute_band_stats(train_table)
    print(f"  -> Channels: {C} | μ/σ computed.")

    dsTrain = So2SatDataset(
        train_table,
        resize_to=resize_to,
        use_zscore=use_zscore,
        mu=mu,
        sigma=sigma,
        use_sar_despeckle=use_sar,
        use_augmentation=use_aug,
        random_seed=random_seed,
    )

    dsTest = So2SatDataset(
        test_table,
        resize_to=resize_to,
        use_zscore=use_zscore,
        mu=mu,
        sigma=sigma,
        use_sar_despeckle=use_sar,
        use_augmentation=False,
        random_seed=random_seed,
    )

    labels_all = pd.concat([train_table["Label"], test_table["Label"]], axis=0).astype(int)
    max_label = int(labels_all.max())
    classes = list(range(1, max_label + 1))

    train_counts_series = train_table["Label"].astype(int).value_counts().sort_index()
    class_counts = np.zeros(max_label, dtype=np.int64)
    class_counts[train_counts_series.index.to_numpy() - 1] = train_counts_series.to_numpy()
    inv_freq = 1.0 / np.maximum(class_counts, 1)
    class_weights = (inv_freq / inv_freq.sum()) * len(class_counts)

    info = dict(
        numClasses=max_label,
        classes=classes,
        classes_str=[str(c) for c in classes],
        mu=mu,
        sigma=sigma,
        resizeTo=resize_to,
        classCounts=class_counts,
        classWeights=class_weights.astype(np.float32),
    )

    print(f"[DatasetReading] Done. TRAIN {len(dsTrain)} | TEST {len(dsTest)}.")
    return dsTrain, dsTest, info
