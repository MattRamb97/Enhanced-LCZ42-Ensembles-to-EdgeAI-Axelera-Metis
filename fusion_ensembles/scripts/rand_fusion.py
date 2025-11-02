import os
from typing import Dict, Optional

import h5py
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from fusion_densenet201 import TDAFusionDenseNet201

# ---------------- Dataset ---------------- #
class FusionTDADataset(Dataset):
    """
    Dataset wrapper that keeps MS patches, optional SAR patches, and aligned TDA features.
    """

    def __init__(
        self,
        ms_dataset: Dataset,
        tda_features: np.ndarray,
        ms_band_indices: Optional[np.ndarray] = None,
        sar_dataset: Optional[Dataset] = None,
        sar_band_indices: Optional[np.ndarray] = None,
    ):
        self.ms_dataset = ms_dataset
        self.sar_dataset = sar_dataset
        self.tda_features = torch.as_tensor(np.asarray(tda_features), dtype=torch.float32)

        self.sample_indices = np.asarray(ms_dataset.table["Index"].to_numpy(), dtype=np.int32)

        if sar_dataset is not None:
            sar_indices = np.asarray(sar_dataset.table["Index"].to_numpy(), dtype=np.int32)
            if not np.array_equal(self.sample_indices, sar_indices):
                raise ValueError("MS and SAR dataset indices are misaligned.")

        self.ms_band_indices = (
            torch.as_tensor(ms_band_indices, dtype=torch.long) if ms_band_indices is not None else None
        )
        self.sar_band_indices = (
            torch.as_tensor(sar_band_indices, dtype=torch.long) if sar_band_indices is not None else None
        )

        if self.sample_indices.max() >= len(self.tda_features) or self.sample_indices.min() < 0:
            raise ValueError("Dataset indices fall outside available TDA features.")
        if not np.array_equal(self.sample_indices, np.arange(len(self.sample_indices))):
            print("[WARN] MS dataset index order differs from TDA ordering.")

    def __len__(self) -> int:
        return len(self.ms_dataset)

    def __getitem__(self, idx: int):
        img_ms, label = self.ms_dataset[idx]
        if self.ms_band_indices is not None:
            img_ms = torch.index_select(img_ms, dim=0, index=self.ms_band_indices)

        image_components = [img_ms]

        if self.sar_dataset is not None:
            img_sar, label_sar = self.sar_dataset[idx]
            if label_sar != label:
                raise ValueError("Label mismatch between MS and SAR datasets.")
            if self.sar_band_indices is not None:
                img_sar = torch.index_select(img_sar, dim=0, index=self.sar_band_indices)
            image_components.append(img_sar)

        image = torch.cat(image_components, dim=0)
        tda = self.tda_features[self.sample_indices[idx]]
        return image, tda, label

# ---------------- Training ---------------- #
def train_fusion_member(cfg: Dict):
    device = cfg["device"]
    mode = cfg.get("mode", "RAND").upper()
    ds_train_ms = cfg["dsTrain"]
    ds_test_ms = cfg["dsTest"]
    ds_train_sar = cfg.get("dsTrainSAR")
    ds_test_sar = cfg.get("dsTestSAR")

    sample_ms, _ = ds_train_ms[0]
    num_bands_ms = sample_ms.shape[0]

    rng_seed = cfg.get("rngSeed", 42)
    member_id = cfg.get("memberID", 0)
    rng = np.random.default_rng(rng_seed + member_id)

    ms_band_indices = None
    sar_band_indices = None

    if mode == "RAND":
        ms_band_indices = rng.choice(num_bands_ms, size=3, replace=False)
        print(f"[RAND] Using MS bands (1-based): {(ms_band_indices + 1).tolist()}")
    elif mode == "RANDRGB":
        rgb_indices = np.array([3, 2, 1])  # B4, B3, B2 (zero-based)
        non_rgb = np.array([b for b in range(num_bands_ms) if b not in rgb_indices])
        random_two = rng.choice(non_rgb, size=2, replace=False)
        rgb_choice = rng.choice(rgb_indices, size=1, replace=False)
        ms_band_indices = np.sort(np.concatenate([random_two, rgb_choice]))
        print(f"[RANDRGB] Using bands (1-based): {(ms_band_indices + 1).tolist()}")
    elif mode == "SAR":
        if ds_train_sar is None or ds_test_sar is None:
            raise ValueError("SAR mode requires dsTrainSAR/dsTestSAR in cfg.")
        ms_band_indices = np.sort(rng.choice(num_bands_ms, size=2, replace=False))
        sample_sar, _ = ds_train_sar[0]
        num_bands_sar = sample_sar.shape[0]
        sar_band_indices = np.array([rng.integers(0, num_bands_sar)])
        print(
            "[SAR] Using MS bands (1-based): "
            f"{(ms_band_indices + 1).tolist()}, SAR band (1-based): {(sar_band_indices + 1).tolist()}"
        )
    else:
        raise ValueError(f"Unsupported fusion mode '{mode}'")

    with h5py.File(cfg["tdaTrainPath"], "r") as f:
        tda_key = list(f.keys())[0]
        tda_train = f[tda_key][:]
    with h5py.File(cfg["tdaTestPath"], "r") as f:
        tda_key = list(f.keys())[0]
        tda_test = f[tda_key][:]
    print(f"[{mode}] TDA feature shapes — train: {tda_train.shape}, test: {tda_test.shape}")

    tda_mean = tda_train.mean(axis=0, dtype=np.float64)
    tda_std = tda_train.std(axis=0, dtype=np.float64)
    tda_std[tda_std < 1e-6] = 1.0
    tda_train = ((tda_train - tda_mean) / tda_std).astype(np.float32)
    tda_test = ((tda_test - tda_mean) / tda_std).astype(np.float32)

    train_dataset = FusionTDADataset(
        ds_train_ms,
        tda_train,
        ms_band_indices=ms_band_indices,
        sar_dataset=ds_train_sar if mode == "SAR" else None,
        sar_band_indices=sar_band_indices,
    )
    test_dataset = FusionTDADataset(
        ds_test_ms,
        tda_test,
        ms_band_indices=ms_band_indices,
        sar_dataset=ds_test_sar if mode == "SAR" else None,
        sar_band_indices=sar_band_indices,
    )

    label_train_path = os.path.join(os.path.dirname(cfg["tdaTrainPath"]), "labels.h5")
    if os.path.exists(label_train_path):
        with h5py.File(label_train_path, "r") as lf:
            labels_global = lf["labels"][:].astype(np.int32)
        table_labels = ds_train_ms.table["Label"].to_numpy(dtype=np.int32)
        mapped_labels = labels_global[train_dataset.sample_indices]
        if not np.array_equal(mapped_labels, table_labels):
            print("[WARN] Label mismatch detected between train table and TDA labels file.")

    label_test_path = os.path.join(os.path.dirname(cfg["tdaTestPath"]), "labels_test.h5")
    if os.path.exists(label_test_path):
        with h5py.File(label_test_path, "r") as lf:
            labels_global_test = lf["labels"][:].astype(np.int32)
        table_labels_test = ds_test_ms.table["Label"].to_numpy(dtype=np.int32)
        mapped_labels_test = labels_global_test[test_dataset.sample_indices]
        if not np.array_equal(mapped_labels_test, table_labels_test):
            print("[WARN] Label mismatch detected between test table and TDA labels file.")

    use_cuda = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["miniBatchSize"],
        shuffle=True,
        num_workers=cfg["numWorkers"],
        pin_memory=use_cuda,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["miniBatchSize"],
        shuffle=False,
        num_workers=cfg["numWorkers"],
        pin_memory=use_cuda,
    )

    input_dim = tda_train.shape[1]
    num_classes = cfg["info"]["numClasses"]

    model = TDAFusionDenseNet201(tda_input_dim=input_dim, num_classes=num_classes).to(device)
    class_weights = cfg["info"].get("classWeights")
    weight_tensor = None
    if class_weights is not None:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    label_smoothing = cfg.get("labelSmoothing", 0.0)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=label_smoothing)
    weight_decay = cfg.get("weightDecay", 1e-4)
    optimizer = optim.AdamW(model.parameters(), lr=cfg["learnRate"], weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    history = {"loss": [], "acc": []}

    for epoch in range(cfg["maxEpochs"]):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for imgs, tda, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['maxEpochs']}"):
            imgs, tda, labels = imgs.to(device), tda.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(imgs, tda)
            loss = criterion(preds, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (preds.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        epoch_loss = total_loss / total
        epoch_acc = correct / total
        history["loss"].append(epoch_loss)
        history["acc"].append(epoch_acc)
        scheduler.step(epoch_loss)

        print(f"[Epoch {epoch+1}] Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

    # ---------------- Evaluation ---------------- #
    model.eval()
    preds_all, labels_all = [], []
    probs_all = []
    with torch.no_grad():
        for imgs, tda, labels in test_loader:
            imgs, tda = imgs.to(device), tda.to(device)
            logits = model(imgs, tda)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            preds_all.append((preds + 1).cpu().numpy())   # convert to 1-based for parity with MATLAB
            labels_all.append((labels + 1).cpu().numpy())
            probs_all.append(probs.cpu().numpy())

    y_pred = np.concatenate(preds_all).astype(np.int32)
    y_true = np.concatenate(labels_all).astype(np.int32)
    top1 = float((y_true == y_pred).mean())
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(1, num_classes + 1)).astype(np.int32)
    probs_concat = np.concatenate(probs_all, axis=0).astype(np.float32)

    bands_info = {
        "ms": (ms_band_indices + 1).tolist() if ms_band_indices is not None else None,
        "sar": (sar_band_indices + 1).tolist() if sar_band_indices is not None else None,
    }

    return {
        "model": model,
        "top1": top1,
        "confusion_mat": cm,
        "y_true": y_true,
        "y_pred": y_pred,
        "classes": cfg["info"]["classes"],
        "history": history,
        "probs": probs_concat,
        "bands": bands_info,
        "mode": mode,
    }
