import os
import torch
import numpy as np
import h5py
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from fusion_resnet18 import TDAFusionResNet18
from sklearn.metrics import confusion_matrix

# ---------------- Dataset ---------------- #
class FusionTDADataset(Dataset):
    def __init__(self, image_dataset, tda_features, band_indices=None):
        self.image_dataset = image_dataset
        self.sample_indices = np.asarray(image_dataset.table["Index"].to_numpy(), dtype=np.int32)
        self.tda_features = torch.as_tensor(
            np.asarray(tda_features), dtype=torch.float32
        )
        if band_indices is None:
            self.band_indices = None
        else:
            self.band_indices = torch.as_tensor(band_indices, dtype=torch.long)

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, idx):
        img, label = self.image_dataset[idx]
        if self.band_indices is not None:
            img = torch.index_select(img, dim=0, index=self.band_indices)
        # Each row points to the underlying HDF5 index; use it to align with precomputed TDA features.
        tda = self.tda_features[self.sample_indices[idx]]
        return img, tda, label

# ---------------- Training ---------------- #
def train_fusion_member(cfg):
    device = cfg["device"]
    ds_train = cfg["dsTrain"]
    ds_test = cfg["dsTest"]

    mu = cfg["info"].get("mu")
    if mu is not None:
        num_bands = len(mu)
    else:
        sample, _ = ds_train[0]
        num_bands = sample.shape[0]
    rng_seed = cfg.get("rngSeed", 42)
    member_id = cfg.get("memberID", 0)
    rng = np.random.default_rng(rng_seed + member_id)
    selected_bands = rng.choice(num_bands, size=3, replace=False)
    print(f"[RAND] Using bands (1-based): {(selected_bands + 1).tolist()}")

    # Load TDA features
    with h5py.File(cfg["tdaTrainPath"], "r") as f:
        tda_key = list(f.keys())[0]
        tda_train = f[tda_key][:]
    with h5py.File(cfg["tdaTestPath"], "r") as f:
        tda_key = list(f.keys())[0]
        tda_test = f[tda_key][:]
    print(f"[RAND] TDA feature shapes — train: {tda_train.shape}, test: {tda_test.shape}")

    tda_mean = tda_train.mean(axis=0, dtype=np.float64)
    tda_std = tda_train.std(axis=0, dtype=np.float64)
    tda_std[tda_std < 1e-6] = 1.0
    tda_train = ((tda_train - tda_mean) / tda_std).astype(np.float32)
    tda_test = ((tda_test - tda_mean) / tda_std).astype(np.float32)

    train_dataset = FusionTDADataset(ds_train, tda_train, band_indices=selected_bands)
    test_dataset = FusionTDADataset(ds_test, tda_test, band_indices=selected_bands)
    if train_dataset.sample_indices.max() >= len(tda_train) or train_dataset.sample_indices.min() < 0:
        raise ValueError("Training table indices fall outside available TDA train features.")
    if test_dataset.sample_indices.max() >= len(tda_test) or test_dataset.sample_indices.min() < 0:
        raise ValueError("Testing table indices fall outside available TDA test features.")
    if not np.array_equal(train_dataset.sample_indices, np.arange(len(train_dataset))):
        print("[WARN] Train table index order differs from TDA ordering.")
    if not np.array_equal(test_dataset.sample_indices, np.arange(len(test_dataset))):
        print("[WARN] Test table index order differs from TDA ordering.")

    label_train_path = os.path.join(os.path.dirname(cfg["tdaTrainPath"]), "labels.h5")
    if os.path.exists(label_train_path):
        with h5py.File(label_train_path, "r") as lf:
            labels_global = lf["labels"][:].astype(np.int32)
        table_labels = ds_train.table["Label"].to_numpy(dtype=np.int32)
        mapped_labels = labels_global[train_dataset.sample_indices]
        if not np.array_equal(mapped_labels, table_labels):
            print("[WARN] Label mismatch detected between train table and TDA labels file.")

    label_test_path = os.path.join(os.path.dirname(cfg["tdaTestPath"]), "labels_test.h5")
    if os.path.exists(label_test_path):
        with h5py.File(label_test_path, "r") as lf:
            labels_global_test = lf["labels"][:].astype(np.int32)
        table_labels_test = ds_test.table["Label"].to_numpy(dtype=np.int32)
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

    model = TDAFusionResNet18(tda_input_dim=input_dim, num_classes=num_classes).to(device)
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
    with torch.no_grad():
        for imgs, tda, labels in test_loader:
            imgs, tda = imgs.to(device), tda.to(device)
            logits = model(imgs, tda)
            preds = torch.argmax(logits, dim=1)
            preds_all.append((preds + 1).cpu().numpy())   # convert to 1-based for parity with MATLAB
            labels_all.append((labels + 1).cpu().numpy())

    y_pred = np.concatenate(preds_all).astype(np.int32)
    y_true = np.concatenate(labels_all).astype(np.int32)
    top1 = float((y_true == y_pred).mean())
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(1, num_classes + 1)).astype(np.int32)

    return {
        "model": model,
        "top1": top1,
        "confusion_mat": cm,
        "y_true": y_true,
        "y_pred": y_pred,
        "classes": cfg["info"]["classes"],
        "history": history,
        "bands": selected_bands.tolist(),
        "bands_one_based": (selected_bands + 1).tolist(),
    }
