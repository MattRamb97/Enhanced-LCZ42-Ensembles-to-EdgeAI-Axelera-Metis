from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision.models import densenet201, DenseNet201_Weights
from tqdm import tqdm

from enable_gpu import enable_gpu


# ==============================================================
# Model definition
# ==============================================================


class DenseNet201MS(nn.Module):
    """DenseNet201 backbone adapted to Sentinel-2 three-band inputs."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        base = densenet201(weights=DenseNet201_Weights.DEFAULT)
        self.features = base.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base.classifier.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# ==============================================================
# Band Selection Dataset Wrapper
# ==============================================================


class BandSelectionDataset(Dataset):
    """Wrapper that applies band selection to MS/SAR data."""

    def __init__(
        self,
        ms_dataset: Dataset,
        ms_band_indices: Optional[np.ndarray] = None,
        sar_dataset: Optional[Dataset] = None,
        sar_band_indices: Optional[np.ndarray] = None,
    ):
        self.ms_dataset = ms_dataset
        self.sar_dataset = sar_dataset
        self.ms_band_indices = (
            torch.as_tensor(ms_band_indices, dtype=torch.long) if ms_band_indices is not None else None
        )
        self.sar_band_indices = (
            torch.as_tensor(sar_band_indices, dtype=torch.long) if sar_band_indices is not None else None
        )

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
        return image, label


# ==============================================================
# Training routine
# ==============================================================

def _make_loader(dataset, batch_size, shuffle, num_workers, use_cuda):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=use_cuda,
        drop_last=False,
    )


def train_rand_fusion(cfgT: Dict) -> Dict:
    """
    Training routine for a single SR-based DenseNet201 member (no TDA).

    Trains one member with band selection based on mode (RAND, RANDRGB, or SAR).
    Model is standard DenseNet201 (no TDA fusion).

    cfgT keys:
        dsTrain, dsTest, info
        maxEpochs (default 12)
        miniBatchSize (default 512)
        learnRate (default 1e-3)
        rngSeed (default 1337)
        numWorkers (default 6)
        device (optional torch.device)
        mode: "RAND", "RANDRGB", or "SAR"
        memberID: member index for seeding
        dsTrainSAR, dsTestSAR: optional SAR datasets for SAR mode
    Returns a dictionary with the member's model and metrics.
    """
    device: Optional[torch.device] = cfgT.get("device")
    if device is None:
        device = enable_gpu(0)
    use_cuda = device.type == "cuda"

    ds_train_ms = cfgT["dsTrain"]
    ds_test_ms = cfgT["dsTest"]
    ds_train_sar = cfgT.get("dsTrainSAR")
    ds_test_sar = cfgT.get("dsTestSAR")
    info = cfgT["info"]

    mode = cfgT.get("mode", "RAND").upper()
    max_epochs = cfgT.get("maxEpochs", 12)
    batch_size = cfgT.get("miniBatchSize", 512)
    learn_rate = cfgT.get("learnRate", 1e-3)
    rng_seed = cfgT.get("rngSeed", 1337)
    num_workers = cfgT.get("numWorkers", 6)
    weight_decay = cfgT.get("weightDecay", cfgT.get("weight_decay", 1e-4))
    member_id = cfgT.get("memberID", 0)

    print(f"[INFO] Training setup → mode={mode} batch_size={batch_size} num_workers={num_workers}")

    # Seed for reproducibility
    rng = np.random.default_rng(rng_seed + member_id)
    torch.manual_seed(rng_seed + member_id)
    if use_cuda:
        torch.cuda.manual_seed_all(rng_seed + member_id)

    # Get number of bands
    sample_ms, _ = ds_train_ms[0]
    num_bands_ms = sample_ms.shape[0]
    classes = info.get("classes_str", info["classes"])
    num_classes = len(classes)

    ms_band_indices = None
    sar_band_indices = None

    # Band selection logic based on mode
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

    # Create datasets with band selection
    train_dataset = BandSelectionDataset(
        ds_train_ms,
        ms_band_indices=ms_band_indices,
        sar_dataset=ds_train_sar if mode == "SAR" else None,
        sar_band_indices=sar_band_indices,
    )
    test_dataset = BandSelectionDataset(
        ds_test_ms,
        ms_band_indices=ms_band_indices,
        sar_dataset=ds_test_sar if mode == "SAR" else None,
        sar_band_indices=sar_band_indices,
    )

    train_loader = _make_loader(train_dataset, batch_size, True, num_workers, use_cuda)
    test_loader = _make_loader(test_dataset, batch_size, False, num_workers, use_cuda)

    # Model
    model = DenseNet201MS(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=learn_rate,
        momentum=0.9,
        weight_decay=weight_decay,
    )

    # Training
    epoch_losses: list = []
    epoch_accs: list = []

    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        samples = 0

        progress = tqdm(train_loader, desc=f"  Epoch {epoch + 1:02d}/{max_epochs}", leave=False)
        for images, labels in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            samples += labels.size(0)

            progress.set_postfix(loss=running_loss / max(samples, 1), acc=correct / max(samples, 1))

        epoch_loss = running_loss / samples
        epoch_acc = correct / samples
        epoch_losses.append(epoch_loss)
        epoch_accs.append(epoch_acc)

        print(f"  Epoch {epoch + 1:02d}: loss={epoch_loss:.4f} acc={epoch_acc:.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        scores_batches = []
        for images, _ in test_loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            scores_batches.append(probs.cpu())
        test_probs = torch.cat(scores_batches, dim=0).numpy()

    y_true_batches = []
    for _, labels in test_loader:
        y_true_batches.append(labels.numpy())
    y_true = np.concatenate(y_true_batches, axis=0) + 1  # convert to one-based
    y_pred = test_probs.argmax(axis=1) + 1

    top1 = float((y_true == y_pred).mean())
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(1, num_classes + 1))

    history = dict(
        loss_per_member=np.array([epoch_losses], dtype=np.float32),
        acc_per_member=np.array([epoch_accs], dtype=np.float32),
        loss_mean=np.array(epoch_losses, dtype=np.float32),
        acc_mean=np.array(epoch_accs, dtype=np.float32),
    )

    bands_info = {
        "ms": (ms_band_indices + 1).tolist() if ms_band_indices is not None else None,
        "sar": (sar_band_indices + 1).tolist() if sar_band_indices is not None else None,
    }

    result = dict(
        model=model.to("cpu"),
        probs=test_probs.astype(np.float32),
        y_true=y_true.astype(np.int32),
        y_pred=y_pred.astype(np.int32),
        top1=top1,
        confusion_mat=cm.astype(np.int32),
        classes=list(classes),
        history=history,
        bands=bands_info,
        rng_seed=rng_seed,
        mode=mode,
    )

    print(f"\n[{mode}] Test Top-1 = {top1:.4f}")
    return result
