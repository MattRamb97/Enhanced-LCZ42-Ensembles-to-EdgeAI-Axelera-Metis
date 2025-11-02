import random
from dataclasses import dataclass
from typing import Dict, List, Optional

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


class RandDenseNet201Ensemble(nn.Module):
    """Torch module that averages member probabilities at inference time."""

    def __init__(self, models: List[nn.Module], bands: List[List[int]]):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.register_buffer("bands_matrix", torch.tensor(bands, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, H, W) with C >= 3
        probs_sum = None
        for model, band_idx in zip(self.models, self.bands_matrix):
            subset = torch.index_select(x, dim=1, index=band_idx)
            logits = model(subset)
            probs = torch.softmax(logits, dim=1)
            probs_sum = probs if probs_sum is None else probs_sum + probs
        return probs_sum / len(self.models)

    @property
    def bands(self) -> List[List[int]]:
        return [band_idx.cpu().tolist() for band_idx in self.bands_matrix]


@dataclass
class MemberSummary:
    bands: List[int]             # zero-based indices
    bands_one_based: List[int]   # human readable
    train_loss_history: List[float]
    train_acc_history: List[float]

    @property
    def final_loss(self) -> float:
        return self.train_loss_history[-1]

    @property
    def final_acc(self) -> float:
        return self.train_acc_history[-1]


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


def train_randsar_densenet201(cfgT: Dict) -> Dict:
    """
    Python counterpart of MATLAB randSAR_DenseNet201.

    cfgT keys (same semantics as MATLAB):
        dsTrain, dsTest, info
        numMembers (default 10)
        maxEpochs (default 10)
        miniBatchSize (default 512)
        learnRate (default 1e-3)
        rngSeed (default 1337)
        numWorkers (default 4)
        device (optional torch.device)
    Returns a dictionary with the full ensemble state and metrics.
    """
    ds_train_ms = cfgT["dsTrain"]
    ds_test_ms  = cfgT["dsTest"]
    ds_train_sar = cfgT["dsTrainSAR"]
    ds_test_sar  = cfgT["dsTestSAR"]
    info = cfgT["info"]

    num_members = cfgT.get("numMembers", 10)
    max_epochs = cfgT.get("maxEpochs", 12)
    batch_size = cfgT.get("miniBatchSize", 512)
    learn_rate = cfgT.get("learnRate", 1e-3)
    rng_seed = cfgT.get("rngSeed", 1337)
    num_workers = cfgT.get("numWorkers", 6)
    weight_decay = cfgT.get("weightDecay", cfgT.get("weight_decay", 1e-4))

    device: Optional[torch.device] = cfgT.get("device")
    if device is None:
        device = enable_gpu(0)
    use_cuda = device.type == "cuda"

    random.seed(rng_seed)
    np.random.seed(rng_seed)
    torch.manual_seed(rng_seed)
    if use_cuda:
        torch.cuda.manual_seed_all(rng_seed)

    sample_ms, _ = ds_train_ms[0]
    sample_sar, _ = ds_train_sar[0]
    num_bands_ms = sample_ms.shape[0]   # e.g. 10
    num_bands_sar = sample_sar.shape[0] # e.g. 8
    classes = info.get("classes_str", info["classes"])
    num_classes = len(classes)

    class _PairedDataset(Dataset):
        """Zip MS and SAR datasets so batches stay aligned."""

        def __init__(self, ds_ms, ds_sar):
            if len(ds_ms) != len(ds_sar):
                raise ValueError(
                    "MS and SAR datasets must share the same length "
                    f"(got {len(ds_ms)} vs {len(ds_sar)})."
                )
            self.ds_ms = ds_ms
            self.ds_sar = ds_sar

        def __len__(self) -> int:
            return len(self.ds_ms)

        def __getitem__(self, idx: int):
            ms_sample, ms_label = self.ds_ms[idx]
            sar_sample, sar_label = self.ds_sar[idx]
            if ms_label != sar_label:
                raise ValueError(
                    f"Label mismatch between modalities at index {idx}: "
                    f"MS={ms_label}, SAR={sar_label}."
                )
            return ms_sample, sar_sample, ms_label

    train_dataset = _PairedDataset(ds_train_ms, ds_train_sar)
    test_dataset = _PairedDataset(ds_test_ms, ds_test_sar)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
        drop_last=False,
    )

    members_models: List[nn.Module] = []
    members_meta: List[MemberSummary] = []
    member_test_scores: List[torch.Tensor] = []

    print(f"[INFO] Training {num_members} ensemble members (SAR / DenseNet201).")

    for member_idx in range(num_members):
        print(f"\n[SAR] Member {member_idx + 1:02d}/{num_members}")
        bands_ms = np.random.choice(num_bands_ms, size=2, replace=False)
        bands_sar = np.random.choice(num_bands_sar, size=1, replace=False)
        bands_ms_t  = torch.tensor(bands_ms,  dtype=torch.long, device=device)
        bands_sar_t = torch.tensor(bands_sar, dtype=torch.long, device=device)
        print(f"  MS bands (1-based): {bands_ms + 1}, SAR band (1-based): {bands_sar + 1}")

        model = DenseNet201MS(num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            model.parameters(),
            lr=learn_rate,
            momentum=0.9,
            weight_decay=weight_decay,
        )

        epoch_losses: List[float] = []
        epoch_accs: List[float] = []

        for epoch in range(max_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            samples = 0

            progress = tqdm(
                train_loader,
                desc=f"  Epoch {epoch + 1:02d}/{max_epochs}",
                leave=False,
            )
            for images_ms, images_sar, labels in progress:

                images_ms = images_ms.to(device, non_blocking=True)
                images_sar = images_sar.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                subset_ms = torch.index_select(images_ms, dim=1, index=bands_ms_t)
                subset_sar = torch.index_select(images_sar, dim=1, index=bands_sar_t)
                # (B,2,H,W) + (B,1,H,W) -> (B,3,H,W)
                subset = torch.cat([subset_ms, subset_sar], dim=1)

                optimizer.zero_grad()
                logits = model(subset)
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

        model.eval()
        with torch.no_grad():
            scores_batches = []
            for images_ms, images_sar, _ in tqdm(
                test_loader, leave=False, desc="  Eval"
            ):

                images_ms = images_ms.to(device, non_blocking=True)
                images_sar = images_sar.to(device, non_blocking=True)

                subset_ms  = torch.index_select(images_ms,  dim=1, index=bands_ms_t)
                subset_sar = torch.index_select(images_sar, dim=1, index=bands_sar_t)
                subset = torch.cat([subset_ms, subset_sar], dim=1)

                logits = model(subset)
                probs = torch.softmax(logits, dim=1)
                scores_batches.append(probs.cpu())
            member_test_scores.append(torch.cat(scores_batches, dim=0))

        members_models.append(model.to("cpu"))
        meta = MemberSummary(
            bands=[*bands_ms.tolist(), *[b + num_bands_ms for b in bands_sar.tolist()]],  # combined 3 bands
            bands_one_based=[*(bands_ms + 1).tolist(), *(bands_sar + num_bands_ms + 1).tolist()],
            train_loss_history=epoch_losses,
            train_acc_history=epoch_accs,
        )
        meta.ms_bands = bands_ms.tolist()
        meta.sar_bands = bands_sar.tolist()
        meta.ms_bands_one_based = (bands_ms + 1).tolist()
        meta.sar_bands_one_based = (bands_sar + 1).tolist()
        meta.sar_components = len(bands_sar)
        meta.bands_one_based = {
            "ms": meta.ms_bands_one_based,
            "sar": meta.sar_bands_one_based,
        }
        members_meta.append(meta)
        if use_cuda:
            torch.cuda.empty_cache()

    # --------------------------------------------------------------
    # Aggregate ensemble metrics
    # --------------------------------------------------------------
    scores_avg = torch.stack(member_test_scores, dim=0).mean(dim=0).numpy()
    y_true_batches = []
    for _, _, labels in test_loader:
        y_true_batches.append(labels.cpu().numpy())
    y_true = np.concatenate(y_true_batches, axis=0) + 1 # convert to one-based
    y_pred = scores_avg.argmax(axis=1) + 1

    top1 = float((y_true == y_pred).mean())
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(1, num_classes + 1))

    loss_matrix = np.array([m.train_loss_history for m in members_meta], dtype=np.float32)
    acc_matrix = np.array([m.train_acc_history for m in members_meta], dtype=np.float32)
    history = dict(
        loss_per_member=loss_matrix,
        acc_per_member=acc_matrix,
        loss_mean=loss_matrix.mean(axis=0),
        acc_mean=acc_matrix.mean(axis=0),
    )

    ensemble = RandDenseNet201Ensemble(members_models, [m.bands for m in members_meta])

    result = dict(
        ensemble=ensemble,
        members=members_meta,
        scores_avg=scores_avg.astype(np.float32),
        y_true=y_true.astype(np.int32),
        y_pred=y_pred.astype(np.int32),
        test_top1=top1,
        confusion_mat=cm.astype(np.int32),
        classes=list(classes),
        history=history,
        rng_seed=rng_seed,
        num_members=num_members,
        mode="SAR",
    )

    print(f"\n[SAR] Test Top-1 = {top1:.4f}")
    return result
