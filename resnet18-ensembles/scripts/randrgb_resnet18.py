import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18
from tqdm import tqdm

from enable_gpu import enable_gpu


# ==============================================================
# Model definition
# ==============================================================


class ResNet18MS(nn.Module):
    """ResNet18 backbone adapted to Sentinel-2 three-band inputs."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(base.children())[:-1])  # drop FC head
        self.fc = nn.Linear(base.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class RandResNet18Ensemble(nn.Module):
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


def train_randrgb_resnet18(cfgT: Dict) -> Dict:
    """
    Python counterpart of MATLAB RandRGB_ResNet18.

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
    ds_train = cfgT["dsTrain"]
    ds_test = cfgT["dsTest"]
    info = cfgT["info"]

    num_members = cfgT.get("numMembers", 10)
    max_epochs = cfgT.get("maxEpochs", 10)
    batch_size = cfgT.get("miniBatchSize", 512)
    learn_rate = cfgT.get("learnRate", 1e-3)
    rng_seed = cfgT.get("rngSeed", 1337)
    num_workers = cfgT.get("numWorkers", 4)
    weight_decay = cfgT.get("weightDecay", 0.0)

    device: Optional[torch.device] = cfgT.get("device")
    if device is None:
        device = enable_gpu(0)
    use_cuda = device.type == "cuda"

    random.seed(rng_seed)
    np.random.seed(rng_seed)
    torch.manual_seed(rng_seed)
    if use_cuda:
        torch.cuda.manual_seed_all(rng_seed)

    sample_tensor, _ = ds_train[0]
    num_bands = sample_tensor.shape[0]
    classes = info.get("classes_str", info["classes"])
    num_classes = len(classes)

    train_loader = _make_loader(ds_train, batch_size, True, num_workers, use_cuda)
    test_loader = _make_loader(ds_test, batch_size, False, num_workers, use_cuda)

    members_models: List[nn.Module] = []
    members_meta: List[MemberSummary] = []
    member_test_scores: List[torch.Tensor] = []

    print(f"[INFO] Training {num_members} ensemble members (RANDRGB / ResNet18).")

    for member_idx in range(num_members):
        print(f"\n[RANDRGB] Member {member_idx + 1:02d}/{num_members}")
        rgb_indices = [3, 2, 1]  # zero-based for B4,B3,B2
        non_rgb = [b for b in range(num_bands) if b not in rgb_indices]
        rand_two = np.random.choice(non_rgb, size=2, replace=False)
        fixed_rgb = np.random.choice(rgb_indices, size=1, replace=False)
        bands = np.sort(np.concatenate([rand_two, fixed_rgb]))
        bands_tensor = torch.tensor(bands, dtype=torch.long, device=device)
        print(f"  Bands (1-based): {bands + 1} (2 random + 1 RGB)")

        model = ResNet18MS(num_classes=num_classes).to(device)
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

            progress = tqdm(train_loader, desc=f"  Epoch {epoch + 1:02d}/{max_epochs}", leave=False)
            for images, labels in progress:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                subset = torch.index_select(images, dim=1, index=bands_tensor)

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
            for images, _ in test_loader:
                images = images.to(device, non_blocking=True)
                subset = torch.index_select(images, dim=1, index=bands_tensor)
                logits = model(subset)
                probs = torch.softmax(logits, dim=1)
                scores_batches.append(probs.cpu())
            member_test_scores.append(torch.cat(scores_batches, dim=0))

        members_models.append(model.to("cpu"))
        members_meta.append(
            MemberSummary(
                bands=bands.tolist(),
                bands_one_based=(bands + 1).tolist(),
                train_loss_history=epoch_losses,
                train_acc_history=epoch_accs,
            )
        )
        if use_cuda:
            torch.cuda.empty_cache()

    # --------------------------------------------------------------
    # Aggregate ensemble metrics
    # --------------------------------------------------------------
    scores_avg = torch.stack(member_test_scores, dim=0).mean(dim=0).numpy()
    y_true_batches = []
    for _, labels in test_loader:
        y_true_batches.append(labels.numpy())
    y_true = np.concatenate(y_true_batches, axis=0) + 1  # convert to one-based
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

    ensemble = RandResNet18Ensemble(members_models, [m.bands for m in members_meta])

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
        mode="RANDRGB",
    )

    print(f"\n[RANDRGB] Test Top-1 = {top1:.4f}")
    return result
