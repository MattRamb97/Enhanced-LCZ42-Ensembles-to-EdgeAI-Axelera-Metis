import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.models import ResNet18_Weights, resnet18
from tqdm import tqdm

from enable_gpu import enable_gpu


SAR_POLARIZATION_BANDS: Tuple[int, int] = (4, 5)  # VH and VV intensities (0-based)


# ==============================================================
# Model definition
# ==============================================================


class ResNet18MS(nn.Module):
    """ResNet18 backbone adapted to Sentinel-2 + SAR inputs with 4→3 adapter."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.adapter = nn.Conv2d(4, 3, kernel_size=1, bias=True)
        with torch.no_grad():
            self.adapter.weight.zero_()
            self.adapter.bias.zero_()
            self.adapter.weight[0, 0, 0, 0] = 1.0  # MS band 1 passthrough
            self.adapter.weight[1, 1, 0, 0] = 1.0  # MS band 2 passthrough
            self.adapter.weight[2, 2, 0, 0] = 1.0  # SAR CP1 passthrough
        self.features = nn.Sequential(*list(base.children())[:-1])  # drop FC head
        self.fc = nn.Linear(base.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.adapter(x)
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class RandResNet18Ensemble(nn.Module):
    """Torch module that averages member probabilities at inference time."""

    def __init__(self, models: List[nn.Module], member_summaries: List["MemberSummary"],
                 num_bands_ms: int):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.num_bands_ms = num_bands_ms
        self.member_configs = []
        for summary in member_summaries:
            ms_idx = torch.tensor(summary.ms_bands, dtype=torch.long)
            sar_idx = torch.tensor(
                [self.num_bands_ms + idx for idx in summary.sar_bands], dtype=torch.long
            )
            sar_mean = torch.tensor(summary.sar_pca_mean, dtype=torch.float32)
            sar_weights = torch.tensor(summary.sar_pca_weights, dtype=torch.float32)
            self.member_configs.append(
                dict(ms_idx=ms_idx, sar_idx=sar_idx, sar_mean=sar_mean, sar_weights=sar_weights)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, H, W) with C >= num_bands_ms + num_sar
        probs_sum = None
        for model, cfg in zip(self.models, self.member_configs):
            ms_idx = cfg["ms_idx"].to(x.device)
            sar_idx = cfg["sar_idx"].to(x.device)
            sar_mean = cfg["sar_mean"].to(x.device).view(1, -1, 1, 1)
            sar_weights = cfg["sar_weights"].to(x.device)

            subset_ms = torch.index_select(x, dim=1, index=ms_idx)
            sar_channels = torch.index_select(x, dim=1, index=sar_idx)
            sar_centered = sar_channels - sar_mean
            weights_expanded = sar_weights.unsqueeze(-1).unsqueeze(-1)
            sar_projected = torch.sum(
                sar_centered.unsqueeze(1) * weights_expanded.unsqueeze(0),
                dim=2,
            )
            subset = torch.cat([subset_ms, sar_projected], dim=1)

            logits = model(subset)
            probs = torch.softmax(logits, dim=1)
            probs_sum = probs if probs_sum is None else probs_sum + probs
        return probs_sum / len(self.models)

    @property
    def bands(self) -> List[Dict[str, List[int]]]:
        result = []
        for cfg in self.member_configs:
            sar_raw = [idx - self.num_bands_ms for idx in cfg["sar_idx"].tolist()]
            result.append(
                dict(
                    ms=cfg["ms_idx"].tolist(),
                    sar=sar_raw,
                    sar_mean=cfg["sar_mean"].tolist(),
                    sar_weights=cfg["sar_weights"].cpu().tolist(),
                    sar_components=cfg["sar_weights"].shape[0],
                )
            )
        return result


@dataclass
class MemberSummary:
    ms_bands: List[int]
    sar_bands: List[int]
    sar_pca_mean: List[float]
    sar_pca_weights: List[List[float]]
    train_loss_history: List[float]
    train_acc_history: List[float]

    @property
    def final_loss(self) -> float:
        return self.train_loss_history[-1]

    @property
    def final_acc(self) -> float:
        return self.train_acc_history[-1]

    @property
    def bands_one_based(self) -> Dict[str, List[int]]:
        return dict(
            ms=[b + 1 for b in self.ms_bands],
            sar=[b + 1 for b in self.sar_bands],
        )

    @property
    def sar_components(self) -> int:
        if isinstance(self.sar_pca_weights, list):
            return len(self.sar_pca_weights)
        return int(self.sar_pca_weights.shape[0])


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


def train_randsar_resnet18(cfgT: Dict) -> Dict:
    """
    Python counterpart of MATLAB randSAR_ResNet18.

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

    sample_ms, _ = ds_train_ms[0]
    sample_sar, _ = ds_train_sar[0]
    num_bands_ms = sample_ms.shape[0]   # e.g. 10
    num_bands_sar = sample_sar.shape[0] # e.g. 8
    classes = info.get("classes_str", info["classes"])
    num_classes = len(classes)

    if any(idx >= num_bands_sar or idx < 0 for idx in SAR_POLARIZATION_BANDS):
        raise ValueError(
            f"Sentinel-1 dataset must include VV/VH intensity bands at indices {SAR_POLARIZATION_BANDS}; "
            f"found only {num_bands_sar} channels."
        )

    sar_pca_cache: Dict[Tuple[int, ...], Tuple[torch.Tensor, torch.Tensor]] = {}

    def get_sar_pca_stats(band_tuple: Tuple[int, ...]) -> Tuple[torch.Tensor, torch.Tensor]:
        if band_tuple in sar_pca_cache:
            return sar_pca_cache[band_tuple]

        k = len(band_tuple)
        with torch.no_grad():
            sum_vec = torch.zeros(k, dtype=torch.float64)
            sum_outer = torch.zeros((k, k), dtype=torch.float64)
            count = 0
            for idx_sample in range(len(ds_train_sar)):
                sar_sample, _ = ds_train_sar[idx_sample]
                if not isinstance(sar_sample, torch.Tensor):
                    sar_sample = torch.as_tensor(sar_sample)
                subset = sar_sample[list(band_tuple)].to(torch.float64)
                flat = subset.reshape(k, -1)
                sum_vec += flat.sum(dim=1)
                sum_outer += flat @ flat.t()
                count += flat.shape[1]
        if count == 0:
            raise RuntimeError("Unable to compute SAR PCA statistics: empty dataset.")

        mean = sum_vec / count
        covariance = sum_outer / count - torch.outer(mean, mean)
        eigvals, eigvecs = torch.linalg.eigh(covariance)
        order = torch.argsort(eigvals, descending=True)
        eigvecs = eigvecs[:, order]
        num_components = min(k, 2)
        components = []
        for idx_comp in range(num_components):
            vec = eigvecs[:, idx_comp]
            norm = vec.norm()
            if norm == 0:
                vec = torch.ones_like(vec)
                norm = vec.norm()
            vec = vec / norm
            max_idx = torch.argmax(vec.abs())
            sign = torch.sign(vec[max_idx])
            if sign == 0:
                sign = torch.tensor(1.0, dtype=vec.dtype)
            components.append(vec * sign)
        principal_matrix = torch.stack(components, dim=0)

        mean_f32 = mean.to(torch.float32)
        principal_f32 = principal_matrix.to(torch.float32)
        sar_pca_cache[band_tuple] = (mean_f32, principal_f32)
        return sar_pca_cache[band_tuple]

    sar_key = tuple(int(idx) for idx in SAR_POLARIZATION_BANDS)
    sar_mean_cpu, sar_weights_cpu = get_sar_pca_stats(sar_key)

    # ---- TRAIN loaders (shared random order) ----
    g = torch.Generator()
    g.manual_seed(rng_seed)  # so both samplers get the same permutation
    idx = torch.randperm(len(ds_train_ms), generator=g).tolist()

    sampler_train = SubsetRandomSampler(idx)
    train_loader_ms  = DataLoader(ds_train_ms,  batch_size=batch_size, sampler=sampler_train,
                                num_workers=num_workers, pin_memory=use_cuda, drop_last=False)
    # reuse the SAME sampler object to guarantee identical order
    train_loader_sar = DataLoader(ds_train_sar, batch_size=batch_size, sampler=sampler_train,
                                num_workers=num_workers, pin_memory=use_cuda, drop_last=False)

    # ---- TEST loaders (deterministic, aligned) ----
    test_loader_ms  = DataLoader(ds_test_ms,  batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=use_cuda, drop_last=False)
    test_loader_sar = DataLoader(ds_test_sar, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=use_cuda, drop_last=False)

    members_models: List[nn.Module] = []
    members_meta: List[MemberSummary] = []
    member_test_scores: List[torch.Tensor] = []

    print(f"[INFO] Training {num_members} ensemble members (SAR / ResNet18).")

    for member_idx in range(num_members):
        print(f"\n[SAR] Member {member_idx + 1:02d}/{num_members}")
        bands_ms = np.random.choice(num_bands_ms, size=2, replace=False)
        bands_sar = np.array(SAR_POLARIZATION_BANDS, dtype=np.int64)
        num_components = sar_weights_cpu.shape[0]

        bands_ms_t = torch.tensor(bands_ms, dtype=torch.long, device=device)
        bands_sar_t = torch.tensor(bands_sar, dtype=torch.long, device=device)
        sar_mean_t = sar_mean_cpu.to(device).view(1, -1, 1, 1)
        sar_weights_t = sar_weights_cpu.to(device)

        print(f"  MS bands (1-based): {bands_ms + 1}, SAR bands (1-based): {bands_sar + 1} -> PCA({num_components})")

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

            total_batches = min(len(train_loader_ms), len(train_loader_sar))
            progress = tqdm(zip(train_loader_ms, train_loader_sar),
                            total=total_batches,
                            desc=f"  Epoch {epoch + 1:02d}/{max_epochs}",
                            leave=False)
            for ms_batch, sar_batch in progress:
                images_ms, labels = ms_batch
                images_sar, _ = sar_batch  # labels are identical due to shared sampler

                images_ms = images_ms.to(device, non_blocking=True)
                images_sar = images_sar.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                subset_ms = torch.index_select(images_ms, dim=1, index=bands_ms_t)
                sar_selected = torch.index_select(images_sar, dim=1, index=bands_sar_t)
                sar_centered = sar_selected - sar_mean_t
                weights_expanded = sar_weights_t.unsqueeze(-1).unsqueeze(-1)
                subset_sar = torch.sum(
                    sar_centered.unsqueeze(1) * weights_expanded.unsqueeze(0),
                    dim=2,
                )
                # (B,2,H,W) + (B,2,H,W) -> (B,4,H,W)
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
            total_batches = min(len(test_loader_ms), len(test_loader_sar))
            for (ms_batch, sar_batch) in tqdm(zip(test_loader_ms, test_loader_sar),
                                             total=total_batches, leave=False, desc="  Eval"):
                images_ms, _ = ms_batch
                images_sar, _ = sar_batch

                images_ms = images_ms.to(device, non_blocking=True)
                images_sar = images_sar.to(device, non_blocking=True)

                subset_ms = torch.index_select(images_ms, dim=1, index=bands_ms_t)
                sar_selected = torch.index_select(images_sar, dim=1, index=bands_sar_t)
                sar_centered = sar_selected - sar_mean_t
                weights_expanded = sar_weights_t.unsqueeze(-1).unsqueeze(-1)
                subset_sar = torch.sum(
                    sar_centered.unsqueeze(1) * weights_expanded.unsqueeze(0),
                    dim=2,
                )
                subset = torch.cat([subset_ms, subset_sar], dim=1)

                logits = model(subset)
                probs = torch.softmax(logits, dim=1)
                scores_batches.append(probs.cpu())
            member_test_scores.append(torch.cat(scores_batches, dim=0))

        members_models.append(model.to("cpu"))
        members_meta.append(
            MemberSummary(
                ms_bands=bands_ms.tolist(),
                sar_bands=bands_sar.tolist(),
                sar_pca_mean=sar_mean_cpu.tolist(),
                sar_pca_weights=sar_weights_cpu.tolist(),
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
    for _, labels in test_loader_ms:
        y_true_batches.append(labels.numpy())
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

    ensemble = RandResNet18Ensemble(members_models, members_meta, num_bands_ms)

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
