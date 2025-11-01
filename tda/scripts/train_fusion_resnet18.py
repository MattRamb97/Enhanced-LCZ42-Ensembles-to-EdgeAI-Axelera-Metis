import argparse
import h5py
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn, optim
from models.fusion_resnet18 import TDAFusionResNet18
from tqdm import tqdm

# ----------------- Config -----------------
H5_PATH = "../data/lcz42/training.h5"
LABELS_PATH = "labels.npy"
NUM_CLASSES = 17
EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 3e-4
DROPOUT = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# ----------------- Custom Dataset -----------------
class FusionDataset(Dataset):
    def __init__(self, h5_path, tda_features, labels, ensemble_type="Rand"):
        self.tda_features = torch.tensor(tda_features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)  # 1–17 labels
        self.ensemble_type = ensemble_type
        self.h5_path = h5_path

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, "r") as h5:
            tda = self.tda_features[idx]
            label = self.labels[idx]

            if self.ensemble_type == "Rand":
                patch = h5["sen2"][idx]
                selected = np.random.choice(10, size=3, replace=False)
                image = patch[..., selected]

            elif self.ensemble_type == "RandRGB":
                patch = h5["sen2"][idx]
                patch = (patch / (2.8 / 255.0)).clip(0, 255).astype(np.float32)
                rgb = np.random.choice([3, 2, 1])  # B4, B3, B2
                others = np.random.choice([i for i in range(10) if i != rgb], size=2, replace=False)
                selected = list(others) + [rgb]
                image = patch[..., selected]

            elif self.ensemble_type == "ensembleSAR":
                ms = h5["sen2"][idx]
                ms = (ms / (2.8 / 255.0)).clip(0, 255).astype(np.float32)
                sar = h5["sen1"][idx]
                sar = (np.clip(sar, -0.5, 0.5) + 0.5) * 255.0
                sar = sar.astype(np.float32)
                ms_selected = np.random.choice(10, size=2, replace=False)
                sar_selected = np.random.choice(8)
                ms_channels = ms[..., ms_selected]
                sar_channel = sar[..., sar_selected][..., np.newaxis]
                image = np.concatenate([ms_channels, sar_channel], axis=-1)

            else:
                raise ValueError("Invalid ensemble type")

            # (H, W, C) → (C, H, W) → resize → tensor
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0 # (H, W, C) → (C, H, W)
            image = torch.nn.functional.interpolate(
                image.unsqueeze(0),  # add batch dim → (1, C, H, W)
                size=(224, 224),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)  # remove batch dim → (C, 224, 224)
            return image, tda, label

# ----------------- Main -----------------
def main(ensemble_type):
    print(f"[INFO] Training fusion model for ensemble type: {ensemble_type}")

    # Pick correct TDA file
    tda_path = "tda_MS_features.npy" if ensemble_type in ["Rand", "RandRGB"] else "tda_SAR_features.npy"
    tda_features = np.load(tda_path).astype(np.float32)
    input_dim = tda_features.shape[1]

    labels = np.load(LABELS_PATH)

    # Datasets & loaders
    dataset = FusionDataset(H5_PATH, tda_features, labels, ensemble_type)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=2)

    # Model
    model = TDAFusionResNet18(tda_input_dim=input_dim, num_classes=NUM_CLASSES, dropout_rate=DROPOUT).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for imgs, tda, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            imgs, tda, targets = imgs.to(DEVICE), tda.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            preds = model(imgs, tda)
            loss = criterion(preds, targets - 1)  # shift to 0–16
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            correct += (preds.argmax(dim=1) == (targets - 1)).sum().item()
            total += imgs.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for imgs, tda, targets in val_loader:
                imgs, tda, targets = imgs.to(DEVICE), tda.to(DEVICE), targets.to(DEVICE)
                preds = model(imgs, tda)
                loss = criterion(preds, targets - 1)  # shift to 0–16

                val_loss += loss.item() * imgs.size(0)
                val_correct += (preds.argmax(dim=1) == (targets - 1)).sum().item()
                val_total += imgs.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total
        scheduler.step(val_loss)

        print(f"[{epoch+1}/{EPOCHS}] Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

    # Save
    save_path = f"fusion_resnet18_{ensemble_type.lower()}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Saved model -> {save_path}")

# ----------------- Entry -----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble_type", choices=["Rand", "RandRGB", "ensembleSAR"], required=True)
    args = parser.parse_args()
    main(args.ensemble_type)