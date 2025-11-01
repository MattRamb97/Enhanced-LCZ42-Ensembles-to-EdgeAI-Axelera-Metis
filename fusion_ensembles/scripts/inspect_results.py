import h5py
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

# ---------- Configuration ----------
RESULTS_DIR = Path("../results")
RESULTS_PATTERNS = [
    ("rand", "fusion_resnet18_rand_*.h5"),
    ("randrgb", "fusion_resnet18_randrgb_*.h5"),
    ("sar", "fusion_resnet18_sar_*.h5"),
]

SAVE_FIGURES = True
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ---------- Load HDF5 results ----------
def read_h5_results(path):
    """Read model metrics and history from a result .h5 file."""
    print(f"\nInspecting: {path}")
    with h5py.File(path, "r") as f:
        name = f["name"][()].decode()
        top1 = float(f["top1"][()])
        classes = [c.decode() for c in f["classes"][:]]
        cm = np.array(f["confusionMat"])
        y_true = np.array(f["yTrue"])
        y_pred = np.array(f["yPred"])

        summary = {}
        cls_report = None
        if "extra" in f and "summary" in f["extra"]:
            summary = json.loads(f["extra/summary"][()].decode())
        if "extra" in f and "classification_report" in f["extra"]:
            cls_report = json.loads(f["extra/classification_report"][()].decode())

        print(f"  Model: {name}")
        print(f"  Top-1 accuracy: {top1:.4f}")
        print(f"  Classes: {len(classes)}")
        print(f"  Confusion matrix shape: {cm.shape}")

    return name, classes, cm, y_true, y_pred, top1, summary, cls_report


# ---------- Plot confusion matrix ----------
def plot_confusion_matrix(cm, classes, title, save_path=None):
    """Visualize and optionally save confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)
    thresh = cm_norm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{cm_norm[i, j]:.2f}",
                ha="center", va="center",
                color="white" if cm_norm[i, j] > thresh else "black",
                fontsize=7,
            )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[✓] Saved confusion matrix → {save_path}")
    plt.close(fig)


# ---------- Plot training curves ----------
def plot_training_curves(h5_path, save_prefix=None):
    """Plot mean and per-member curves and optionally save."""
    with h5py.File(h5_path, "r") as f:
        name = f["name"][()].decode()
        hist = f["history"]
        loss = np.array(hist["loss"]) if "loss" in hist else None
        acc = np.array(hist["acc"]) if "acc" in hist else None
        loss_mean = np.array(hist["loss_mean"]) if "loss_mean" in hist else loss
        acc_mean = np.array(hist["acc_mean"]) if "acc_mean" in hist else acc

    if loss_mean is None and acc_mean is None:
        print("[WARN] No history data available for curves.")
        return

    epochs = np.arange(1, (len(loss_mean) if loss_mean is not None else len(acc_mean)) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"{name} — Training Curves", fontsize=14, fontweight="bold")

    # Loss plot
    if loss_mean is not None:
        axes[0].plot(epochs, loss_mean, color="black", linewidth=1.5)
        axes[0].set_title("Training Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
    else:
        axes[0].axis("off")

    # Accuracy plot
    if acc_mean is not None:
        axes[1].plot(epochs, acc_mean, color="black", linewidth=1.5)
        axes[1].set_title("Training Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
    else:
        axes[1].axis("off")

    plt.tight_layout()

    if save_prefix:
        fig.savefig(f"{save_prefix}_training_curves.png", dpi=300, bbox_inches="tight")
        print(f"[✓] Saved training curves → {save_prefix}_training_curves.png")
    plt.close(fig)


def save_classification_report(report, save_path):
    if report is None:
        return
    try:
        import pandas as pd
    except ImportError:
        print("[WARN] pandas not available — skipping classification report export.")
        return
    df = pd.DataFrame(report).T
    df.to_csv(save_path, float_format="%.6f")
    print(f"[✓] Saved classification report → {save_path}")


# ---------- Main ----------
if __name__ == "__main__":
    for subdir_name, pattern in RESULTS_PATTERNS:
        subdir = FIG_DIR / subdir_name
        subdir.mkdir(exist_ok=True, parents=True)

        search_root = RESULTS_DIR / subdir_name
        if not search_root.exists():
            print(f"[WARN] Missing directory: {search_root}")
            continue

        matched_files = sorted(search_root.glob(pattern))
        if not matched_files:
            print(f"[WARN] No files matched pattern '{pattern}' in {search_root}")
            continue

        for result_path in matched_files:
            name, classes, cm, y_true, y_pred, top1, summary, cls_report = read_h5_results(result_path)

            stem = result_path.stem
            plot_confusion_matrix(cm, classes, f"{name} — Confusion Matrix",
                                  save_path=subdir / f"{stem}_confusion.png")
            plot_training_curves(result_path, save_prefix=subdir / stem)
            save_classification_report(cls_report, subdir / f"{stem}_classification_report.csv")

    print("\n[INFO] Figures exported successfully.")
