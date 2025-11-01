import h5py
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

# ---------- Configuration ----------
RESULTS_DIR = Path("../results")
FILES = {
    "Rand": "rand/rand_eval_TEST.h5",
    "RandRGB": "randrgb/randrgb_eval_TEST.h5",
    "SAR": "sar/sar_eval_TEST.h5",
}

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
        if "extra" in f and "summary" in f["extra"]:
            summary = json.loads(f["extra/summary"][()].decode())

        print(f"  Model: {name}")
        print(f"  Top-1 accuracy: {top1:.4f}")
        print(f"  Classes: {len(classes)}")
        print(f"  Confusion matrix shape: {cm.shape}")

    return name, classes, cm, y_true, y_pred, top1, summary


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
        loss_mean = np.array(hist["loss_mean"])
        acc_mean = np.array(hist["acc_mean"])
        loss_per_member = np.array(hist["loss_per_member"])
        acc_per_member = np.array(hist["acc_per_member"])

    epochs = np.arange(1, len(loss_mean) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"{name} — Training Curves", fontsize=14, fontweight="bold")

    # Loss plot
    for i in range(loss_per_member.shape[0]):
        axes[0].plot(epochs, loss_per_member[i], alpha=0.2)
    axes[0].plot(epochs, loss_mean, color="black", linewidth=1, label="Mean")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")

    # Accuracy plot
    for i in range(acc_per_member.shape[0]):
        axes[1].plot(epochs, acc_per_member[i], alpha=0.2)
    axes[1].plot(epochs, acc_mean, color="black", linewidth=1, label="Mean")
    axes[1].set_title("Training Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")

    plt.tight_layout()

    if save_prefix:
        fig.savefig(f"{save_prefix}_training_curves.png", dpi=300, bbox_inches="tight")
        print(f"[✓] Saved training curves → {save_prefix}_training_curves.png")
    plt.close(fig)


# ---------- Main ----------
if __name__ == "__main__":
    for model_name, rel_path in FILES.items():
        path = RESULTS_DIR / rel_path
        if not path.exists():
            print(f"[WARN] Missing file: {rel_path}")
            continue

        name, classes, cm, y_true, y_pred, top1, summary = read_h5_results(path)

        subdir = FIG_DIR / model_name.lower()
        subdir.mkdir(exist_ok=True, parents=True)

        # Plot and save confusion matrix + training curves
        plot_confusion_matrix(cm, classes, f"{name} — Confusion Matrix",
                              save_path=subdir / f"{model_name}_confusion.png")
        plot_training_curves(path, save_prefix=subdir / model_name)

    print("\n[INFO] Figures exported successfully.")