import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np

try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency
    pd = None

plt.switch_backend("Agg")  # ensure headless execution


def read_h5_results(path: Path) -> Dict:
    """Read model metrics, history, and probabilities from a result .h5 file."""
    print(f"\nInspecting: {path}")
    with h5py.File(path, "r") as f:
        name = f["name"][()].decode()
        top1 = float(f["top1"][()])
        classes = [c.decode() for c in f["classes"][:]]
        cm = np.array(f["confusionMat"])
        y_true = np.array(f["yTrue"])
        y_pred = np.array(f["yPred"])
        probs = np.array(f["probs"]) if "probs" in f else None

        summary = {}
        cls_report = None
        if "extra" in f and "summary" in f["extra"]:
            summary = json.loads(f["extra/summary"][()].decode())
        if "extra" in f and "classification_report" in f["extra"]:
            cls_report = json.loads(f["extra/classification_report"][()].decode())

        history = {}
        if "history" in f:
            for key in f["history"].keys():
                history[key] = np.array(f["history"][key])

        print(f"  Model: {name}")
        print(f"  Top-1 accuracy: {top1:.4f}")
        print(f"  Classes: {len(classes)}")
        print(f"  Confusion matrix shape: {cm.shape}")
        if probs is None:
            print("  [WARN] probabilities dataset not found.")

    return {
        "name": name,
        "classes": classes,
        "confusion_mat": cm,
        "y_true": y_true,
        "y_pred": y_pred,
        "top1": top1,
        "summary": summary,
        "report": cls_report,
        "history": history,
        "probs": probs,
        "path": path,
    }


# ---------- Plot confusion matrix ----------
def plot_confusion_matrix(cm: np.ndarray, classes: List[str], title: str, save_path: Optional[Path] = None):
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
        fig.savefig(str(save_path), dpi=300, bbox_inches="tight")
        print(f"[✓] Saved confusion matrix → {save_path}")
    plt.close(fig)


# ---------- Plot training curves ----------
def plot_training_curves(name: str, history: Dict[str, np.ndarray], save_prefix: Optional[Path] = None):
    """Plot training loss/accuracy curves when present."""
    loss = history.get("loss")
    acc = history.get("acc")
    loss_mean = history.get("loss_mean", loss)
    acc_mean = history.get("acc_mean", acc)

    if loss_mean is None and acc_mean is None:
        print("[WARN] No history data available for curves.")
        return

    reference = loss_mean if loss_mean is not None else acc_mean
    epochs = np.arange(1, len(reference) + 1)
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
        out_path = Path(str(save_prefix) + "_training_curves.png")
        fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
        print(f"[✓] Saved training curves → {out_path}")
    plt.close(fig)


def save_classification_report(report: Optional[Dict], save_path: Path):
    if report is None:
        return
    if pd is None:
        print("[WARN] pandas not available — skipping classification report export.")
        return
    df = pd.DataFrame(report).T
    df.to_csv(save_path, float_format="%.6f")
    print(f"[✓] Saved classification report → {save_path}")


def extract_metrics(entry: Dict, mode: str, member_type: str) -> Dict:
    summary = entry.get("summary") or {}
    report = entry.get("report") or {}
    macro = report.get("macro avg", {})
    weighted = report.get("weighted avg", {})

    def _fetch(key, default=None):
        return summary.get(key, default)

    metrics = {
        "mode": mode,
        "member_type": member_type,
        "name": entry["name"],
        "path": str(entry["path"]),
        "top1": entry["top1"],
        "accuracy": summary.get("accuracy", report.get("accuracy", entry["top1"])),
        "macro_precision": summary.get("macro_precision", macro.get("precision")),
        "macro_recall": summary.get("macro_recall", macro.get("recall")),
        "macro_f1": summary.get("macro_f1", macro.get("f1-score")),
        "weighted_precision": summary.get("weighted_precision", weighted.get("precision")),
        "weighted_recall": summary.get("weighted_recall", weighted.get("recall")),
        "weighted_f1": summary.get("weighted_f1", weighted.get("f1-score")),
        "num_members": _fetch("num_members"),
    }
    return metrics


def ensure_same_labels(outputs: List[Dict]):
    if not outputs:
        return
    ref = outputs[0]["y_true"]
    for entry in outputs[1:]:
        if not np.array_equal(ref, entry["y_true"]):
            raise ValueError("Inconsistent y_true across members — cannot fuse.")


def accuracy_over_members(outputs: List[Dict]) -> List[Tuple[int, float]]:
    if not outputs:
        return []
    ensure_same_labels(outputs)
    if outputs[0]["probs"] is None:
        print("[WARN] Missing probabilities for members — skipping accuracy curve.")
        return []
    running = np.zeros_like(outputs[0]["probs"])
    y_true = outputs[0]["y_true"]
    curve = []
    for idx, entry in enumerate(outputs, start=1):
        if entry["probs"] is None:
            print("[WARN] Missing probabilities for some members — aborting accuracy curve.")
            return []
        running += entry["probs"]
        avg_probs = running / idx
        preds = avg_probs.argmax(axis=1) + 1
        top1 = float((preds == y_true).mean())
        curve.append((idx, top1))
    return curve


def accuracy_over_triplets(outputs_by_mode: Dict[str, List[Dict]], order: List[str]) -> List[Tuple[int, float]]:
    if any(mode not in outputs_by_mode for mode in order):
        return []
    min_len = min(len(outputs_by_mode[mode]) for mode in order)
    if min_len == 0:
        return []

    ensure_same_labels([outputs_by_mode[mode][0] for mode in order])
    if outputs_by_mode[order[0]][0]["probs"] is None:
        print("[WARN] Missing probabilities for combined curve — skipping.")
        return []
    running = np.zeros_like(outputs_by_mode[order[0]][0]["probs"])
    y_true = outputs_by_mode[order[0]][0]["y_true"]
    curve = []
    total_members = 0

    for idx in range(min_len):
        for mode in order:
            entry_probs = outputs_by_mode[mode][idx]["probs"]
            if entry_probs is None:
                print(f"[WARN] Missing probabilities for {mode} member {idx+1} — skipping combined curve.")
                return []
            running += entry_probs
            total_members += 1
        avg_probs = running / total_members
        preds = avg_probs.argmax(axis=1) + 1
        top1 = float((preds == y_true).mean())
        curve.append((total_members, top1))
    return curve


def plot_accuracy_curve(points: List[Tuple[int, float]], title: str, save_path: Optional[Path] = None):
    if not points:
        return
    xs, ys = zip(*points)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys, marker="o", linewidth=1.5)
    ax.set_xlabel("Number of ensemble members")
    ax.set_ylabel("Top-1 accuracy")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_ylim(0, 1)
    if save_path:
        fig.savefig(str(save_path), dpi=300, bbox_inches="tight")
        print(f"[✓] Saved accuracy curve → {save_path}")
    plt.close(fig)


def save_curve(points: List[Tuple[int, float]], save_path: Path):
    if not points:
        return
    array = np.array(points, dtype=np.float64)
    if pd is not None:
        df = pd.DataFrame(array, columns=["members", "top1_accuracy"])
        df.to_csv(save_path, index=False, float_format="%.6f")
    else:
        np.savetxt(save_path, array, delimiter=",", header="members,top1_accuracy", comments="")
    print(f"[✓] Saved accuracy table → {save_path}")


def collect_member_files(
    results_dir: Path,
    mode: str,
    prefix: str,
    suffix: str,
) -> Tuple[List[Path], List[Path]]:
    mode_dir = results_dir / mode.lower()
    if not mode_dir.exists():
        print(f"[WARN] Missing directory: {mode_dir}")
        return [], []

    pattern = f"{prefix}_{mode.lower()}_*{suffix}"
    all_files = sorted(mode_dir.glob(pattern))
    members, sumrules = [], []
    for file in all_files:
        if "_sumrule" in file.stem:
            sumrules.append(file)
        else:
            members.append(file)
    return members, sumrules


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect stored fusion ensemble results.")
    parser.add_argument("--results-dir", default="../results", help="Root directory with result .h5 files.")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["RAND", "RANDRGB", "SAR"],
        help="Modes to inspect (default: RAND RANDRGB SAR).",
    )
    parser.add_argument(
        "--prefix",
        default="fusion_densenet201",
        help="File prefix before _<mode>_ (default: fusion_densenet201).",
    )
    parser.add_argument(
        "--suffix",
        default="_eval.h5",
        help="File suffix to match (default: _eval.h5).",
    )
    parser.add_argument(
        "--figures",
        action="store_true",
        help="Export confusion matrices, training curves, and accuracy plots.",
    )
    parser.add_argument(
        "--no-figures",
        dest="figures",
        action="store_false",
        help="Disable figure export.",
    )
    parser.set_defaults(figures=True)
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir).resolve()
    fig_dir = results_dir / "figures"
    if args.figures:
        fig_dir.mkdir(exist_ok=True, parents=True)

    metrics_rows = []
    outputs_by_mode: Dict[str, List[Dict]] = {}

    for mode in args.modes:
        members, sumrules = collect_member_files(results_dir, mode, args.prefix, args.suffix)
        mode_outputs = []

        if not members:
            print(f"[WARN] No member files found for mode {mode}.")
        for file_path in members:
            entry = read_h5_results(file_path)
            mode_outputs.append(entry)
            metrics_rows.append(extract_metrics(entry, mode, "member"))
            if args.figures:
                subset_dir = fig_dir / mode.lower()
                subset_dir.mkdir(exist_ok=True, parents=True)
                stem = file_path.stem
                plot_confusion_matrix(
                    entry["confusion_mat"],
                    entry["classes"],
                    f"{entry['name']} — Confusion Matrix",
                    save_path=subset_dir / f"{stem}_confusion.png",
                )
                if entry["history"]:
                    plot_training_curves(entry["name"], entry["history"], save_prefix=subset_dir / stem)
                save_classification_report(entry["report"], subset_dir / f"{stem}_classification_report.csv")

        outputs_by_mode[mode] = mode_outputs

        for file_path in sumrules:
            entry = read_h5_results(file_path)
            metrics_rows.append(extract_metrics(entry, mode, "sumrule"))
            if args.figures:
                subset_dir = fig_dir / mode.lower()
                subset_dir.mkdir(exist_ok=True, parents=True)
                stem = file_path.stem
                plot_confusion_matrix(
                    entry["confusion_mat"],
                    entry["classes"],
                    f"{entry['name']} — Confusion Matrix",
                    save_path=subset_dir / f"{stem}_confusion.png",
                )
                save_classification_report(entry["report"], subset_dir / f"{stem}_classification_report.csv")

        if mode_outputs and args.figures:
            curve = accuracy_over_members(mode_outputs)
            curve_path = fig_dir / f"{args.prefix}_{mode.lower()}_accuracy_vs_members.csv"
            plot_path = fig_dir / f"{args.prefix}_{mode.lower()}_accuracy_vs_members.png"
            save_curve(curve, curve_path)
            plot_accuracy_curve(curve, f"{mode} Ensemble Growth", plot_path)

    # Combined curve in triplets (RAND, RANDRGB, SAR)
    canonical_order = ["RAND", "RANDRGB", "SAR"]
    if all(mode in outputs_by_mode and outputs_by_mode[mode] for mode in canonical_order) and args.figures:
        combined_curve = accuracy_over_triplets(outputs_by_mode, canonical_order)
        curve_path = fig_dir / f"{args.prefix}_full_accuracy_vs_members.csv"
        plot_path = fig_dir / f"{args.prefix}_full_accuracy_vs_members.png"
        save_curve(combined_curve, curve_path)
        plot_accuracy_curve(combined_curve, "RAND + RANDRGB + SAR (Triplets)", plot_path)

    # Handle full fusion directory (cross-mode sum rule)
    fusion_dir = results_dir / "fusion"
    if fusion_dir.exists():
        full_pattern = f"{args.prefix}_full_sumrule*{args.suffix}"
        for file_path in sorted(fusion_dir.glob(full_pattern)):
            entry = read_h5_results(file_path)
            metrics_rows.append(extract_metrics(entry, "FUSION", "sumrule"))
            if args.figures:
                subset_dir = fig_dir / "fusion"
                subset_dir.mkdir(exist_ok=True, parents=True)
                stem = file_path.stem
                plot_confusion_matrix(
                    entry["confusion_mat"],
                    entry["classes"],
                    f"{entry['name']} — Confusion Matrix",
                    save_path=subset_dir / f"{stem}_confusion.png",
                )
                save_classification_report(entry["report"], subset_dir / f"{stem}_classification_report.csv")

    # Save aggregated metrics table
    if metrics_rows:
        if pd is not None:
            df = pd.DataFrame(metrics_rows)
            df.sort_values(by=["member_type", "mode", "name"], inplace=True)
            summary_path = results_dir / f"{args.prefix}_metrics_summary.csv"
            df.to_csv(summary_path, index=False, float_format="%.6f")
            print(f"[✓] Saved metrics summary → {summary_path}")
        else:
            print("[WARN] pandas not available — metrics summary table skipped.")

    print("\n[INFO] Inspection completed.")


if __name__ == "__main__":
    main()
