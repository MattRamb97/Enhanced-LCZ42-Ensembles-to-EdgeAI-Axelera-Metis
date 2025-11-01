import os
import json
import time
import random
import numpy as np
import pandas as pd
import torch

from scipy.io import loadmat
from sklearn.metrics import classification_report

from enable_gpu import enable_gpu
from utils_results import save_h5_results
from rand_fusion import train_fusion_member
from dataset_reading import DatasetReading

# ---------------- Configuration ---------------- #
SEED = 42
DATA_ROOT = "../../data/lcz42"
TDA_ROOT = "../../TDA/data"
SAVE_DIR = "../models/trained"
EPOCHS = 12
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
USE_ZSCORE = True
USE_SAR_DESPECKLE = False
USE_AUG = False

methods = [
    ("","baseline1"),
    ("","baseline2"),
    ("_vdsr2x", "vdsr2x"),
    ("_edsr2x", "edsr2x"),
    ("_esrgan2x", "esrgan2x"),
    ("_edsr4x", "edsr4x"),
    ("_swinir2x", "swinir2x"),
    ("_vdsr3x", "vdsr3x"),
    ("_bsrnet2x", "bsrnet2x"),
    ("_realesrgan4x", "realesrgan4x"),
]


# ---------------- Utilities ---------------- #
def setup_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def _matlab_to_scalar(value):
    if isinstance(value, (int, np.integer, float, np.floating)):
        return int(value)
    if isinstance(value, np.ndarray) and value.size == 1:
        return _matlab_to_scalar(value.reshape(-1)[0])
    return int(np.array(value).reshape(-1)[0])

def _matlab_to_string(value):
    if isinstance(value, str): return value
    if isinstance(value, (bytes, bytearray)): return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.size == 1: return _matlab_to_string(value.reshape(-1)[0])
        if value.dtype.kind in {"U", "S"}: return "".join(value.astype(str).reshape(-1))
        if value.dtype == object: return "".join(_matlab_to_string(v) for v in value.reshape(-1))
    return str(value)

def _matlab_table_to_df(table_array):
    df = pd.DataFrame(table_array, columns=["Path", "Label", "Index", "Modality"])
    df["Path"] = df["Path"].map(_matlab_to_string)
    df["Label"] = df["Label"].map(_matlab_to_scalar)
    df["Index"] = df["Index"].map(_matlab_to_scalar).astype(int) - 1
    df["Modality"] = df["Modality"].map(lambda x: _matlab_to_string(x).upper())
    return df

def load_table_mat(path, train_key, test_key):
    data = loadmat(path, simplify_cells=False)
    return _matlab_table_to_df(data[train_key]), _matlab_table_to_df(data[test_key])


# ---------------- Main training loop ---------------- #
def train_teacher_fusion(mode="RAND"):
    print("\n[INFO] Starting Fusion Teacher Training")
    setup_seed(SEED)
    device = enable_gpu(0)
    print(f"[INFO] Using device: {device}")

    cfg_common = dict(
        useZscore=USE_ZSCORE,
        useSARdespeckle=USE_SAR_DESPECKLE,
        useAugmentation=USE_AUG,
    )

    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs("../results/rand", exist_ok=True)

    if mode not in ["RAND"]:
        raise ValueError("Only RAND mode is implemented for fusion teacher script")

    # ---------------- RAND FUSION MEMBERS ---------------- #
    for i, (suffix, tag) in enumerate(methods):
        print(f"\n=== Training RAND Fusion Member {i+1:02d} ({tag}) ===")

        table_path = os.path.join(DATA_ROOT, f"tables_MS{suffix}.mat")
        if not os.path.exists(table_path):
            print(f"[WARN] Missing table file {table_path}, skipping member {i+1}")
            continue

        train_key, test_key = f"train_MS{suffix}", f"test_MS{suffix}"
        train_table, test_table = load_table_mat(table_path, train_key, test_key)

        cfg = dict(cfg_common)
        cfg["trainTable"] = train_table
        cfg["testTable"] = test_table
        dsTr, dsTe, info = DatasetReading(cfg)

        tda_file = f"tda_MS_features{suffix}.h5"
        tda_file_test = f"tda_MS_features_test{suffix}.h5"

        cfgT = dict(
            dsTrain=dsTr,
            dsTest=dsTe,
            info=info,
            maxEpochs=EPOCHS,
            miniBatchSize=BATCH_SIZE,
            learnRate=LEARNING_RATE,
            weightDecay=WEIGHT_DECAY,
            labelSmoothing=LABEL_SMOOTHING,
            rngSeed=SEED,
            numWorkers=0,
            device=device,
            tdaTrainPath=os.path.join(TDA_ROOT, tda_file),
            tdaTestPath=os.path.join(TDA_ROOT, tda_file_test),
            ensembleType="Rand",
            memberID=i + 1,
            suffix=tag,
        )

        start_time = time.time()
        result = train_fusion_member(cfgT)
        elapsed = time.time() - start_time

        model = result["model"]
        history = result["history"]
        top1 = result["top1"]
        cm = result["confusion_mat"]
        y_true = result["y_true"]
        y_pred = result["y_pred"]
        classes = [str(c) for c in result["classes"]]

        model_name = f"fusion_resnet18_rand_{tag}"
        model_path = os.path.join(SAVE_DIR, f"{model_name}.pth")
        torch.save(model.state_dict(), model_path)

        report = classification_report(
            y_true, y_pred,
            labels=np.arange(1, len(classes) + 1),
            target_names=classes,
            output_dict=True,
            zero_division=0,
        )

        summary = {
            "model_name": model_name,
            "architecture": "FusionResNet18",
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "label_smoothing": LABEL_SMOOTHING,
            "optimizer": "AdamW",
            "scheduler": "ReduceLROnPlateau",
            "final_top1": float(top1),
            "train_time_sec": elapsed,
            "num_classes": len(classes),
            "device": str(device),
            "seed": SEED,
            "selected_bands": result["bands_one_based"],
            "class_counts": info.get("classCounts").tolist() if "classCounts" in info else None,
            "class_weights": info.get("classWeights").tolist() if "classWeights" in info else None,
        }

        result_dir = "../results/rand"
        save_h5_results(
            h5_path=os.path.join(result_dir, f"{model_name}_eval.h5"),
            name=model_name,
            classes=classes,
            top1=top1,
            cm=cm,
            y_true=y_true,
            y_pred=y_pred,
            history=history,
            extra={
                "summary": json.dumps(summary),
                "classification_report": json.dumps(report),
            },
        )

        pd.DataFrame(history).to_csv(os.path.join(result_dir, f"{model_name}_history.csv"), index=False)
        with open(os.path.join(result_dir, f"{model_name}_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        print(f"[✓] Member {i+1} ({tag}) finished — Top-1: {top1:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["RAND"], default="RAND")
    args = parser.parse_args()
    train_teacher_fusion(args.mode)
