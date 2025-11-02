import json
import os
import time
import random
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from sklearn.metrics import classification_report, confusion_matrix

from dataset_reading import DatasetReading
from enable_gpu import enable_gpu
from rand_fusion import train_fusion_member
from utils_results import save_h5_results

# ---------------- Configuration ---------------- #
SEED = 42
DATA_ROOT = "../../data/lcz42"
TDA_ROOT = "../../TDA/data"
SAVE_DIR = "../models/trained"
EPOCHS = 12
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
USE_ZSCORE = True
USE_SAR_DESPECKLE = True
USE_AUG = False

METHODS_MS = [
    ("", "baseline1"),
    ("", "baseline2"),
    ("_vdsr2x", "vdsr2x"),
    ("_edsr2x", "edsr2x"),
    ("_esrgan2x", "esrgan2x"),
    ("_edsr4x", "edsr4x"),
    ("_swinir2x", "swinir2x"),
    ("_vdsr3x", "vdsr3x"),
    ("_bsrnet2x", "bsrnet2x"),
    ("_realesrgan4x", "realesrgan4x"),
]

METHODS_SAR = [
    ("", "baseline1"),
    ("", "baseline2"),
]


# ---------------- Utilities ---------------- #
def setup_seed(seed: int = 42):
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
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return _matlab_to_string(value.reshape(-1)[0])
        if value.dtype.kind in {"U", "S"}:
            return "".join(value.astype(str).reshape(-1))
        if value.dtype == object:
            return "".join(_matlab_to_string(v) for v in value.reshape(-1))
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


def _compute_sumrule(
    tag: str,
    outputs: List[Dict],
    result_dir: str,
    seed: int,
    extra_components: List[str],
):
    if not outputs:
        return None

    classes = [str(c) for c in outputs[0]["classes"]]
    y_true = outputs[0]["y_true"]
    probs_stack = np.stack([out["probs"] for out in outputs], axis=0)

    for out in outputs[1:]:
        if not np.array_equal(out["y_true"], y_true):
            raise ValueError("Mismatch in ground-truth labels across ensemble members.")

    fused_probs = probs_stack.mean(axis=0)
    fused_pred = fused_probs.argmax(axis=1) + 1
    top1 = float((fused_pred == y_true).mean())
    cm = confusion_matrix(y_true, fused_pred, labels=np.arange(1, len(classes) + 1))

    report = classification_report(
        y_true,
        fused_pred,
        labels=np.arange(1, len(classes) + 1),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )

    macro_avg = report.get("macro avg", {})
    weighted_avg = report.get("weighted avg", {})

    summary = {
        "model_name": tag,
        "architecture": "FusionDenseNet201",
        "components": extra_components,
        "final_top1": float(top1),
        "accuracy": float(report.get("accuracy", top1)),
        "macro_precision": float(macro_avg.get("precision", 0.0)),
        "macro_recall": float(macro_avg.get("recall", 0.0)),
        "macro_f1": float(macro_avg.get("f1-score", 0.0)),
        "weighted_precision": float(weighted_avg.get("precision", 0.0)),
        "weighted_recall": float(weighted_avg.get("recall", 0.0)),
        "weighted_f1": float(weighted_avg.get("f1-score", 0.0)),
        "num_classes": len(classes),
        "seed": seed,
        "num_members": len(outputs),
    }

    os.makedirs(result_dir, exist_ok=True)

    save_h5_results(
        h5_path=os.path.join(result_dir, f"{tag}_eval.h5"),
        name=tag,
        classes=classes,
        top1=top1,
        cm=cm,
        y_true=y_true,
        y_pred=fused_pred,
        history={"num_members": np.array([len(outputs)], dtype=np.float32)},
        extra={
            "summary": json.dumps(summary),
            "classification_report": json.dumps(report),
        },
    )

    with open(os.path.join(result_dir, f"{tag}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# ---------------- Main training loop ---------------- #
def train_teacher_fusion(mode="ALL"):
    print("\n[INFO] Starting Fusion Teacher Training (DenseNet201)")
    setup_seed(SEED)
    device = enable_gpu(0)
    print(f"[INFO] Using device: {device}")

    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs("../results", exist_ok=True)
    for subdir in ["rand", "randrgb", "sar", "fusion"]:
        os.makedirs(os.path.join("../results", subdir), exist_ok=True)

    if mode == "ALL":
        modes_to_run = ["RAND", "RANDRGB", "SAR"]
    else:
        modes_to_run = [mode.upper()]
        if modes_to_run[0] not in {"RAND", "RANDRGB", "SAR"}:
            raise ValueError("Mode must be one of RAND | RANDRGB | SAR | ALL")

    cfg_common = dict(
        useZscore=USE_ZSCORE,
        useSARdespeckle=USE_SAR_DESPECKLE,
        useAugmentation=USE_AUG,
    )

    ensemble_outputs: Dict[str, List[Dict]] = {}

    for current_mode in modes_to_run:
        methods = METHODS_MS if current_mode in {"RAND", "RANDRGB"} else METHODS_SAR
        mode_dir = os.path.join("../results", current_mode.lower())
        outputs_for_mode: List[Dict] = []

        print(f"\n=== Training Fusion Mode: {current_mode} ===")
        for member_idx, (suffix, tag) in enumerate(methods, start=1):
            table_path = os.path.join(DATA_ROOT, f"tables_MS{suffix}.mat")
            if not os.path.exists(table_path):
                print(f"[WARN] Missing table file {table_path}, skipping member {member_idx}")
                continue

            train_key, test_key = f"train_MS{suffix}", f"test_MS{suffix}"
            train_table, test_table = load_table_mat(table_path, train_key, test_key)

            cfg_ms = dict(cfg_common)
            cfg_ms["trainTable"] = train_table
            cfg_ms["testTable"] = test_table
            dsTrMS, dsTeMS, infoMS = DatasetReading(cfg_ms)

            dsTrSAR = dsTeSAR = None
            if current_mode == "SAR":
                sar_train_table, sar_test_table = load_table_mat(
                    os.path.join(DATA_ROOT, "tables_SAR.mat"), "train_SAR", "test_SAR"
                )
                cfg_sar = dict(cfg_common)
                cfg_sar["useSARdespeckle"] = True
                cfg_sar["trainTable"] = sar_train_table
                cfg_sar["testTable"] = sar_test_table
                dsTrSAR, dsTeSAR, _ = DatasetReading(cfg_sar)

            tda_train_path = os.path.join(TDA_ROOT, f"tda_MS_features{suffix}.h5")
            tda_test_path = os.path.join(TDA_ROOT, f"tda_MS_features_test{suffix}.h5")

            cfgT = dict(
                dsTrain=dsTrMS,
                dsTest=dsTeMS,
                info=infoMS,
                maxEpochs=EPOCHS,
                miniBatchSize=BATCH_SIZE,
                learnRate=LEARNING_RATE,
                weightDecay=WEIGHT_DECAY,
                labelSmoothing=LABEL_SMOOTHING,
                rngSeed=SEED,
                numWorkers=0,
                device=device,
                tdaTrainPath=tda_train_path,
                tdaTestPath=tda_test_path,
                mode=current_mode,
                memberID=member_idx,
                suffix=tag,
            )

            if current_mode == "SAR":
                cfgT["dsTrainSAR"] = dsTrSAR
                cfgT["dsTestSAR"] = dsTeSAR

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

            model_name = f"fusion_densenet201_{current_mode.lower()}_{tag}"
            model_path = os.path.join(SAVE_DIR, f"{model_name}.pth")
            torch.save(model.state_dict(), model_path)

            report = classification_report(
                y_true,
                y_pred,
                labels=np.arange(1, len(classes) + 1),
                target_names=classes,
                output_dict=True,
                zero_division=0,
            )

            macro_avg = report.get("macro avg", {})
            weighted_avg = report.get("weighted avg", {})

            bands_info = result["bands"]
            summary = {
                "model_name": model_name,
                "architecture": "FusionDenseNet201",
                "mode": current_mode,
                "member_id": member_idx,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "label_smoothing": LABEL_SMOOTHING,
                "optimizer": "AdamW",
                "scheduler": "ReduceLROnPlateau",
                "final_top1": float(top1),
                "accuracy": float(report.get("accuracy", top1)),
                "macro_precision": float(macro_avg.get("precision", 0.0)),
                "macro_recall": float(macro_avg.get("recall", 0.0)),
                "macro_f1": float(macro_avg.get("f1-score", 0.0)),
                "weighted_precision": float(weighted_avg.get("precision", 0.0)),
                "weighted_recall": float(weighted_avg.get("recall", 0.0)),
                "weighted_f1": float(weighted_avg.get("f1-score", 0.0)),
                "train_time_sec": elapsed,
                "num_classes": len(classes),
                "device": str(device),
                "seed": SEED,
                "selected_ms_bands": bands_info.get("ms"),
                "selected_sar_band": bands_info.get("sar"),
                "class_counts": infoMS.get("classCounts").tolist() if "classCounts" in infoMS else None,
                "class_weights": infoMS.get("classWeights").tolist() if "classWeights" in infoMS else None,
            }

            save_h5_results(
                h5_path=os.path.join(mode_dir, f"{model_name}_eval.h5"),
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

            pd.DataFrame(history).to_csv(os.path.join(mode_dir, f"{model_name}_history.csv"), index=False)
            with open(os.path.join(mode_dir, f"{model_name}_summary.json"), "w") as f:
                json.dump(summary, f, indent=2)

            outputs_for_mode.append(result)
            print(f"[✓] {current_mode} member {member_idx:02d} ({tag}) — Top-1: {top1:.4f}")

        if outputs_for_mode:
            ensemble_outputs[current_mode] = outputs_for_mode
            sumrule_name = f"fusion_densenet201_{current_mode.lower()}_sumrule"
            _compute_sumrule(
                sumrule_name,
                outputs_for_mode,
                mode_dir,
                SEED,
                [f"{current_mode}_{idx+1}" for idx in range(len(outputs_for_mode))],
            )

    # Cross-mode sum rule (RAND + RANDRGB + SAR)
    required_modes = {"RAND", "RANDRGB", "SAR"}
    if required_modes.issubset(ensemble_outputs.keys()):
        all_outputs = []
        component_tags = []
        for mode_name in ["RAND", "RANDRGB", "SAR"]:
            all_outputs.extend(ensemble_outputs[mode_name])
            component_tags.extend([f"{mode_name}_{idx+1}" for idx in range(len(ensemble_outputs[mode_name]))])

        fusion_dir = os.path.join("../results", "fusion")
        _compute_sumrule(
            "fusion_densenet201_full_sumrule",
            all_outputs,
            fusion_dir,
            SEED,
            component_tags,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["RAND", "RANDRGB", "SAR", "ALL"], default="ALL")
    args = parser.parse_args()
    train_teacher_fusion(args.mode)
