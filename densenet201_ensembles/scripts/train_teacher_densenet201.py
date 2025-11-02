import json
import os
import random
import time
from typing import Any, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from sklearn.metrics import classification_report, confusion_matrix

from dataset_reading import DatasetReading
from enable_gpu import enable_gpu
from rand_densenet201 import train_rand_densenet201
from randrgb_densenet201 import train_randrgb_densenet201
from randsar_densenet201 import train_randsar_densenet201
from utils_results import save_h5_results

# ---------------- Configuration ---------------- #
SEED = 42
DATA_ROOT = "../../data/lcz42"
SAVE_DIR = "../models/trained"
EPOCHS = 12
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
USE_ZSCORE = True
USE_SAR_DESPECKLE = True
USE_AUG = True


def setup_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def _matlab_to_scalar(value: Any) -> int:
    """Convert MATLAB numeric cells/arrays into Python ints."""
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return int(value)
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return _matlab_to_scalar(value.reshape(-1)[0])
    return int(np.array(value).reshape(-1)[0])


def _matlab_to_string(value: Any) -> str:
    """Convert MATLAB char/cell arrays into Python strings."""
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


def _matlab_table_to_df(table_array: np.ndarray) -> pd.DataFrame:
    """Create a clean DataFrame (Path, Label, Index, Modality) from MATLAB-exported content."""
    df = pd.DataFrame(table_array, columns=["Path", "Label", "Index", "Modality"])
    df["Path"] = df["Path"].map(_matlab_to_string)
    df["Label"] = df["Label"].map(_matlab_to_scalar)
    df["Index"] = df["Index"].map(_matlab_to_scalar).astype(int) - 1  # MATLAB -> zero-based
    df["Modality"] = df["Modality"].map(lambda x: _matlab_to_string(x).upper())
    return df


def _format_member_record(idx: int, meta) -> dict:
    """Uniform CSV row for ensemble member diagnostics."""
    record = {
        "member": idx + 1,
        "final_train_loss": meta.final_loss,
        "final_train_acc": meta.final_acc,
    }
    bands_info = getattr(meta, "bands_one_based", [])
    if isinstance(bands_info, dict):
        record["ms_bands"] = ",".join(map(str, bands_info.get("ms", [])))
        record["sar_bands"] = ",".join(map(str, bands_info.get("sar", [])))
    else:
        record["bands"] = ",".join(map(str, bands_info))

    if hasattr(meta, "ms_bands_one_based"):
        record["ms_bands"] = ",".join(map(str, getattr(meta, "ms_bands_one_based", [])))
    if hasattr(meta, "sar_bands_one_based"):
        record["sar_bands"] = ",".join(map(str, getattr(meta, "sar_bands_one_based", [])))
    if hasattr(meta, "sar_components"):
        record["sar_components"] = getattr(meta, "sar_components")
    return record


def load_table_mat(path: str, train_key: str, test_key: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data = loadmat(path, simplify_cells=False)
    train = _matlab_table_to_df(data[train_key])
    test = _matlab_table_to_df(data[test_key])
    return train, test


def train_teacher_densenet201(mode="ALL"):
    """
    mode: "RAND", "RANDRGB", "SAR", or "ALL"
    """
    setup_seed(SEED)
    device = enable_gpu(0)
    print(f"[INFO] Using device: {device}")

    # Load tables
    train_MS, test_MS = load_table_mat(os.path.join(DATA_ROOT, "tables_MS.mat"), "train_MS", "test_MS")
    train_SAR, test_SAR = load_table_mat(os.path.join(DATA_ROOT, "tables_SAR.mat"), "train_SAR", "test_SAR")

    cfg_common = dict(
        useZscore=USE_ZSCORE,
        useSARdespeckle=USE_SAR_DESPECKLE,
        useAugmentation=USE_AUG,
    )

    do_rand = mode in ["RAND", "ALL"]
    do_randrgb = mode in ["RANDRGB", "ALL"]
    do_sar = mode in ["SAR", "ALL"]

    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs("../results", exist_ok=True)

    ensemble_results = {}

    # ---------------- RAND ---------------- #
    if do_rand:
        print("\n=== Training RAND (MS) ===")
        cfg = dict(cfg_common)
        cfg["trainTable"] = train_MS
        cfg["testTable"] = test_MS
        dsTr, dsTe, info = DatasetReading(cfg)
        cfgT = dict(
            dsTrain=dsTr,
            dsTest=dsTe,
            info=info,
            maxEpochs=EPOCHS,
            miniBatchSize=BATCH_SIZE,
            learnRate=LEARNING_RATE,
            rngSeed=SEED,
            numWorkers=6,
            device=device,
            weightDecay=WEIGHT_DECAY,
        )

        start_time = time.time()
        res = train_rand_densenet201(cfgT)
        elapsed = time.time() - start_time
        ensemble_results["Rand"] = res

        model = res["ensemble"]
        history = res["history"]
        classes_list = [str(c) for c in res["classes"]]
        members_meta = res["members"]
        top1 = res["test_top1"]
        cm = res["confusion_mat"]
        y_true = res["y_true"]
        y_pred = res["y_pred"]

        checkpoint_path = os.path.join(SAVE_DIR, "Rand_densenet201.pth")
        torch.save(model.state_dict(), checkpoint_path)

        report = classification_report(
            y_true, y_pred,
            labels=np.arange(1, len(classes_list) + 1),
            target_names=classes_list,
            output_dict=True, zero_division=0
        )

        macro_avg = report.get("macro avg", {})
        weighted_avg = report.get("weighted avg", {})

        summary = {
            "model_name": "Rand",
            "architecture": "DenseNet201",
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "optimizer": "SGD(momentum=0.9)",
            "scheduler": "None",
            "final_top1": float(top1),
            "train_time_sec": elapsed,
            "num_classes": len(classes_list),
            "device": str(device),
            "seed": SEED,
            "num_members": res["num_members"],
            "accuracy": float(report.get("accuracy", top1)),
            "macro_precision": float(macro_avg.get("precision", 0.0)),
            "macro_recall": float(macro_avg.get("recall", 0.0)),
            "macro_f1": float(macro_avg.get("f1-score", 0.0)),
            "weighted_precision": float(weighted_avg.get("precision", 0.0)),
            "weighted_recall": float(weighted_avg.get("recall", 0.0)),
            "weighted_f1": float(weighted_avg.get("f1-score", 0.0)),
        }

        members_records = [
            _format_member_record(idx, meta) for idx, meta in enumerate(members_meta)
        ]
        history_curves = pd.DataFrame({
            "epoch": np.arange(1, history["loss_mean"].shape[0] + 1),
            "loss_mean": history["loss_mean"],
            "acc_mean": history["acc_mean"],
        })

        save_h5_results(
            h5_path="../results/rand/rand_eval_TEST.h5",
            name="Rand",
            classes=classes_list,
            top1=top1,
            cm=cm,
            y_true=y_true,
            y_pred=y_pred,
            history=history,
            extra={
                "checkpoint_path": checkpoint_path,
                "classification_report": json.dumps(report),
                "summary": json.dumps(summary),
                "members": json.dumps(members_records),
            },
        )

        pd.DataFrame(members_records).to_csv("../results/rand/rand_members.csv", index=False)
        history_curves.to_csv("../results/rand/rand_history.csv", index=False)
        with open("../results/rand/rand_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"[INFO] RAND model finished — Top-1: {top1:.4f}")

    # ---------------- RANDRGB ---------------- #
    if do_randrgb:
        print("\n=== Training RANDRGB (MS) ===")
        cfg = dict(cfg_common)
        cfg["trainTable"] = train_MS
        cfg["testTable"] = test_MS
        dsTr, dsTe, info = DatasetReading(cfg)
        cfgT = dict(
            dsTrain=dsTr,
            dsTest=dsTe,
            info=info,
            maxEpochs=EPOCHS,
            miniBatchSize=BATCH_SIZE,
            learnRate=LEARNING_RATE,
            rngSeed=SEED,
            numWorkers=6,
            device=device,
            weightDecay=WEIGHT_DECAY,
        )

        start_time = time.time()
        res = train_randrgb_densenet201(cfgT)
        elapsed = time.time() - start_time
        ensemble_results["RandRGB"] = res

        model = res["ensemble"]
        history = res["history"]
        classes_list = res["classes"]
        members_meta = res["members"]
        top1 = res["test_top1"]
        cm = res["confusion_mat"]
        y_true = res["y_true"]
        y_pred = res["y_pred"]

        checkpoint_path = os.path.join(SAVE_DIR, "RandRGB_densenet201.pth")
        torch.save(model.state_dict(), checkpoint_path)

        report = classification_report(
            y_true, y_pred,
            labels=np.arange(1, len(classes_list) + 1),
            target_names=classes_list,
            output_dict=True, zero_division=0
        )

        macro_avg = report.get("macro avg", {})
        weighted_avg = report.get("weighted avg", {})

        summary = {
            "model_name": "RandRGB",
            "architecture": "DenseNet201",
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "optimizer": "SGD(momentum=0.9)",
            "scheduler": "None",
            "final_top1": float(top1),
            "train_time_sec": elapsed,
            "num_classes": len(classes_list),
            "device": str(device),
            "seed": SEED,
            "num_members": res["num_members"],
            "accuracy": float(report.get("accuracy", top1)),
            "macro_precision": float(macro_avg.get("precision", 0.0)),
            "macro_recall": float(macro_avg.get("recall", 0.0)),
            "macro_f1": float(macro_avg.get("f1-score", 0.0)),
            "weighted_precision": float(weighted_avg.get("precision", 0.0)),
            "weighted_recall": float(weighted_avg.get("recall", 0.0)),
            "weighted_f1": float(weighted_avg.get("f1-score", 0.0)),
        }

        members_records = [
            _format_member_record(idx, meta) for idx, meta in enumerate(members_meta)
        ]
        history_curves = pd.DataFrame({
            "epoch": np.arange(1, history["loss_mean"].shape[0] + 1),
            "loss_mean": history["loss_mean"],
            "acc_mean": history["acc_mean"],
        })

        save_h5_results(
            h5_path="../results/randrgb/randrgb_eval_TEST.h5",
            name="RandRGB",
            classes=classes_list,
            top1=top1,
            cm=cm,
            y_true=y_true,
            y_pred=y_pred,
            history=history,
            extra={
                "checkpoint_path": checkpoint_path,
                "classification_report": json.dumps(report),
                "summary": json.dumps(summary),
                "members": json.dumps(members_records),
            },
        )

        pd.DataFrame(members_records).to_csv("../results/randrgb/randrgb_members.csv", index=False)
        history_curves.to_csv("../results/randrgb/randrgb_history.csv", index=False)
        with open("../results/randrgb/randrgb_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"[INFO] RANDRGB model finished — Top-1: {top1:.4f}")

    # ---------------- SAR ---------------- #
    if do_sar:
        print("\n=== Training SAR (MS + SAR) ===")
        cfgMS = dict(cfg_common)
        cfgMS["trainTable"] = train_MS
        cfgMS["testTable"] = test_MS
        dsTrMS, dsTeMS, infoMS = DatasetReading(cfgMS)

        cfgSAR = dict(cfg_common)
        cfgSAR["trainTable"] = train_SAR
        cfgSAR["testTable"] = test_SAR
        dsTrSAR, dsTeSAR, infoSAR = DatasetReading(cfgSAR)

        cfgT = dict(
            dsTrain=dsTrMS,
            dsTest=dsTeMS,
            dsTrainSAR=dsTrSAR,
            dsTestSAR=dsTeSAR,
            info=infoMS,
            maxEpochs=EPOCHS,
            miniBatchSize=BATCH_SIZE,
            learnRate=LEARNING_RATE,
            rngSeed=SEED,
            numWorkers=6,
            device=device,
            weightDecay=WEIGHT_DECAY,
        )

        start_time = time.time()
        res = train_randsar_densenet201(cfgT)
        elapsed = time.time() - start_time
        ensemble_results["SAR"] = res

        model = res["ensemble"]
        history = res["history"]
        classes_list = res["classes"]
        members_meta = res["members"]
        top1 = res["test_top1"]
        cm = res["confusion_mat"]
        y_true = res["y_true"]
        y_pred = res["y_pred"]

        checkpoint_path = os.path.join(SAVE_DIR, "SAR_densenet201.pth")
        torch.save(model.state_dict(), checkpoint_path)

        report = classification_report(
            y_true, y_pred,
            labels=np.arange(1, len(classes_list) + 1),
            target_names=classes_list,
            output_dict=True, zero_division=0
        )

        macro_avg = report.get("macro avg", {})
        weighted_avg = report.get("weighted avg", {})

        summary = {
            "model_name": "SAR",
            "architecture": "DenseNet201",
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "optimizer": "SGD(momentum=0.9)",
            "scheduler": "None",
            "final_top1": float(top1),
            "train_time_sec": elapsed,
            "num_classes": len(classes_list),
            "device": str(device),
            "seed": SEED,
            "num_members": res["num_members"],
            "accuracy": float(report.get("accuracy", top1)),
            "macro_precision": float(macro_avg.get("precision", 0.0)),
            "macro_recall": float(macro_avg.get("recall", 0.0)),
            "macro_f1": float(macro_avg.get("f1-score", 0.0)),
            "weighted_precision": float(weighted_avg.get("precision", 0.0)),
            "weighted_recall": float(weighted_avg.get("recall", 0.0)),
            "weighted_f1": float(weighted_avg.get("f1-score", 0.0)),
        }

        members_records = [
            _format_member_record(idx, meta) for idx, meta in enumerate(members_meta)
        ]
        history_curves = pd.DataFrame({
            "epoch": np.arange(1, history["loss_mean"].shape[0] + 1),
            "loss_mean": history["loss_mean"],
            "acc_mean": history["acc_mean"],
        })

        save_h5_results(
            h5_path="../results/sar/sar_eval_TEST.h5",
            name="SAR",
            classes=classes_list,
            top1=top1,
            cm=cm,
            y_true=y_true,
            y_pred=y_pred,
            history=history,
            extra={
                "checkpoint_path": checkpoint_path,
                "classification_report": json.dumps(report),
                "summary": json.dumps(summary),
                "members": json.dumps(members_records),
            },
        )

        pd.DataFrame(members_records).to_csv("../results/sar/sar_members.csv", index=False)
        history_curves.to_csv("../results/sar/sar_history.csv", index=False)
        with open("../results/sar/sar_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"[INFO] SAR model finished — Top-1: {top1:.4f}")

    if {"Rand", "RandRGB", "SAR"}.issubset(ensemble_results.keys()):
        print("\n=== Computing DenseNet201 Sum-Rule Fusion (Rand + RandRGB + SAR) ===")
        rand_res = ensemble_results["Rand"]
        randrgb_res = ensemble_results["RandRGB"]
        sar_res = ensemble_results["SAR"]

        y_true = rand_res["y_true"]
        classes_list = [str(c) for c in rand_res["classes"]]

        fused_scores = (
            rand_res["scores_avg"]
            + randrgb_res["scores_avg"]
            + sar_res["scores_avg"]
        ) / 3.0
        fused_pred = fused_scores.argmax(axis=1) + 1
        fused_top1 = float((fused_pred == y_true).mean())
        fused_cm = confusion_matrix(
            y_true, fused_pred, labels=np.arange(1, len(classes_list) + 1)
        )

        report = classification_report(
            y_true,
            fused_pred,
            labels=np.arange(1, len(classes_list) + 1),
            target_names=classes_list,
            output_dict=True,
            zero_division=0,
        )
        macro_avg = report.get("macro avg", {})
        weighted_avg = report.get("weighted avg", {})

        fusion_summary = {
            "model_name": "DenseNet201_SumRule",
            "architecture": "DenseNet201",
            "components": ["Rand", "RandRGB", "SAR"],
            "final_top1": float(fused_top1),
            "accuracy": float(report.get("accuracy", fused_top1)),
            "macro_precision": float(macro_avg.get("precision", 0.0)),
            "macro_recall": float(macro_avg.get("recall", 0.0)),
            "macro_f1": float(macro_avg.get("f1-score", 0.0)),
            "weighted_precision": float(weighted_avg.get("precision", 0.0)),
            "weighted_recall": float(weighted_avg.get("recall", 0.0)),
            "weighted_f1": float(weighted_avg.get("f1-score", 0.0)),
            "num_classes": len(classes_list),
            "seed": SEED,
        }

        fusion_dir = "../results/fusion"
        os.makedirs(fusion_dir, exist_ok=True)

        save_h5_results(
            h5_path=os.path.join(fusion_dir, "densenet201_sumrule_eval_TEST.h5"),
            name="DenseNet201_SumRule",
            classes=classes_list,
            top1=fused_top1,
            cm=fused_cm,
            y_true=y_true,
            y_pred=fused_pred,
            history={},
            extra={
                "summary": json.dumps(fusion_summary),
                "classification_report": json.dumps(report),
            },
        )

        with open(os.path.join(fusion_dir, "densenet201_sumrule_summary.json"), "w") as f:
            json.dump(fusion_summary, f, indent=2)

        print(f"[INFO] Sum-rule fusion finished — Top-1: {fused_top1:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["RAND", "RANDRGB", "SAR", "ALL"], default="ALL")
    args = parser.parse_args()
    train_teacher_densenet201(args.mode)
