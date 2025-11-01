import json
import os
import random
import time
from typing import Any, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from sklearn.metrics import classification_report

from dataset_reading import DatasetReading
from enable_gpu import enable_gpu
from rand_resnet18 import train_rand_resnet18
from randrgb_resnet18 import train_randrgb_resnet18
from randsar_resnet18 import train_randsar_resnet18
from utils_results import save_h5_results

# ---------------- Configuration ---------------- #
SEED = 42
DATA_ROOT = "../../data/lcz42"
SAVE_DIR = "../models/trained"
EPOCHS = 12
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
USE_ZSCORE = True
USE_SAR_DESPECKLE = False
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


def load_table_mat(path: str, train_key: str, test_key: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data = loadmat(path, simplify_cells=False)
    train = _matlab_table_to_df(data[train_key])
    test = _matlab_table_to_df(data[test_key])
    return train, test


def train_teacher_resnet18(mode="ALL"):
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
        )

        start_time = time.time()
        res = train_rand_resnet18(cfgT)
        elapsed = time.time() - start_time

        model = res["ensemble"]
        history = res["history"]
        classes_list = [str(c) for c in res["classes"]]
        members_meta = res["members"]
        top1 = res["test_top1"]
        cm = res["confusion_mat"]
        y_true = res["y_true"]
        y_pred = res["y_pred"]

        checkpoint_path = os.path.join(SAVE_DIR, "Rand_resnet18.pth")
        torch.save(model.state_dict(), checkpoint_path)

        report = classification_report(
            y_true, y_pred,
            labels=np.arange(1, len(classes_list) + 1),
            target_names=classes_list,
            output_dict=True, zero_division=0
        )

        summary = {
            "model_name": "Rand",
            "architecture": "ResNet18",
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
        }

        members_records = [
            {
                "member": idx + 1,
                "bands": ",".join(map(str, meta.bands_one_based)),
                "final_train_loss": meta.final_loss,
                "final_train_acc": meta.final_acc,
            }
            for idx, meta in enumerate(members_meta)
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
        )

        start_time = time.time()
        res = train_randrgb_resnet18(cfgT)
        elapsed = time.time() - start_time

        model = res["ensemble"]
        history = res["history"]
        classes_list = res["classes"]
        members_meta = res["members"]
        top1 = res["test_top1"]
        cm = res["confusion_mat"]
        y_true = res["y_true"]
        y_pred = res["y_pred"]

        checkpoint_path = os.path.join(SAVE_DIR, "RandRGB_resnet18.pth")
        torch.save(model.state_dict(), checkpoint_path)

        report = classification_report(
            y_true, y_pred,
            labels=np.arange(1, len(classes_list) + 1),
            target_names=classes_list,
            output_dict=True, zero_division=0
        )

        summary = {
            "model_name": "RandRGB",
            "architecture": "ResNet18",
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
        }

        members_records = [
            {
                "member": idx + 1,
                "bands": ",".join(map(str, meta.bands_one_based)),
                "final_train_loss": meta.final_loss,
                "final_train_acc": meta.final_acc,
            }
            for idx, meta in enumerate(members_meta)
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
        )

        start_time = time.time()
        res = train_randsar_resnet18(cfgT)
        elapsed = time.time() - start_time

        model = res["ensemble"]
        history = res["history"]
        classes_list = res["classes"]
        members_meta = res["members"]
        top1 = res["test_top1"]
        cm = res["confusion_mat"]
        y_true = res["y_true"]
        y_pred = res["y_pred"]

        checkpoint_path = os.path.join(SAVE_DIR, "SAR_resnet18.pth")
        torch.save(model.state_dict(), checkpoint_path)

        report = classification_report(
            y_true, y_pred,
            labels=np.arange(1, len(classes_list) + 1),
            target_names=classes_list,
            output_dict=True, zero_division=0
        )

        summary = {
            "model_name": "SAR",
            "architecture": "ResNet18",
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
        }

        members_records = [
            {
                "member": idx + 1,
                "bands": ",".join(map(str, meta.bands_one_based)),
                "final_train_loss": meta.final_loss,
                "final_train_acc": meta.final_acc,
            }
            for idx, meta in enumerate(members_meta)
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["RAND", "RANDRGB", "SAR", "ALL"], default="ALL")
    args = parser.parse_args()
    train_teacher_resnet18(args.mode)
