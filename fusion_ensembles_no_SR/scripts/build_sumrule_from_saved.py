import argparse
import os
from pathlib import Path

import h5py
import numpy as np

from train_teacher_fusion import _compute_sumrule, SEED


def _read_member_output(h5_path: Path):
    with h5py.File(h5_path, "r") as h5f:
        if "probs" not in h5f:
            raise RuntimeError(
                f"Dataset 'probs' not found in {h5_path}. "
                "Re-run training after the latest changes so that probabilities are saved."
            )
        classes_raw = h5f["classes"][:]
        classes = [
            cls.decode("utf-8") if isinstance(cls, (bytes, np.bytes_, np.ndarray)) else str(cls)
            for cls in classes_raw
        ]
        y_true = h5f["yTrue"][:].astype(np.int32)
        probs = h5f["probs"][:].astype(np.float32)
    return {"classes": classes, "y_true": y_true, "probs": probs}


def _collect_members(mode: str, results_root: Path):
    mode_lower = mode.lower()
    mode_dir = results_root / mode_lower
    if not mode_dir.exists():
        raise FileNotFoundError(f"Results directory not found for mode '{mode}': {mode_dir}")

    pattern = f"fusion_densenet201_{mode_lower}_*_eval.h5"
    members = sorted(
        [
            path
            for path in mode_dir.glob(pattern)
            if "_sumrule_" not in path.name  # skip already fused evaluations
        ]
    )

    if not members:
        raise FileNotFoundError(f"No member eval files found matching {pattern} in {mode_dir}")

    outputs = [_read_member_output(path) for path in members]
    component_tags = [f"{mode}_{idx+1}" for idx in range(len(outputs))]
    sumrule_tag = f"fusion_densenet201_{mode_lower}_sumrule"

    summary = _compute_sumrule(
        sumrule_tag,
        outputs,
        str(mode_dir),
        SEED,
        component_tags,
    )
    print(f"[INFO] {mode} sum rule → Top-1 {summary['final_top1']:.4f} ({len(outputs)} members)")
    return outputs


def _compute_full_sumrule(outputs_by_mode, results_root: Path):
    required_modes = ["RAND", "RANDRGB", "SAR"]
    missing = [m for m in required_modes if m not in outputs_by_mode]
    if missing:
        raise RuntimeError(f"Missing member outputs for modes: {', '.join(missing)}")

    all_outputs = []
    component_tags = []
    for mode in required_modes:
        mode_outputs = outputs_by_mode[mode]
        all_outputs.extend(mode_outputs)
        component_tags.extend([f"{mode}_{idx+1}" for idx in range(len(mode_outputs))])

    fusion_dir = results_root / "fusion"
    os.makedirs(fusion_dir, exist_ok=True)
    summary = _compute_sumrule(
        "fusion_densenet201_full_sumrule",
        all_outputs,
        str(fusion_dir),
        SEED,
        component_tags,
    )
    print(
        "[INFO] FULL sum rule (RAND+RANDRGB+SAR) → "
        f"Top-1 {summary['final_top1']:.4f} ({len(all_outputs)} members)"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Build sum-rule ensembles from saved member evaluations."
    )
    parser.add_argument(
        "--mode",
        choices=["RAND", "RANDRGB", "SAR", "ALL"],
        default="ALL",
        help="Which ensemble(s) to fuse. 'ALL' fuses each mode (if possible) and then the full 30-member set.",
    )
    parser.add_argument(
        "--results-root",
        default=str(Path(__file__).resolve().parent.parent / "results"),
        help="Root directory containing per-mode results folders.",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root).resolve()
    if not results_root.exists():
        raise FileNotFoundError(f"Results root directory does not exist: {results_root}")

    outputs_cache = {}

    if args.mode == "ALL":
        for mode in ["RAND", "RANDRGB", "SAR"]:
            try:
                outputs_cache[mode] = _collect_members(mode, results_root)
            except FileNotFoundError as err:
                print(f"[WARN] {err}")
        if len(outputs_cache) == 3:
            _compute_full_sumrule(outputs_cache, results_root)
        else:
            missing = ", ".join(
                sorted(set(["RAND", "RANDRGB", "SAR"]) - set(outputs_cache.keys()))
            )
            raise RuntimeError(
                f"Cannot build full sum rule because the following modes are missing: {missing}"
            )
    else:
        outputs_cache[args.mode] = _collect_members(args.mode, results_root)


if __name__ == "__main__":
    main()
