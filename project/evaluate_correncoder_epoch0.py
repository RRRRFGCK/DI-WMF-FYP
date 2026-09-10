"""Evaluate Correncoder initialisations before any BIDMC optimiser update."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.stats import t as student_t
from torch.utils.data import DataLoader, TensorDataset

from domain_mf.models import PublishedCorrEncoder1D
from domain_mf.regression import (
    calibrate_regression_output,
    fit_safe_calibration_head,
    initialise_1d_layerwise_matched_filter,
    initialise_1d_matched_filter,
)
from run_correncoder_regression import (
    _evaluate,
    build_fold_tensors,
    load_bidmc_subjects,
    overlap_average,
    lag_robust_waveform_correlation,
    respiratory_rate_errors,
    seed_everything,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-mat", type=Path, default=Path("data_bidmc/bidmc_data.mat"))
    parser.add_argument("--pretrained-capnobase", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("correncoder_epoch0_results"))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=55)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--matched-max-patches", type=int, default=50_000)
    parser.add_argument("--matched-deep-max-patches", type=int, default=10_000)
    parser.add_argument("--matched-shrinkage", type=float, default=0.1)
    parser.add_argument("--matched-covariance-rank", type=int, default=16)
    parser.add_argument("--calibration-min-gain", type=float, default=0.05)
    return parser.parse_args()


def mean_ci(values):
    values = np.asarray(values, dtype=np.float64)
    half = float(student_t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / math.sqrt(len(values)))
    return float(values.mean()), half


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    subjects = load_bidmc_subjects(args.data_mat)
    pretrained_state = torch.load(args.pretrained_capnobase, map_location="cpu", weights_only=True)
    rows = []
    for fold in range(len(subjects)):
        train_x, train_y, test_x, test_y, starts = build_fold_tensors(subjects, fold)
        loader = DataLoader(
            TensorDataset(test_x, test_y),
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=device.type == "cuda",
        )
        methods = (
            "random",
            "pretrained",
            "matched_1d",
            "matched_1d_calibrated",
            "matched_1d_safe_calibration",
            "matched_depth2_diagonal_calibrated",
            "matched_layerwise",
            "matched_layerwise_calibrated",
            "matched_layerwise_lowrank",
            "matched_layerwise_lowrank_calibrated",
            "matched_layerwise_lowrank_safe_calibration",
        )
        for method in methods:
            fold_seed = args.seed + fold
            seed_everything(fold_seed)
            model = PublishedCorrEncoder1D(dropout=0.5).to(device)
            if method == "pretrained":
                model.load_state_dict(pretrained_state)
            elif method.startswith("matched_1d"):
                initialise_1d_matched_filter(
                    model,
                    train_x,
                    train_y,
                    max_patches=args.matched_max_patches,
                    shrinkage=args.matched_shrinkage,
                    seed=fold_seed,
                )
            elif method.startswith("matched_depth2"):
                initialise_1d_layerwise_matched_filter(
                    model,
                    train_x,
                    train_y,
                    stem_max_patches=args.matched_max_patches,
                    deep_max_patches=args.matched_deep_max_patches,
                    shrinkage=args.matched_shrinkage,
                    depth=2,
                    seed=fold_seed,
                )
            elif method.startswith("matched_layerwise"):
                initialise_1d_layerwise_matched_filter(
                    model,
                    train_x,
                    train_y,
                    stem_max_patches=args.matched_max_patches,
                    deep_max_patches=args.matched_deep_max_patches,
                    shrinkage=args.matched_shrinkage,
                    deep_covariance=(
                        "lowrank" if "lowrank" in method else "diagonal"
                    ),
                    covariance_rank=args.matched_covariance_rank,
                    seed=fold_seed,
                )
            if method.endswith("_calibrated"):
                calibrate_regression_output(model, train_x, train_y)
            elif method.endswith("_safe_calibration"):
                fit_safe_calibration_head(
                    model,
                    train_x,
                    train_y,
                    minimum_absolute_gain=args.calibration_min_gain,
                )
            evaluation = _evaluate(model, loader, device)
            prediction = overlap_average(evaluation["prediction"], starts)
            target = overlap_average(evaluation["target"], starts)
            rr_errors = respiratory_rate_errors(prediction, target)
            max_lag_correlation, best_lag = lag_robust_waveform_correlation(
                prediction, target
            )
            rows.append(
                {
                    "method": method,
                    "fold": fold,
                    "test_subject": fold + 1,
                    "mse": evaluation["mse"],
                    "mae": evaluation["mae"],
                    "waveform_correlation": lag_robust_waveform_correlation(
                        prediction, target, max_lag_samples=0
                    )[0],
                    "max_lag_waveform_correlation": max_lag_correlation,
                    "best_waveform_lag_samples": best_lag,
                    "rr_mean_absolute_error_bpm_30p6s": float(np.mean(rr_errors)),
                    "device": str(device),
                    "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                }
            )
        print(f"epoch0 fold={fold:02d}/52 complete")

    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / "fold_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    aggregate = []
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        for metric in (
            "mse",
            "mae",
            "waveform_correlation",
            "max_lag_waveform_correlation",
            "rr_mean_absolute_error_bpm_30p6s",
        ):
            mean, ci95 = mean_ci([row[metric] for row in selected])
            aggregate.append({"method": method, "metric": metric, "mean": mean, "ci95_half_width": ci95})
    (args.output_root / "aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
