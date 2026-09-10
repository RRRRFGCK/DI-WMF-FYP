"""Reproduce the published 1-D Correncoder on paired PPG/respiration data.

The image classifiers called ``correncoder`` elsewhere in this repository are
controlled classification adaptations.  This script is deliberately separate:
it uses the authors' six-layer regression topology and the paper's 30-Hz,
288-sample, leave-one-subject-out protocol on the public BIDMC data.

The original headline BIDMC experiment first trained on CapnoBase and then
fine-tuned on BIDMC.  Unless a compatible CapnoBase checkpoint is supplied,
this script reports a BIDMC-only architecture/protocol reproduction and labels
it as such in every output file.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import h5py
from scipy.io import loadmat
from scipy.signal import resample_poly
from scipy.stats import t as student_t
from torch.utils.data import DataLoader, TensorDataset

from domain_mf.models import PublishedCorrEncoder1D
from domain_mf.regression import (
    calibrate_regression_output,
    correncoder_regression_loss,
    fit_safe_calibration_head,
    initialise_1d_matched_filter,
    initialise_1d_layerwise_matched_filter,
)


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Published PPG-to-respiration Correncoder reproduction"
    )
    parser.add_argument("--dataset", choices=["bidmc", "capnobase"], default="bidmc")
    parser.add_argument(
        "--data-mat", type=Path, default=ROOT / "data_bidmc" / "bidmc_data.mat"
    )
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "outputs_correncoder_regression"
    )
    parser.add_argument(
        "--capnobase-root", type=Path, default=ROOT / "data_capnobase"
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=None,
        help="Zero-based test-subject folds; omitted means all 53 LOSO folds.",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=55)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--pretrained-capnobase",
        type=Path,
        default=None,
        help="Optional compatible CapnoBase checkpoint before BIDMC fine-tuning.",
    )
    parser.add_argument(
        "--initialisation",
        choices=["auto", "random", "pretrained", "matched_1d", "matched_layerwise"],
        default="auto",
        help=(
            "Initialisation under test. 'auto' preserves the previous behaviour: "
            "pretrained when a checkpoint is supplied, otherwise random."
        ),
    )
    parser.add_argument("--matched-max-patches", type=int, default=50_000)
    parser.add_argument("--matched-deep-max-patches", type=int, default=10_000)
    parser.add_argument("--matched-shrinkage", type=float, default=0.1)
    parser.add_argument("--matched-depth", type=int, choices=[1, 2, 3], default=3)
    parser.add_argument(
        "--matched-deep-covariance",
        choices=["diagonal", "lowrank"],
        default="diagonal",
    )
    parser.add_argument("--matched-covariance-rank", type=int, default=16)
    parser.add_argument("--affine-calibration", action="store_true")
    parser.add_argument("--safe-calibration-head", action="store_true")
    parser.add_argument("--calibration-min-gain", type=float, default=0.05)
    parser.add_argument("--calibration-max-gain", type=float, default=10.0)
    parser.add_argument("--calibration-warmup-epochs", type=int, default=3)
    parser.add_argument("--correlation-lambda", type=float, default=0.0)
    parser.add_argument(
        "--correlation-mode",
        choices=["zero_lag", "max_lag", "soft_lag"],
        default="zero_lag",
    )
    parser.add_argument("--max-lag-samples", type=int, default=30)
    parser.add_argument("--lag-step", type=int, default=3)
    parser.add_argument("--lag-temperature", type=float, default=0.05)
    parser.add_argument("--spectral-lambda", type=float, default=0.0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed fold metrics in the selected output directory.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _finite_signal(values) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("Signal contains no finite samples")
    if not finite.all():
        positions = np.arange(values.size)
        values[~finite] = np.interp(positions[~finite], positions[finite], values[finite])
    return values


def _normalise_pair(ppg, respiration) -> tuple[np.ndarray, np.ndarray]:
    ppg = _finite_signal(ppg)
    respiration = _finite_signal(respiration)
    ppg = (ppg - ppg.mean()) / max(ppg.std(), 1e-8)
    low = float(respiration.min())
    high = float(respiration.max())
    respiration = (respiration - low) / max(high - low, 1e-8)
    return ppg.astype(np.float32), respiration.astype(np.float32)


def load_bidmc_subjects(path: Path, target_hz: int = 30) -> list[dict]:
    contents = loadmat(path, simplify_cells=True)
    subjects = []
    for subject_index, record in enumerate(contents["data"]):
        ppg_record = record["ppg"]
        respiration_record = record["ref"]["resp_sig"]["imp"]
        ppg_hz = int(round(float(ppg_record["fs"])))
        respiration_hz = int(round(float(respiration_record["fs"])))
        ppg = resample_poly(_finite_signal(ppg_record["v"]), target_hz, ppg_hz)
        respiration = resample_poly(
            _finite_signal(respiration_record["v"]), target_hz, respiration_hz
        )
        usable = min(ppg.size, respiration.size)
        # The paper uses 50 non-overlapping 9.6-second windows per subject.
        usable = min(usable, 50 * PublishedCorrEncoder1D.input_length)
        ppg, respiration = _normalise_pair(ppg[:usable], respiration[:usable])
        subjects.append(
            {
                "subject": subject_index,
                "ppg": ppg,
                "respiration": respiration,
                "sampling_hz": target_hz,
            }
        )
    return subjects


def load_capnobase_subjects(path: Path, target_hz: int = 30) -> list[dict]:
    subjects = []
    for subject_index, mat_path in enumerate(sorted(path.glob("*_8min.mat"))):
        with h5py.File(mat_path, "r") as contents:
            source_hz = int(
                round(float(np.asarray(contents["param/samplingrate/pleth"])[0, 0]))
            )
            ppg = np.asarray(contents["signal/pleth/y"])[0]
            respiration = np.asarray(contents["signal/co2/y"])[0]
        ppg = resample_poly(_finite_signal(ppg), target_hz, source_hz)
        respiration = resample_poly(
            _finite_signal(respiration), target_hz, source_hz
        )
        usable = min(ppg.size, respiration.size, 50 * PublishedCorrEncoder1D.input_length)
        ppg, respiration = _normalise_pair(ppg[:usable], respiration[:usable])
        subjects.append(
            {
                "subject": subject_index,
                "source_id": mat_path.stem,
                "ppg": ppg,
                "respiration": respiration,
                "sampling_hz": target_hz,
            }
        )
    if len(subjects) != 42:
        raise RuntimeError(f"Expected 42 CapnoBase records, found {len(subjects)} in {path}")
    return subjects


def _non_overlapping_segments(signal: np.ndarray, length: int = 288) -> np.ndarray:
    count = signal.size // length
    return signal[: count * length].reshape(count, length)


def _sliding_segments(
    signal: np.ndarray, length: int = 288, stride: int = 30
) -> tuple[np.ndarray, np.ndarray]:
    starts = np.arange(0, signal.size - length + 1, stride, dtype=np.int64)
    segments = np.stack([signal[start : start + length] for start in starts])
    return segments, starts


def build_fold_tensors(subjects: list[dict], test_fold: int):
    train_ppg = []
    train_respiration = []
    for index, subject in enumerate(subjects):
        if index == test_fold:
            continue
        train_ppg.append(_non_overlapping_segments(subject["ppg"]))
        train_respiration.append(_non_overlapping_segments(subject["respiration"]))
    train_x = torch.from_numpy(np.concatenate(train_ppg)[:, None, :])
    train_y = torch.from_numpy(np.concatenate(train_respiration)[:, None, :])
    test_subject = subjects[test_fold]
    test_x, starts = _sliding_segments(test_subject["ppg"])
    test_y, target_starts = _sliding_segments(test_subject["respiration"])
    if not np.array_equal(starts, target_starts):
        raise RuntimeError("PPG and respiration windows are misaligned")
    return (
        train_x,
        train_y,
        torch.from_numpy(test_x[:, None, :]),
        torch.from_numpy(test_y[:, None, :]),
        starts,
    )


def overlap_average(windows: np.ndarray, starts: np.ndarray) -> np.ndarray:
    output_length = int(starts[-1]) + windows.shape[-1]
    total = np.zeros(output_length, dtype=np.float64)
    count = np.zeros(output_length, dtype=np.float64)
    for window, start in zip(windows, starts):
        stop = int(start) + window.size
        total[int(start) : stop] += window
        count[int(start) : stop] += 1.0
    return total / np.maximum(count, 1.0)


def respiratory_rate(signal: np.ndarray, sampling_hz: int = 30) -> float:
    centred = signal - signal.mean()
    frequencies = np.fft.rfftfreq(centred.size, d=1.0 / sampling_hz)
    power = np.abs(np.fft.rfft(centred)) ** 2
    respiratory_band = (frequencies >= 0.08) & (frequencies <= 0.8)
    if not respiratory_band.any():
        return float("nan")
    band_indices = np.flatnonzero(respiratory_band)
    peak = band_indices[int(np.argmax(power[respiratory_band]))]
    return float(60.0 * frequencies[peak])


def respiratory_rate_errors(
    prediction: np.ndarray,
    target: np.ndarray,
    sampling_hz: int = 30,
    window_seconds: float = 30.6,
    stride_seconds: float = 1.0,
) -> np.ndarray:
    length = int(round(window_seconds * sampling_hz))
    stride = int(round(stride_seconds * sampling_hz))
    errors = []
    for start in range(0, min(prediction.size, target.size) - length + 1, stride):
        predicted_rate = respiratory_rate(
            prediction[start : start + length], sampling_hz
        )
        target_rate = respiratory_rate(target[start : start + length], sampling_hz)
        errors.append(abs(predicted_rate - target_rate))
    return np.asarray(errors, dtype=np.float64)


def lag_robust_waveform_correlation(
    prediction: np.ndarray,
    target: np.ndarray,
    max_lag_samples: int = 30,
    lag_step: int = 1,
) -> tuple[float, int]:
    """Maximum fixed-overlap Pearson correlation and its target lag."""

    def safe_correlation(first: np.ndarray, second: np.ndarray) -> float:
        first = first - first.mean()
        second = second - second.mean()
        denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
        if not np.isfinite(denominator) or denominator <= np.finfo(np.float64).eps:
            return 0.0
        return float(np.dot(first, second) / denominator)

    length = min(prediction.size, target.size)
    if max_lag_samples < 0 or lag_step < 1 or 2 * max_lag_samples >= length:
        raise ValueError("Invalid lag range for waveform length")
    if max_lag_samples == 0:
        return safe_correlation(prediction[:length], target[:length]), 0
    prediction_core = prediction[max_lag_samples : length - max_lag_samples]
    if float(np.linalg.norm(prediction_core - prediction_core.mean())) <= np.finfo(
        np.float64
    ).eps:
        return 0.0, 0
    correlations = []
    lags = list(range(-max_lag_samples, max_lag_samples + 1, lag_step))
    for lag in lags:
        target_core = target[
            max_lag_samples + lag : length - max_lag_samples + lag
        ]
        correlations.append(safe_correlation(prediction_core, target_core))
    best = int(np.argmax(correlations))
    return correlations[best], lags[best]


def _evaluate(model, loader, device):
    model.eval()
    squared_error = 0.0
    absolute_error = 0.0
    count = 0
    predictions = []
    targets = []
    with torch.no_grad():
        for inputs, reference in loader:
            inputs = inputs.to(device, non_blocking=device.type == "cuda")
            reference = reference.to(device, non_blocking=device.type == "cuda")
            output = model(inputs)
            squared_error += float(F.mse_loss(output, reference, reduction="sum"))
            absolute_error += float(F.l1_loss(output, reference, reduction="sum"))
            count += reference.numel()
            predictions.append(output.cpu())
            targets.append(reference.cpu())
    return {
        "mse": squared_error / count,
        "mae": absolute_error / count,
        "prediction": torch.cat(predictions).numpy()[:, 0],
        "target": torch.cat(targets).numpy()[:, 0],
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_fold(args, subjects, fold, device):
    fold_seed = args.seed + fold
    seed_everything(fold_seed)
    train_x, train_y, test_x, test_y, starts = build_fold_tensors(subjects, fold)
    generator = torch.Generator().manual_seed(fold_seed)
    common = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        TensorDataset(train_x, train_y), shuffle=True, generator=generator, **common
    )
    test_loader = DataLoader(TensorDataset(test_x, test_y), shuffle=False, **common)
    model = PublishedCorrEncoder1D(dropout=0.5).to(device)
    initialisation_start = time.perf_counter()
    initialisation_report = {"method": args.initialisation}
    if args.initialisation == "pretrained":
        state = torch.load(
            args.pretrained_capnobase, map_location="cpu", weights_only=True
        )
        model.load_state_dict(state)
        initialisation_report["checkpoint"] = str(
            args.pretrained_capnobase.resolve()
        )
    elif args.initialisation == "matched_1d":
        report = initialise_1d_matched_filter(
            model,
            train_x,
            train_y,
            max_patches=args.matched_max_patches,
            shrinkage=args.matched_shrinkage,
            seed=fold_seed,
        )
        initialisation_report = report.to_dict()
    elif args.initialisation == "matched_layerwise":
        initialisation_report = initialise_1d_layerwise_matched_filter(
            model,
            train_x,
            train_y,
            stem_max_patches=args.matched_max_patches,
            deep_max_patches=args.matched_deep_max_patches,
            shrinkage=args.matched_shrinkage,
            depth=args.matched_depth,
            deep_covariance=args.matched_deep_covariance,
            covariance_rank=args.matched_covariance_rank,
            seed=fold_seed,
        )
    if args.affine_calibration:
        initialisation_report["affine_calibration"] = calibrate_regression_output(
            model, train_x, train_y
        ).to_dict()
    if args.safe_calibration_head:
        initialisation_report["safe_calibration_head"] = fit_safe_calibration_head(
            model,
            train_x,
            train_y,
            minimum_absolute_gain=args.calibration_min_gain,
            maximum_absolute_gain=args.calibration_max_gain,
        ).to_dict()
    initialisation_seconds = time.perf_counter() - initialisation_start
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    calibration_parameters = (
        set(model.output_calibration.parameters())
        if model.output_calibration is not None
        else set()
    )
    base_parameters = [
        parameter for parameter in model.parameters() if parameter not in calibration_parameters
    ]
    history = []
    start_time = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        base_frozen = bool(
            args.safe_calibration_head and epoch <= args.calibration_warmup_epochs
        )
        for parameter in base_parameters:
            parameter.requires_grad_(not base_frozen)
        model.train()
        total_objective = 0.0
        total_mse = 0.0
        total_correlation = 0.0
        total_spectral = 0.0
        examples = 0
        for inputs, reference in train_loader:
            inputs = inputs.to(device, non_blocking=device.type == "cuda")
            reference = reference.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            output = model(inputs)
            loss, components = correncoder_regression_loss(
                output,
                reference,
                correlation_lambda=args.correlation_lambda,
                spectral_lambda=args.spectral_lambda,
                correlation_mode=args.correlation_mode,
                max_lag_samples=args.max_lag_samples,
                lag_step=args.lag_step,
                lag_temperature=args.lag_temperature,
            )
            loss.backward()
            optimizer.step()
            batch_examples = reference.shape[0]
            total_objective += float(loss.detach()) * batch_examples
            total_mse += float(components["mse"].detach()) * batch_examples
            total_correlation += (
                float(components["correlation_loss"].detach()) * batch_examples
            )
            total_spectral += (
                float(components["spectral_loss"].detach()) * batch_examples
            )
            examples += batch_examples
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        history.append(
            {
                "epoch": epoch,
                "train_objective": total_objective / examples,
                "train_mse": total_mse / examples,
                "train_correlation_loss": total_correlation / examples,
                "train_spectral_loss": total_spectral / examples,
                "base_frozen": int(base_frozen),
                "elapsed_seconds": time.perf_counter() - start_time,
            }
        )
        print(
            f"fold={fold:02d} epoch={epoch:03d}/{args.epochs} "
            f"objective={total_objective / examples:.6f} "
            f"mse={total_mse / examples:.6f}"
        )

    # The held-out subject is the test set in LOSO. Evaluate it once after
    # training rather than once per epoch, so the test subject cannot influence
    # epoch selection and the reported protocol remains leakage-free.
    final_evaluation = _evaluate(model, test_loader, device)
    predicted_waveform = overlap_average(final_evaluation["prediction"], starts)
    target_waveform = overlap_average(final_evaluation["target"], starts)
    rr_errors = respiratory_rate_errors(predicted_waveform, target_waveform)
    correlation = lag_robust_waveform_correlation(
        predicted_waveform, target_waveform, max_lag_samples=0
    )[0]
    max_lag_correlation, best_lag = lag_robust_waveform_correlation(
        predicted_waveform,
        target_waveform,
        max_lag_samples=args.max_lag_samples,
        lag_step=1,
    )
    fold_dir = args.output_root / f"{args.dataset}_fold{fold:02d}_seed{fold_seed}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "fold": fold,
        "seed": fold_seed,
        "test_subject": fold + 1,
        "protocol": (
            f"{args.dataset}_loso_{args.initialisation}_"
            f"corr{args.correlation_lambda:g}_{args.correlation_mode}_"
            f"spec{args.spectral_lambda:g}_affine{int(args.affine_calibration)}_"
            f"safehead{int(args.safe_calibration_head)}"
        ),
        "initialisation": args.initialisation,
        "initialisation_seconds": initialisation_seconds,
        "initialisation_report": initialisation_report,
        "affine_calibration": args.affine_calibration,
        "safe_calibration_head": args.safe_calibration_head,
        "calibration_warmup_epochs": args.calibration_warmup_epochs,
        "final_calibration_gain": (
            float(model.output_calibration.gain().detach())
            if model.output_calibration is not None
            else None
        ),
        "final_calibration_offset": (
            float(model.output_calibration.offset.detach())
            if model.output_calibration is not None
            else None
        ),
        "correlation_lambda": args.correlation_lambda,
        "correlation_mode": args.correlation_mode,
        "max_lag_samples": args.max_lag_samples,
        "lag_step": args.lag_step,
        "lag_temperature": args.lag_temperature,
        "spectral_lambda": args.spectral_lambda,
        "test_mse": final_evaluation["mse"],
        "test_mae": final_evaluation["mae"],
        "waveform_correlation": correlation,
        "max_lag_waveform_correlation": max_lag_correlation,
        "best_waveform_lag_samples": best_lag,
        "rr_median_absolute_error_bpm_30p6s": float(np.median(rr_errors)),
        "rr_mean_absolute_error_bpm_30p6s": float(np.mean(rr_errors)),
        "rr_windows": int(rr_errors.size),
        "train_segments": int(train_x.shape[0]),
        "test_segments": int(test_x.shape[0]),
        "training_seconds": history[-1]["elapsed_seconds"],
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
    }
    _write_csv(fold_dir / "history.csv", history)
    with (fold_dir / "final_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    np.savez_compressed(
        fold_dir / "waveforms.npz",
        prediction=predicted_waveform,
        target=target_waveform,
        rr_absolute_errors=rr_errors,
    )
    torch.save(model.state_dict(), fold_dir / "checkpoint_final.pt")
    return metrics


def _mean_ci(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    ci95 = (
        float(student_t.ppf(0.975, values.size - 1) * values.std(ddof=1) / math.sqrt(values.size))
        if values.size > 1
        else 0.0
    )
    return mean, ci95


def main():
    args = parse_args()
    if args.initialisation == "auto":
        args.initialisation = (
            "pretrained" if args.pretrained_capnobase is not None else "random"
        )
    if args.initialisation == "pretrained" and args.pretrained_capnobase is None:
        raise ValueError("--initialisation pretrained requires --pretrained-capnobase")
    if args.initialisation != "pretrained" and args.pretrained_capnobase is not None:
        raise ValueError(
            "--pretrained-capnobase is only valid with --initialisation pretrained"
        )
    if args.correlation_lambda < 0.0 or args.spectral_lambda < 0.0:
        raise ValueError("Loss weights must be non-negative")
    if args.max_lag_samples < 0 or args.lag_step < 1:
        raise ValueError("Lag settings must be non-negative/positive")
    if args.lag_temperature <= 0.0:
        raise ValueError("--lag-temperature must be positive")
    if args.affine_calibration and args.safe_calibration_head:
        raise ValueError("Choose either absorbed affine calibration or the safe head")
    if args.calibration_warmup_epochs < 0:
        raise ValueError("--calibration-warmup-epochs must be non-negative")
    if not 0.0 < args.calibration_min_gain < args.calibration_max_gain:
        raise ValueError("Calibration gain bounds must satisfy 0 < min < max")
    if args.matched_covariance_rank < 1:
        raise ValueError("--matched-covariance-rank must be positive")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    subjects = (
        load_bidmc_subjects(args.data_mat)
        if args.dataset == "bidmc"
        else load_capnobase_subjects(args.capnobase_root)
    )
    folds = list(range(len(subjects))) if args.folds is None else args.folds
    invalid = [fold for fold in folds if not 0 <= fold < len(subjects)]
    if invalid:
        raise ValueError(f"Invalid folds {invalid}; dataset contains {len(subjects)} subjects")
    args.output_root.mkdir(parents=True, exist_ok=True)
    configuration = {
        "dataset": args.dataset,
        "folds": folds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "initialisation": args.initialisation,
        "pretrained_capnobase": (
            str(args.pretrained_capnobase.resolve())
            if args.pretrained_capnobase is not None
            else None
        ),
        "matched_max_patches": args.matched_max_patches,
        "matched_deep_max_patches": args.matched_deep_max_patches,
        "matched_shrinkage": args.matched_shrinkage,
        "matched_depth": args.matched_depth,
        "matched_deep_covariance": args.matched_deep_covariance,
        "matched_covariance_rank": args.matched_covariance_rank,
        "affine_calibration": args.affine_calibration,
        "safe_calibration_head": args.safe_calibration_head,
        "calibration_min_gain": args.calibration_min_gain,
        "calibration_max_gain": args.calibration_max_gain,
        "calibration_warmup_epochs": args.calibration_warmup_epochs,
        "correlation_lambda": args.correlation_lambda,
        "correlation_mode": args.correlation_mode,
        "max_lag_samples": args.max_lag_samples,
        "lag_step": args.lag_step,
        "lag_temperature": args.lag_temperature,
        "spectral_lambda": args.spectral_lambda,
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        ),
    }
    with (args.output_root / "experiment_config.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(configuration, handle, indent=2)
    rows = []
    for fold in folds:
        fold_seed = args.seed + fold
        metrics_path = (
            args.output_root
            / f"{args.dataset}_fold{fold:02d}_seed{fold_seed}"
            / "final_metrics.json"
        )
        if args.resume and metrics_path.exists():
            with metrics_path.open(encoding="utf-8") as handle:
                rows.append(json.load(handle))
            print(f"fold={fold:02d} reused completed metrics")
        else:
            rows.append(run_fold(args, subjects, fold, device))
    _write_csv(args.output_root / "fold_metrics.csv", rows)
    aggregate = {
        "folds_completed": len(rows),
        "folds_expected_for_full_loso": len(subjects),
        "initialisation": args.initialisation,
        "correlation_lambda": args.correlation_lambda,
        "spectral_lambda": args.spectral_lambda,
        "confidence_interval": "two-sided 95% Student-t interval across LOSO folds",
    }
    for metric in (
        "test_mse",
        "test_mae",
        "waveform_correlation",
        "max_lag_waveform_correlation",
        "rr_median_absolute_error_bpm_30p6s",
        "rr_mean_absolute_error_bpm_30p6s",
        "training_seconds",
    ):
        mean, ci95 = _mean_ci([row[metric] for row in rows])
        aggregate[f"{metric}_mean"] = mean
        aggregate[f"{metric}_ci95"] = ci95
        aggregate[f"{metric}_median"] = float(np.median([row[metric] for row in rows]))
    with (args.output_root / "aggregate.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2)
    print(json.dumps(aggregate, indent=2))
    print(f"Results: {args.output_root.resolve()}")


if __name__ == "__main__":
    main()
