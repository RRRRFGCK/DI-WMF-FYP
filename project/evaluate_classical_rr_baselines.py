"""Classical PPG respiratory-rate baselines on the BIDMC LOSO test subjects."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt
from scipy.stats import t as student_t
from scipy.stats import ttest_rel

from run_correncoder_regression import load_bidmc_subjects, respiratory_rate


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-mat", type=Path, default=ROOT / "data_bidmc" / "bidmc_data.mat")
    parser.add_argument(
        "--correncoder-fold-metrics",
        type=Path,
        default=ROOT / "outputs_correncoder_regression_full" / "fold_metrics.csv",
    )
    parser.add_argument("--output-root", type=Path, default=ROOT / "classical_rr_results")
    return parser.parse_args()


def bandpass(values: np.ndarray, sampling_hz: int, low: float, high: float) -> np.ndarray:
    sos = butter(4, [low, high], btype="bandpass", fs=sampling_hz, output="sos")
    return sosfiltfilt(sos, values).astype(np.float64)


def respiratory_envelope(ppg: np.ndarray, sampling_hz: int) -> np.ndarray:
    cardiac = bandpass(ppg, sampling_hz, 0.8, min(4.0, 0.45 * sampling_hz))
    envelope = np.abs(hilbert(cardiac))
    return bandpass(envelope, sampling_hz, 0.08, 0.8)


def autocorrelation_rate(values: np.ndarray, sampling_hz: int) -> float:
    values = values - values.mean()
    scale = float(np.linalg.norm(values))
    if not np.isfinite(scale) or scale <= np.finfo(np.float64).eps:
        return float("nan")
    correlation = np.correlate(values, values, mode="full")[values.size - 1 :]
    minimum_lag = max(1, int(math.floor(sampling_hz / 0.8)))
    maximum_lag = min(values.size - 1, int(math.ceil(sampling_hz / 0.08)))
    lag = minimum_lag + int(np.argmax(correlation[minimum_lag : maximum_lag + 1]))
    return float(60.0 * sampling_hz / lag)


def subject_errors(ppg, target, sampling_hz, method, window_seconds=30.6, stride_seconds=1.0):
    length = int(round(window_seconds * sampling_hz))
    stride = int(round(stride_seconds * sampling_hz))
    if method == "bandpass_fft":
        proxy = bandpass(ppg, sampling_hz, 0.08, 0.8)
        estimator = lambda window: respiratory_rate(window, sampling_hz)
    elif method == "envelope_fft":
        proxy = respiratory_envelope(ppg, sampling_hz)
        estimator = lambda window: respiratory_rate(window, sampling_hz)
    elif method == "bandpass_autocorrelation":
        proxy = bandpass(ppg, sampling_hz, 0.08, 0.8)
        estimator = lambda window: autocorrelation_rate(window, sampling_hz)
    else:
        raise ValueError(method)
    errors = []
    for start in range(0, min(proxy.size, target.size) - length + 1, stride):
        predicted = estimator(proxy[start : start + length])
        reference = respiratory_rate(target[start : start + length], sampling_hz)
        if np.isfinite(predicted) and np.isfinite(reference):
            errors.append(abs(predicted - reference))
    return np.asarray(errors, dtype=np.float64)


def mean_ci(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    half = float(student_t.ppf(0.975, values.size - 1) * values.std(ddof=1) / math.sqrt(values.size))
    return mean, mean - half, mean + half


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    subjects = load_bidmc_subjects(args.data_mat)
    methods = ("bandpass_fft", "envelope_fft", "bandpass_autocorrelation")
    rows = []
    for subject in subjects:
        for method in methods:
            errors = subject_errors(
                subject["ppg"],
                subject["respiration"],
                subject["sampling_hz"],
                method,
            )
            rows.append(
                {
                    "fold": subject["subject"],
                    "test_subject": subject["subject"] + 1,
                    "method": method,
                    "rr_windows": errors.size,
                    "rr_mean_absolute_error_bpm_30p6s": float(errors.mean()),
                    "rr_median_absolute_error_bpm_30p6s": float(np.median(errors)),
                }
            )
    aggregate_rows = []
    for method in methods:
        subset = [row for row in rows if row["method"] == method]
        for metric in (
            "rr_mean_absolute_error_bpm_30p6s",
            "rr_median_absolute_error_bpm_30p6s",
        ):
            values = np.array([float(row[metric]) for row in subset])
            mean, low, high = mean_ci(values)
            aggregate_rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "subjects": values.size,
                    "mean": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )

    with args.correncoder_fold_metrics.open(newline="", encoding="utf-8") as handle:
        corr_rows = {int(row["fold"]): row for row in csv.DictReader(handle)}
    paired_rows = []
    metric = "rr_mean_absolute_error_bpm_30p6s"
    for method in methods:
        subset = {int(row["fold"]): row for row in rows if row["method"] == method}
        folds = sorted(set(subset) & set(corr_rows))
        classical = np.array([float(subset[fold][metric]) for fold in folds])
        correncoder = np.array([float(corr_rows[fold][metric]) for fold in folds])
        improvement = classical - correncoder
        mean, low, high = mean_ci(improvement)
        paired_rows.append(
            {
                "comparison": f"correncoder_improvement_over_{method}",
                "metric": metric,
                "paired_subjects": len(folds),
                "classical_mean": float(classical.mean()),
                "correncoder_mean": float(correncoder.mean()),
                "mean_error_reduction": mean,
                "ci95_low": low,
                "ci95_high": high,
                "p_value_raw": float(ttest_rel(classical, correncoder).pvalue),
                "correncoder_wins": int((correncoder < classical).sum()),
            }
        )

    write_csv(args.output_root / "classical_rr_subject_rows.csv", rows)
    write_csv(args.output_root / "classical_rr_aggregate.csv", aggregate_rows)
    write_csv(args.output_root / "classical_rr_paired.csv", paired_rows)
    for row in aggregate_rows:
        if row["metric"].startswith("rr_mean"):
            print(row)
    for row in paired_rows:
        print(row)


if __name__ == "__main__":
    main()
