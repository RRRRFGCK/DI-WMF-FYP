"""Paired subject-level analysis for the extended Correncoder Epoch-0 study."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t


METRICS = {
    "mse": "lower",
    "mae": "lower",
    "waveform_correlation": "higher",
    "max_lag_waveform_correlation": "higher",
    "rr_mean_absolute_error_bpm_30p6s": "lower",
}


def mean_ci(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    half = float(
        student_t.ppf(0.975, len(values) - 1)
        * values.std(ddof=1)
        / math.sqrt(len(values))
    )
    return mean, half


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    args = parser.parse_args()
    with (args.results_root / "fold_metrics.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    by_method = {}
    for row in rows:
        by_method.setdefault(row["method"], {})[int(row["test_subject"])] = row
    comparisons = (
        ("pretrained", "random"),
        ("matched_1d_calibrated", "matched_1d"),
        ("matched_layerwise_calibrated", "matched_layerwise"),
        ("matched_layerwise_calibrated", "matched_1d_calibrated"),
    )
    paired = []
    for method, reference in comparisons:
        common = sorted(set(by_method[method]) & set(by_method[reference]))
        for metric, direction in METRICS.items():
            values = np.array([float(by_method[method][s][metric]) for s in common])
            baseline = np.array([float(by_method[reference][s][metric]) for s in common])
            delta = values - baseline
            improvement = delta if direction == "higher" else -delta
            delta_mean, delta_ci = mean_ci(delta)
            improvement_mean, improvement_ci = mean_ci(improvement)
            paired.append(
                {
                    "method": method,
                    "reference": reference,
                    "metric": metric,
                    "n": len(common),
                    "method_minus_reference": delta_mean,
                    "delta_ci95_half_width": delta_ci,
                    "oriented_improvement": improvement_mean,
                    "improvement_ci95_half_width": improvement_ci,
                    "ci_excludes_zero": abs(improvement_mean) > improvement_ci,
                    "subjects_improved": int((improvement > 0).sum()),
                }
            )
    with (args.results_root / "paired_comparisons.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    (args.results_root / "paired_comparisons.json").write_text(
        json.dumps(paired, indent=2), encoding="utf-8"
    )
    print(json.dumps(paired, indent=2))


if __name__ == "__main__":
    main()
