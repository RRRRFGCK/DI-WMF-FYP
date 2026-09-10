from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

METRICS = (
    "test_mse",
    "test_mae",
    "waveform_correlation",
    "rr_median_absolute_error_bpm_30p6s",
    "rr_mean_absolute_error_bpm_30p6s",
    "training_seconds",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Reaggregate Correncoder LOSO fold metrics")
    parser.add_argument("--results-root", type=Path, default=Path("outputs_correncoder_regression_full"))
    return parser.parse_args()


def mean_ci(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    if values.size < 2:
        return mean, 0.0
    critical = float(student_t.ppf(0.975, values.size - 1))
    return mean, float(critical * values.std(ddof=1) / math.sqrt(values.size))


def main():
    args = parse_args()
    with (args.results_root / "fold_metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("No fold metrics found")
    aggregate = {
        "folds_completed": len(rows),
        "folds_expected_for_full_loso": 53,
        "confidence_interval": "two-sided 95% Student-t interval across LOSO folds",
    }
    for metric in METRICS:
        values = [float(row[metric]) for row in rows]
        mean, ci95 = mean_ci(values)
        aggregate[f"{metric}_mean"] = mean
        aggregate[f"{metric}_ci95"] = ci95
        aggregate[f"{metric}_median"] = float(np.median(values))
    with (args.results_root / "aggregate.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
