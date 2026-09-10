"""Aggregate and pair Correncoder initialisation/loss ablations by BIDMC subject."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t


METRICS = {
    "test_mse": "lower",
    "test_mae": "lower",
    "waveform_correlation": "higher",
    "rr_mean_absolute_error_bpm_30p6s": "lower",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Repeat once per experiment.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def mean_ci(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, 0.0
    half = float(student_t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / math.sqrt(len(values)))
    return mean, half


def load_rows(path):
    with (path / "fold_metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No fold metrics in {path}")
    return {int(row["test_subject"]): row for row in rows}


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    experiments = {}
    for item in args.experiment:
        label, raw_path = item.split("=", 1)
        experiments[label] = load_rows(Path(raw_path))

    aggregate_rows = []
    for label, rows in experiments.items():
        for metric in METRICS:
            values = [float(row[metric]) for row in rows.values()]
            mean, ci95 = mean_ci(values)
            aggregate_rows.append(
                {
                    "experiment": label,
                    "metric": metric,
                    "n_subjects": len(values),
                    "mean": mean,
                    "ci95_half_width": ci95,
                    "median": float(np.median(values)),
                }
            )

    # The two baselines answer different causal questions.
    comparisons = [
        ("pretrained_mse", "random_mse"),
        ("matched_1d_mse", "random_mse"),
        ("pretrained_corr", "pretrained_mse"),
        ("pretrained_spectral", "pretrained_mse"),
        ("pretrained_corr_spectral", "pretrained_mse"),
        ("pretrained_corr_spectral", "pretrained_corr"),
    ]
    paired_rows = []
    for label, baseline in comparisons:
        if label not in experiments or baseline not in experiments:
            continue
        common = sorted(set(experiments[label]) & set(experiments[baseline]))
        for metric, direction in METRICS.items():
            method = np.array([float(experiments[label][s][metric]) for s in common])
            base = np.array([float(experiments[baseline][s][metric]) for s in common])
            raw_delta = method - base
            improvement = raw_delta if direction == "higher" else -raw_delta
            delta_mean, delta_ci = mean_ci(raw_delta)
            imp_mean, imp_ci = mean_ci(improvement)
            paired_rows.append(
                {
                    "experiment": label,
                    "reference": baseline,
                    "metric": metric,
                    "n_paired_subjects": len(common),
                    "method_minus_reference": delta_mean,
                    "delta_ci95_half_width": delta_ci,
                    "oriented_improvement": imp_mean,
                    "improvement_ci95_half_width": imp_ci,
                    "ci_excludes_zero": abs(imp_mean) > imp_ci,
                    "subjects_improved": int((improvement > 0).sum()),
                }
            )

    write_csv(args.output_root / "aggregate.csv", aggregate_rows)
    write_csv(args.output_root / "paired_comparisons.csv", paired_rows)
    payload = {
        "confidence_interval": "two-sided 95% Student-t interval across BIDMC LOSO subjects",
        "aggregate": aggregate_rows,
        "paired_comparisons": paired_rows,
    }
    (args.output_root / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    labels = list(experiments)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for axis, metric, title in zip(
        axes,
        ("waveform_correlation", "rr_mean_absolute_error_bpm_30p6s"),
        ("Waveform correlation (higher is better)", "Respiratory-rate MAE (lower is better)"),
    ):
        rows = [row for row in aggregate_rows if row["metric"] == metric]
        means = [row["mean"] for row in rows]
        cis = [row["ci95_half_width"] for row in rows]
        axis.errorbar(range(len(labels)), means, yerr=cis, fmt="o", capsize=4)
        axis.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_root / "correncoder_ablation.png", dpi=200)
    plt.close(fig)
    print(json.dumps({"experiments": labels, "paired_comparisons": len(paired_rows)}, indent=2))


if __name__ == "__main__":
    main()
