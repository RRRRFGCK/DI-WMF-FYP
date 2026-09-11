"""Compare the trained fitted-head control with the existing 50-epoch runs."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t
from scipy.stats import ttest_rel


ROOT = Path(__file__).resolve().parent
EXPECTED_SEEDS = set(range(10))


def mean_ci(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    half = float(student_t.ppf(0.975, values.size - 1) * values.std(ddof=1) / math.sqrt(values.size))
    return mean, mean - half, mean + half


def post_aulc(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return float(np.mean([float(row["val_accuracy"]) for row in rows if int(row["epoch"]) >= 1]))


def holm_adjust(p_values):
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (total - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def write_csv(path, rows):
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    output = ROOT / "fitted_head_training_control_results"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for config_path in (ROOT / "outputs_convergence50").glob("*/config.json"):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not (
            config.get("dataset") in {"fashion", "cifar10", "sign"}
            and config.get("model") == "standard"
            and config.get("init") in {"kaiming", "lowrank_wmf"}
            and int(config.get("epochs", 0)) == 50
            and int(config.get("seed", -1)) in EXPECTED_SEEDS
        ):
            continue
        metrics = json.loads((config_path.parent / "final_metrics.json").read_text(encoding="utf-8"))
        rows.append(
            {
                "dataset": config["dataset"],
                "seed": config["seed"],
                "condition": (
                    "di_conv__fitted_head"
                    if config["init"] == "lowrank_wmf"
                    else "kaiming_conv__kaiming_head"
                ),
                "initial_validation_accuracy": metrics["initial_val_accuracy"],
                "post_training_aulc": post_aulc(config_path.parent / "history.csv"),
                "final_test_accuracy": metrics["test_accuracy"],
            }
        )
    # Read per-run records instead of relying on a root summary that can be
    # overwritten when independent seeds are trained concurrently.
    for summary_path in (
        ROOT / "outputs_convergence50_fitted_head_control"
    ).glob("*/control_summary.json"):
        row = json.loads(summary_path.read_text(encoding="utf-8"))
        if int(row.get("seed", -1)) in EXPECTED_SEEDS:
            rows.append(row)

    aggregate = []
    paired = []
    metrics = (
        "initial_validation_accuracy",
        "post_training_aulc",
        "final_test_accuracy",
    )
    contrasts = (
        ("head_fit_effect_on_kaiming_conv", "kaiming_conv__fitted_head", "kaiming_conv__kaiming_head"),
        ("di_conv_effect_with_fitted_head", "di_conv__fitted_head", "kaiming_conv__fitted_head"),
    )
    for dataset in ("fashion", "cifar10", "sign"):
        by_condition = {}
        for condition in (
            "kaiming_conv__kaiming_head",
            "kaiming_conv__fitted_head",
            "di_conv__fitted_head",
        ):
            by_condition[condition] = {
                int(row["seed"]): row
                for row in rows
                if row["dataset"] == dataset and row["condition"] == condition
            }
            for metric in metrics:
                values = np.array(
                    [float(by_condition[condition][seed][metric]) for seed in sorted(by_condition[condition])]
                )
                mean, low, high = mean_ci(values)
                aggregate.append(
                    {
                        "dataset": dataset,
                        "condition": condition,
                        "metric": metric,
                        "runs": values.size,
                        "mean": mean,
                        "ci95_low": low,
                        "ci95_high": high,
                    }
                )
        for contrast, left, right in contrasts:
            common = sorted(set(by_condition[left]) & set(by_condition[right]))
            for metric in metrics:
                left_values = np.array([float(by_condition[left][seed][metric]) for seed in common])
                right_values = np.array([float(by_condition[right][seed][metric]) for seed in common])
                delta = left_values - right_values
                mean, low, high = mean_ci(delta)
                paired.append(
                    {
                        "dataset": dataset,
                        "contrast": contrast,
                        "metric": metric,
                        "paired_seeds": len(common),
                        "reference_mean": float(right_values.mean()),
                        "method_mean": float(left_values.mean()),
                        "mean_paired_difference": mean,
                        "ci95_low": low,
                        "ci95_high": high,
                        "p_value_raw": float(ttest_rel(left_values, right_values).pvalue),
                    }
                )
    for contrast in {row["contrast"] for row in paired}:
        for metric in metrics:
            selected = [row for row in paired if row["contrast"] == contrast and row["metric"] == metric]
            adjusted = holm_adjust([float(row["p_value_raw"]) for row in selected])
            for row, value in zip(selected, adjusted):
                row["p_value_holm_across_tasks"] = value
                row["holm_significant_0p05"] = value < 0.05
    write_csv(output / "training_control_rows.csv", rows)
    write_csv(output / "training_control_aggregate.csv", aggregate)
    write_csv(output / "training_control_paired.csv", paired)
    for row in paired:
        if row["contrast"] == "di_conv_effect_with_fitted_head":
            print(row)


if __name__ == "__main__":
    main()
