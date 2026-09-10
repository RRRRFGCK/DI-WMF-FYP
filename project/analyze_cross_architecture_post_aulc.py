"""Recompute cross-architecture AULC after excluding Epoch-0."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t
from scipy.stats import ttest_rel


ROOT = Path(__file__).resolve().parent


def read_post_aulc(path: Path) -> float:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return float(np.mean([float(row["val_accuracy"]) for row in rows if int(row["epoch"]) >= 1]))


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    half = float(student_t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / math.sqrt(len(values)))
    return mean, mean - half, mean + half


def holm(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    adjusted = [0.0] * len(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    specifications = [
        ("resnet18", ROOT / "outputs_resnet18_cross_task", "di_wmf"),
        ("correncoder", ROOT / "outputs_correncoder_cross_task", "lowrank_wmf"),
    ]
    rows = []
    for architecture, directory, method_name in specifications:
        values = {}
        for config_path in directory.glob("*/config.json"):
            config = json.loads(config_path.read_text(encoding="utf-8"))
            method = config["init"]
            if method not in {"kaiming", method_name} or int(config["seed"]) not in range(5):
                continue
            key = (config["dataset"], int(config["seed"]), method)
            values[key] = read_post_aulc(config_path.parent / "history.csv")
        architecture_rows = []
        for dataset in ("fashion", "cifar10", "sign"):
            seeds = [s for s in range(5) if (dataset, s, "kaiming") in values and (dataset, s, method_name) in values]
            reference = np.array([values[(dataset, s, "kaiming")] for s in seeds])
            method = np.array([values[(dataset, s, method_name)] for s in seeds])
            mean, low, high = mean_ci(method - reference)
            architecture_rows.append(
                {
                    "architecture": architecture,
                    "dataset": dataset,
                    "paired_seeds": len(seeds),
                    "kaiming_mean_aulc_1_to_10": float(reference.mean()),
                    "di_mean_aulc_1_to_10": float(method.mean()),
                    "mean_paired_difference": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "p_value_raw": float(ttest_rel(method, reference).pvalue),
                }
            )
        adjusted = holm([row["p_value_raw"] for row in architecture_rows])
        for row, p_value in zip(architecture_rows, adjusted):
            row["p_value_holm_across_tasks"] = p_value
            row["holm_significant_0p05"] = p_value < 0.05
        rows.extend(architecture_rows)
    output = ROOT / "submission_control_results" / "cross_architecture_post_aulc.csv"
    write_csv(output, rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
