"""Reanalyse existing results needed for the submission validity checks."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t
from scipy.stats import ttest_rel


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "submission_control_results"


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_ci(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    half = float(student_t.ppf(0.975, values.size - 1) * values.std(ddof=1) / math.sqrt(values.size))
    return mean, mean - half, mean + half


def recompute_aulc():
    rows = []
    for config_path in (ROOT / "outputs_convergence50").glob("*/config.json"):
        with config_path.open(encoding="utf-8") as handle:
            config = json.load(handle)
        if not (
            config.get("dataset") in {"fashion", "cifar10", "sign"}
            and config.get("model") == "standard"
            and config.get("init") in {"kaiming", "lowrank_wmf"}
            and int(config.get("epochs", 0)) == 50
            and float(config.get("train_fraction", 0)) == 1.0
            and int(config.get("seed", -1)) in range(10)
        ):
            continue
        history_path = config_path.parent / "history.csv"
        with history_path.open(newline="", encoding="utf-8") as handle:
            history = list(csv.DictReader(handle))
        initial = float(next(row["val_accuracy"] for row in history if int(row["epoch"]) == 0))
        trained = np.array(
            [float(row["val_accuracy"]) for row in history if int(row["epoch"]) >= 1]
        )
        inclusive = np.array([float(row["val_accuracy"]) for row in history])
        rows.append(
            {
                "dataset": config["dataset"],
                "method": config["init"],
                "seed": config["seed"],
                "epoch0_accuracy": initial,
                "aulc_0_to_50": float(inclusive.mean()),
                "aulc_1_to_50": float(trained.mean()),
                "epoch0_contribution_to_inclusive_aulc": float(initial / inclusive.size),
            }
        )
    aggregate = []
    paired = []
    for dataset in ("fashion", "cifar10", "sign"):
        by_method = {
            method: {
                int(row["seed"]): row
                for row in rows
                if row["dataset"] == dataset and row["method"] == method
            }
            for method in ("kaiming", "lowrank_wmf")
        }
        for method, method_rows in by_method.items():
            for metric in ("epoch0_accuracy", "aulc_0_to_50", "aulc_1_to_50"):
                values = np.array([float(method_rows[seed][metric]) for seed in sorted(method_rows)])
                mean, low, high = mean_ci(values)
                aggregate.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "metric": metric,
                        "runs": values.size,
                        "mean": mean,
                        "ci95_low": low,
                        "ci95_high": high,
                    }
                )
        common = sorted(set(by_method["kaiming"]) & set(by_method["lowrank_wmf"]))
        for metric in ("epoch0_accuracy", "aulc_0_to_50", "aulc_1_to_50"):
            reference = np.array([float(by_method["kaiming"][seed][metric]) for seed in common])
            method = np.array([float(by_method["lowrank_wmf"][seed][metric]) for seed in common])
            delta = method - reference
            mean, low, high = mean_ci(delta)
            paired.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "paired_seeds": len(common),
                    "kaiming_mean": float(reference.mean()),
                    "lowrank_mean": float(method.mean()),
                    "mean_paired_difference": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "p_value_raw": float(ttest_rel(method, reference).pvalue),
                }
            )
    write_csv(OUTPUT / "aulc_recomputed_rows.csv", rows)
    write_csv(OUTPUT / "aulc_recomputed_aggregate.csv", aggregate)
    write_csv(OUTPUT / "aulc_recomputed_paired.csv", paired)
    return paired


def extract_retention():
    source = ROOT / "corruption_results" / "full20_cross_task_paired.csv"
    with source.open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["method"] == "lowrank_wmf" and row["metric"] == "mean_retention"
        ]
    write_csv(OUTPUT / "full_model_retention_paired.csv", rows)
    return rows


def extract_adversarial():
    aggregate_source = ROOT / "reliability_results" / "full20_standard" / "adversarial_aggregate.csv"
    paired_source = ROOT / "reliability_results" / "full20_standard" / "adversarial_paired.csv"
    with aggregate_source.open(newline="", encoding="utf-8") as handle:
        aggregate = list(csv.DictReader(handle))
    with paired_source.open(newline="", encoding="utf-8") as handle:
        paired = list(csv.DictReader(handle))
    write_csv(OUTPUT / "adversarial_aggregate.csv", aggregate)
    write_csv(OUTPUT / "adversarial_paired.csv", paired)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paired = recompute_aulc()
    retention = extract_retention()
    extract_adversarial()
    print("AULC post-training paired effects")
    for row in paired:
        if row["metric"] == "aulc_1_to_50":
            print(row)
    print("Full-model retention effects")
    for row in retention:
        print(
            row["dataset"], row["corruption"], row["mean_paired_difference"],
            row["ci95_low"], row["ci95_high"]
        )


if __name__ == "__main__":
    main()
