"""Build the controlled cross-task first-layer comparison tables and figures.

The input experiment changes only conv1.  Every later convolution and the
classifier use the same Kaiming initialisation, so differences can be assigned
to the first-layer prior rather than to a coherent layer-wise initialisation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t


METHODS = [
    "kaiming",
    "random_stem",
    "gabor",
    "pca",
    "kmeans",
    "di_wmf",
    "lowrank_wmf",
]
METHOD_LABELS = {
    "kaiming": "Kaiming",
    "random_stem": "Random stem",
    "gabor": "Gabor",
    "pca": "PCA",
    "kmeans": "K-means",
    "di_wmf": "Diagonal DI-WMF",
    "lowrank_wmf": "Low-rank DI-WMF",
}
DATASETS = ["fashion", "cifar10", "sign"]
DATASET_LABELS = {"fashion": "Fashion-MNIST", "cifar10": "CIFAR-10", "sign": "Sign"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_ci(values: list[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    if len(array) < 2:
        return mean, mean, mean
    half = float(student_t.ppf(0.975, len(array) - 1)) * float(array.std(ddof=1)) / math.sqrt(len(array))
    return mean, mean - half, mean + half


def paired_rows(rows: list[dict[str, str]], value_key: str) -> list[dict]:
    by_key = {(r["dataset"], r["method"], int(r["model_seed"])): float(r[value_key]) for r in rows}
    output: list[dict] = []
    for dataset in DATASETS:
        for method in METHODS[1:]:
            reference_seeds = {key[2] for key in by_key if key[:2] == (dataset, "kaiming")}
            candidate_seeds = {key[2] for key in by_key if key[:2] == (dataset, method)}
            seeds = sorted(reference_seeds & candidate_seeds)
            reference = [by_key[(dataset, "kaiming", seed)] for seed in seeds]
            candidate = [by_key[(dataset, method, seed)] for seed in seeds]
            differences = [candidate[i] - reference[i] for i in range(len(seeds))]
            mean, low, high = mean_ci(differences)
            output.append(
                {
                    "dataset": dataset,
                    "reference": "kaiming",
                    "method": method,
                    "metric": value_key,
                    "paired_seeds": len(seeds),
                    "reference_mean": np.mean(reference),
                    "method_mean": np.mean(candidate),
                    "mean_paired_difference": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "significant": bool(low > 0 or high < 0),
                }
            )
    return output


def build_main_table(aggregate: list[dict[str, str]]) -> list[dict]:
    lookup = {(r["dataset"], r["method"]): r for r in aggregate}
    table: list[dict] = []
    for dataset in DATASETS:
        for method in METHODS:
            row = lookup[(dataset, method)]
            table.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "runs": int(row["runs"]),
                    "epoch0_mean": float(row["mean_initial_val_accuracy"]),
                    "epoch0_std": float(row["std_initial_val_accuracy"]),
                    "aulc_mean": float(row["mean_validation_aulc"]),
                    "aulc_std": float(row["std_validation_aulc"]),
                    "test_mean": float(row["mean_test_accuracy"]),
                    "test_std": float(row["std_test_accuracy"]),
                    "conv1_drift_mean": float(row["mean_conv1_drift"]),
                    "conv1_drift_std": float(row["std_conv1_drift"]),
                    "device": row["device_resolved"],
                }
            )
    return table


def plot_metrics(table: list[dict], output: Path) -> None:
    lookup = {(r["dataset"], r["method"]): r for r in table}
    colours = {
        "kaiming": "#555555",
        "random_stem": "#aaaaaa",
        "gabor": "#238b45",
        "pca": "#756bb1",
        "kmeans": "#c98b00",
        "di_wmf": "#e6550d",
        "lowrank_wmf": "#b2182b",
    }
    fig, axes = plt.subplots(3, 2, figsize=(13.8, 9.2), constrained_layout=True)
    for row_index, dataset in enumerate(DATASETS):
        for column, (key, title) in enumerate(
            [
                ("aulc", "Post-update validation AULC"),
                ("test", "Final test accuracy (%)"),
            ]
        ):
            axis = axes[row_index, column]
            means = [lookup[(dataset, method)][f"{key}_mean"] for method in METHODS]
            stds = [lookup[(dataset, method)][f"{key}_std"] for method in METHODS]
            runs = [int(lookup[(dataset, method)]["runs"]) for method in METHODS]
            cis = [
                float(student_t.ppf(0.975, run_count - 1)) * value / math.sqrt(run_count)
                for value, run_count in zip(stds, runs)
            ]
            positions = np.arange(len(METHODS))
            best_index = int(np.argmax(means))
            axis.axhspan(best_index - 0.43, best_index + 0.43, color="#f8ece9", zorder=-3)
            axis.axvline(means[0], color="#777777", linestyle="--", linewidth=0.9, alpha=0.75)
            for position, method, mean, ci in zip(positions, METHODS, means, cis):
                axis.errorbar(
                    mean,
                    position,
                    xerr=ci,
                    fmt="o",
                    ms=7.4 if method == METHODS[best_index] else 6.2,
                    mfc=colours[method],
                    mec="white",
                    mew=0.8,
                    ecolor=colours[method],
                    elinewidth=1.5,
                    capsize=3,
                    zorder=3,
                )
            axis.set_title(f"{DATASET_LABELS[dataset]}  ·  {title}", loc="left", fontweight="bold")
            axis.set_yticks(
                positions,
                [
                    METHOD_LABELS[method] + ("  [best mean]" if index == best_index else "")
                    for index, method in enumerate(METHODS)
                ],
            )
            axis.invert_yaxis()
            axis.grid(axis="x", alpha=0.18)
            axis.spines[["top", "right", "left"]].set_visible(False)
            axis.tick_params(axis="y", length=0)
    fig.suptitle(
        "First-layer priors are task-dependent",
        fontsize=15,
        fontweight="bold",
        y=1.015,
    )
    fig.savefig(output / "first_layer_cross_task_main.png", dpi=240, bbox_inches="tight")
    fig.savefig(output / "first_layer_cross_task_main.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_selectivity(paired: list[dict], output: Path) -> None:
    lookup = {(r["dataset"], r["method"]): r for r in paired}
    methods = METHODS[1:]
    values = np.array(
        [[float(lookup[(dataset, method)]["mean_paired_difference"]) for method in methods] for dataset in DATASETS]
    )
    limit = max(abs(float(values.min())), abs(float(values.max())))
    fig, axis = plt.subplots(figsize=(10.5, 3.7), constrained_layout=True)
    image = axis.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    for i, dataset in enumerate(DATASETS):
        for j, method in enumerate(methods):
            row = lookup[(dataset, method)]
            star = "*" if row["significant"] else ""
            axis.text(j, i, f"{values[i, j]:+.3f}{star}", ha="center", va="center", fontsize=9)
    axis.set_xticks(np.arange(len(methods)), [METHOD_LABELS[m] for m in methods], rotation=25, ha="right")
    axis.set_yticks(np.arange(len(DATASETS)), [DATASET_LABELS[d] for d in DATASETS])
    axis.set_title("Best-class channel selectivity relative to Kaiming (*: paired 95% CI excludes zero)")
    fig.colorbar(image, ax=axis, label="Paired difference")
    fig.savefig(output / "first_layer_selectivity_effects.png", dpi=240, bbox_inches="tight")
    fig.savefig(output / "first_layer_selectivity_effects.pdf", bbox_inches="tight")
    plt.close(fig)


def build_rankings(table: list[dict]) -> list[dict]:
    output: list[dict] = []
    for dataset in DATASETS:
        subset = [r for r in table if r["dataset"] == dataset]
        for metric in ["epoch0_mean", "aulc_mean", "test_mean"]:
            for rank, row in enumerate(sorted(subset, key=lambda item: item[metric], reverse=True), start=1):
                output.append(
                    {
                        "dataset": dataset,
                        "metric": metric,
                        "rank": rank,
                        "method": row["method"],
                        "mean": row[metric],
                    }
                )
    return output


def summarise_corruptions(rows: list[dict[str, str]]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["metric"] in {"normalised_accuracy_auc", "mean_retention"}:
            grouped[(row["dataset"], row["method"], row["metric"])].append(row)
    output: list[dict] = []
    for (dataset, method, metric), group in sorted(grouped.items()):
        differences = [float(row["mean_paired_difference"]) for row in group]
        output.append(
            {
                "dataset": dataset,
                "method": method,
                "metric": metric,
                "corruptions": len(group),
                "mean_paired_difference_across_corruptions": float(np.mean(differences)),
                "significant_positive_corruptions": sum(float(row["ci95_low"]) > 0 for row in group),
                "significant_negative_corruptions": sum(float(row["ci95_high"]) < 0 for row in group),
            }
        )
    return output


def plot_corruptions(summary: list[dict], output: Path) -> None:
    lookup = {(r["dataset"], r["method"], r["metric"]): r for r in summary}
    methods = METHODS[1:]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), constrained_layout=True)
    for axis, metric, title in [
        (axes[0], "normalised_accuracy_auc", "Absolute corruption accuracy AUC"),
        (axes[1], "mean_retention", "Relative accuracy retention"),
    ]:
        values = np.array(
            [
                [lookup[(dataset, method, metric)]["mean_paired_difference_across_corruptions"] for method in methods]
                for dataset in DATASETS
            ],
            dtype=float,
        )
        limit = max(abs(float(values.min())), abs(float(values.max())))
        image = axis.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
        for i, dataset in enumerate(DATASETS):
            for j, method in enumerate(methods):
                row = lookup[(dataset, method, metric)]
                axis.text(
                    j,
                    i,
                    f"{values[i, j]:+.2f}\n{row['significant_positive_corruptions']}+/"
                    f"{row['significant_negative_corruptions']}−",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
        axis.set_xticks(np.arange(len(methods)), [METHOD_LABELS[m] for m in methods], rotation=28, ha="right")
        axis.set_yticks(np.arange(len(DATASETS)), [DATASET_LABELS[d] for d in DATASETS])
        axis.set_title(f"{title}\nmean paired difference; significant corruption counts")
        fig.colorbar(image, ax=axis, shrink=0.82)
    fig.savefig(output / "first_layer_corruption_effects.png", dpi=240, bbox_inches="tight")
    fig.savefig(output / "first_layer_corruption_effects.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, default=Path("outputs_first_layer_cross_task/aggregate.csv"))
    parser.add_argument("--paired-training", type=Path, default=Path("first_layer_cross_task_results/paired_training.csv"))
    parser.add_argument(
        "--selectivity-rows",
        type=Path,
        default=Path("first_layer_cross_task_results/selectivity_combined_rows.csv"),
    )
    parser.add_argument(
        "--paired-corruptions",
        type=Path,
        default=Path("first_layer_cross_task_results/paired_corruptions.csv"),
    )
    parser.add_argument("--output", type=Path, default=Path("first_layer_cross_task_results"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    aggregate = read_csv(args.aggregate)
    table = build_main_table(aggregate)
    write_csv(args.output / "main_metrics.csv", table)
    write_csv(args.output / "method_rankings.csv", build_rankings(table))
    plot_metrics(table, args.output)

    selectivity = read_csv(args.selectivity_rows)
    selectivity_paired = paired_rows(selectivity, "mean_best_class_selectivity")
    write_csv(args.output / "paired_selectivity.csv", selectivity_paired)
    plot_selectivity(selectivity_paired, args.output)

    paired_training = read_csv(args.paired_training)
    significant = [
        row
        for row in paired_training
        if float(row["ci95_low"]) > 0 or float(row["ci95_high"]) < 0
    ]
    write_csv(args.output / "significant_training_effects.csv", significant)

    corruption_count = 0
    if args.paired_corruptions.exists():
        corruption_rows = read_csv(args.paired_corruptions)
        corruption_summary = summarise_corruptions(corruption_rows)
        write_csv(args.output / "corruption_effect_summary.csv", corruption_summary)
        plot_corruptions(corruption_summary, args.output)
        corruption_count = len({row["corruption"] for row in corruption_rows})

    payload = {
        "design": {
            "datasets": DATASETS,
            "methods": METHODS,
            "seeds_by_dataset": {
                "fashion": [0, 1, 2, 3, 4],
                "cifar10": [0, 1, 2, 3, 4],
                "sign": list(range(10)),
            },
            "epochs": 10,
            "train_fraction": 0.1,
            "changed_component": "first convolution only",
            "common_later_initialisation": "Kaiming",
        },
        "validation": {
            "expected_runs": 140,
            "observed_runs": sum(int(row["runs"]) for row in table),
            "all_cuda": all(row["device"] == "cuda" for row in table),
            "evaluated_corruptions": corruption_count,
            "corruption_seeds_per_checkpoint": 3,
            "corruption_model_seeds_by_dataset": {
                "fashion": 5,
                "cifar10": 5,
                "sign": 10,
            },
        },
    }
    (args.output / "experiment_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
