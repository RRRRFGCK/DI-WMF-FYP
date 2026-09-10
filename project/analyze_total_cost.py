"""Analyse time/energy-to-accuracy including domain-initialisation overhead."""

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


ROOT = Path(__file__).resolve().parent
DEFAULT_THRESHOLDS = {
    "mnist": (80.0, 90.0, 95.0),
    "fashion": (60.0, 75.0, 82.0),
    "cifar10": (25.0, 40.0, 50.0),
    "sign": (20.0, 40.0, 60.0),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Total time/energy to fixed validation-accuracy targets"
    )
    parser.add_argument(
        "--roots", nargs="+", type=Path, default=[ROOT / "outputs_cost_measured"]
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "total_cost_results"
    )
    parser.add_argument("--model", default="standard")
    parser.add_argument(
        "--methods", nargs="+", default=["kaiming", "di_wmf", "lowrank_wmf"]
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["fashion", "cifar10", "sign"]
    )
    parser.add_argument(
        "--idle-watts",
        type=float,
        default=None,
        help=(
            "Optional common idle-power baseline. Recommended for sequential runs "
            "because per-run pre-measurements can retain heat/power from the prior run."
        ),
    )
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean_ci(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan"), float("nan")
    mean = float(values.mean())
    ci95 = (
        float(student_t.ppf(0.975, values.size - 1) * values.std(ddof=1) / math.sqrt(values.size))
        if values.size > 1
        else 0.0
    )
    return mean, ci95


def _energy_parts(metrics, common_idle_watts=None):
    energy = metrics.get("energy_measurement")
    if not energy:
        return None
    initialisation = energy.get("initialisation", {})
    training = energy.get("training_and_evaluation", {})
    if not initialisation.get("available") or not training.get("available"):
        return None
    initialisation_net = float(initialisation["net_joules_above_idle"])
    training_net = float(training["net_joules_above_idle"])
    if common_idle_watts is not None:
        initialisation_net = max(
            0.0, float(initialisation["mean_watts"]) - common_idle_watts
        ) * float(initialisation["duration_seconds"])
        training_net = max(
            0.0, float(training["mean_watts"]) - common_idle_watts
        ) * float(training["duration_seconds"])
    return {
        "initialisation_net": initialisation_net,
        "training_net": training_net,
        "initialisation_gross": float(initialisation["gross_joules"]),
        "training_gross": float(training["gross_joules"]),
    }


def collect_rows(args):
    rows = []
    for root in args.roots:
        for metrics_path in root.rglob("final_metrics.json"):
            run_dir = metrics_path.parent
            config_path = run_dir / "config.json"
            history_path = run_dir / "history.csv"
            if not config_path.exists() or not history_path.exists():
                continue
            config = json.loads(config_path.read_text(encoding="utf-8"))
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            dataset = config.get("dataset")
            method = config.get("init")
            model = config.get("model")
            if dataset not in args.datasets or method not in args.methods or model != args.model:
                continue
            history = _read_csv(history_path)
            initialisation_seconds = float(config.get("initialisation_seconds", 0.0))
            training_seconds = float(metrics["training_seconds"])
            energy_parts = _energy_parts(metrics, args.idle_watts)
            for threshold in DEFAULT_THRESHOLDS[dataset]:
                hit = next(
                    (
                        item
                        for item in history
                        if float(item["val_accuracy"]) >= threshold
                    ),
                    None,
                )
                reached = hit is not None
                elapsed = (
                    float(hit["elapsed_seconds"]) if reached else training_seconds
                )
                total_seconds = initialisation_seconds + elapsed
                total_net_energy = float("nan")
                total_gross_energy = float("nan")
                if energy_parts is not None:
                    fraction = min(max(elapsed / max(training_seconds, 1e-12), 0.0), 1.0)
                    total_net_energy = (
                        energy_parts["initialisation_net"]
                        + fraction * energy_parts["training_net"]
                    )
                    total_gross_energy = (
                        energy_parts["initialisation_gross"]
                        + fraction * energy_parts["training_gross"]
                    )
                rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "method": method,
                        "seed": int(config["seed"]),
                        "threshold_accuracy": threshold,
                        "reached": int(reached),
                        "epoch_to_threshold": int(hit["epoch"]) if reached else "",
                        "optimisation_seconds_to_threshold": elapsed,
                        "initialisation_seconds": initialisation_seconds,
                        "total_seconds_to_threshold_or_censor": total_seconds,
                        "total_gross_joules_to_threshold_or_censor": total_gross_energy,
                        "total_net_joules_to_threshold_or_censor": total_net_energy,
                        "best_val_accuracy": float(metrics["best_val_accuracy"]),
                        "run_dir": str(run_dir.resolve()),
                    }
                )
    return rows


def aggregate_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["dataset"],
                row["model"],
                row["method"],
                row["threshold_accuracy"],
            )
        ].append(row)
    aggregates = []
    for key, group in sorted(grouped.items()):
        reached = [row for row in group if row["reached"]]
        time_mean, time_ci = _mean_ci(
            [row["total_seconds_to_threshold_or_censor"] for row in reached]
        )
        gross_energy_values = [
            row["total_gross_joules_to_threshold_or_censor"]
            for row in reached
            if np.isfinite(row["total_gross_joules_to_threshold_or_censor"])
        ]
        gross_energy_mean, gross_energy_ci = _mean_ci(gross_energy_values)
        net_energy_values = [
            row["total_net_joules_to_threshold_or_censor"]
            for row in reached
            if np.isfinite(row["total_net_joules_to_threshold_or_censor"])
        ]
        net_energy_mean, net_energy_ci = _mean_ci(net_energy_values)
        censored_time_mean, censored_time_ci = _mean_ci(
            [row["total_seconds_to_threshold_or_censor"] for row in group]
        )
        aggregates.append(
            {
                "dataset": key[0],
                "model": key[1],
                "method": key[2],
                "threshold_accuracy": key[3],
                "runs": len(group),
                "reached_runs": len(reached),
                "reach_rate": len(reached) / len(group),
                "mean_total_seconds_among_reached": time_mean,
                "ci95_total_seconds_among_reached": time_ci,
                "mean_gross_joules_among_reached": gross_energy_mean,
                "ci95_gross_joules_among_reached": gross_energy_ci,
                "mean_net_joules_among_reached": net_energy_mean,
                "ci95_net_joules_among_reached": net_energy_ci,
                "mean_observed_or_censored_seconds": censored_time_mean,
                "ci95_observed_or_censored_seconds": censored_time_ci,
            }
        )
    return aggregates


def paired_rows(rows, baseline="kaiming"):
    lookup = {
        (row["dataset"], row["method"], row["seed"], row["threshold_accuracy"]): row
        for row in rows
    }
    comparisons = []
    methods = sorted({row["method"] for row in rows if row["method"] != baseline})
    for dataset in sorted({row["dataset"] for row in rows}):
        for threshold in DEFAULT_THRESHOLDS[dataset]:
            for method in methods:
                available_pairs = []
                for seed in sorted({row["seed"] for row in rows}):
                    base = lookup.get((dataset, baseline, seed, threshold))
                    candidate = lookup.get((dataset, method, seed, threshold))
                    if base and candidate:
                        available_pairs.append((base, candidate))
                pairs = [
                    (base, candidate)
                    for base, candidate in available_pairs
                    if base["reached"] and candidate["reached"]
                ]
                time_deltas = [
                    candidate["total_seconds_to_threshold_or_censor"]
                    - base["total_seconds_to_threshold_or_censor"]
                    for base, candidate in pairs
                ]
                energy_deltas = [
                    candidate["total_net_joules_to_threshold_or_censor"]
                    - base["total_net_joules_to_threshold_or_censor"]
                    for base, candidate in pairs
                    if np.isfinite(base["total_net_joules_to_threshold_or_censor"])
                    and np.isfinite(candidate["total_net_joules_to_threshold_or_censor"])
                ]
                gross_energy_deltas = [
                    candidate["total_gross_joules_to_threshold_or_censor"]
                    - base["total_gross_joules_to_threshold_or_censor"]
                    for base, candidate in pairs
                    if np.isfinite(base["total_gross_joules_to_threshold_or_censor"])
                    and np.isfinite(candidate["total_gross_joules_to_threshold_or_censor"])
                ]
                time_mean, time_ci = _mean_ci(time_deltas)
                energy_mean, energy_ci = _mean_ci(energy_deltas)
                gross_energy_mean, gross_energy_ci = _mean_ci(gross_energy_deltas)
                comparisons.append(
                    {
                        "dataset": dataset,
                        "threshold_accuracy": threshold,
                        "baseline": baseline,
                        "method": method,
                        "paired_runs": len(available_pairs),
                        "baseline_reached_runs": sum(base["reached"] for base, _ in available_pairs),
                        "method_reached_runs": sum(candidate["reached"] for _, candidate in available_pairs),
                        "reach_rate_delta": (
                            float(
                                np.mean([candidate["reached"] for _, candidate in available_pairs])
                                - np.mean([base["reached"] for base, _ in available_pairs])
                            )
                            if available_pairs else float("nan")
                        ),
                        "method_only_reached_runs": sum(
                            candidate["reached"] and not base["reached"]
                            for base, candidate in available_pairs
                        ),
                        "baseline_only_reached_runs": sum(
                            base["reached"] and not candidate["reached"]
                            for base, candidate in available_pairs
                        ),
                        "paired_reached_runs": len(pairs),
                        "mean_time_delta_seconds": time_mean,
                        "ci95_time_delta_seconds": time_ci,
                        "time_win_rate": (
                            float(np.mean(np.asarray(time_deltas) < 0)) if time_deltas else float("nan")
                        ),
                        "mean_gross_energy_delta_joules": gross_energy_mean,
                        "ci95_gross_energy_delta_joules": gross_energy_ci,
                        "gross_energy_win_rate": (
                            float(np.mean(np.asarray(gross_energy_deltas) < 0))
                            if gross_energy_deltas else float("nan")
                        ),
                        "mean_energy_delta_joules": energy_mean,
                        "ci95_energy_delta_joules": energy_ci,
                        "energy_win_rate": (
                            float(np.mean(np.asarray(energy_deltas) < 0)) if energy_deltas else float("nan")
                        ),
                    }
                )
    return comparisons


def plot_aggregates(aggregates, output_dir):
    datasets = sorted({row["dataset"] for row in aggregates})
    methods = sorted({row["method"] for row in aggregates})
    figure, axes = plt.subplots(2, len(datasets), figsize=(5 * len(datasets), 8), squeeze=False)
    for column, dataset in enumerate(datasets):
        for method in methods:
            selected = sorted(
                [row for row in aggregates if row["dataset"] == dataset and row["method"] == method],
                key=lambda row: row["threshold_accuracy"],
            )
            if not selected:
                continue
            x = [row["threshold_accuracy"] for row in selected]
            axes[0, column].errorbar(
                x,
                [row["mean_total_seconds_among_reached"] for row in selected],
                yerr=[row["ci95_total_seconds_among_reached"] for row in selected],
                marker="o",
                capsize=3,
                label=method,
            )
            axes[1, column].errorbar(
                x,
                [row["mean_gross_joules_among_reached"] for row in selected],
                yerr=[row["ci95_gross_joules_among_reached"] for row in selected],
                marker="o",
                capsize=3,
                label=method,
            )
        axes[0, column].set_title(dataset)
        axes[0, column].set_ylabel("Total seconds to target")
        axes[1, column].set_ylabel("Gross GPU joules to target")
        axes[1, column].set_xlabel("Validation accuracy target (%)")
        for row in range(2):
            axes[row, column].grid(alpha=0.25)
            axes[row, column].legend()
    figure.tight_layout()
    figure.savefig(output_dir / "time_energy_to_accuracy.png", dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = collect_rows(args)
    if not rows:
        raise RuntimeError("No compatible completed runs were found")
    aggregates = aggregate_rows(rows)
    comparisons = paired_rows(rows)
    _write_csv(args.output_dir / "cost_rows.csv", rows)
    _write_csv(args.output_dir / "cost_aggregate.csv", aggregates)
    _write_csv(args.output_dir / "cost_paired.csv", comparisons)
    plot_aggregates(aggregates, args.output_dir)
    summary = {
        "runs": len({row["run_dir"] for row in rows}),
        "datasets": sorted({row["dataset"] for row in rows}),
        "methods": sorted({row["method"] for row in rows}),
        "energy_available": any(
            np.isfinite(row["total_gross_joules_to_threshold_or_censor"]) for row in rows
        ),
        "cost_definition": (
            "initialisation cost + optimisation/evaluation cost up to first fixed "
            "validation-accuracy crossing; failures are right-censored at run end"
        ),
        "common_idle_watts_for_net_energy": args.idle_watts,
        "confidence_interval": "two-sided 95% Student-t interval across seeds",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
