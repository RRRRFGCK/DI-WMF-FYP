"""Aggregate extended Correncoder runs, hydrating lag metrics from waveforms."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t

from run_correncoder_regression import lag_robust_waveform_correlation


METRICS = {
    "test_mse": "lower",
    "test_mae": "lower",
    "waveform_correlation": "higher",
    "max_lag_waveform_correlation": "higher",
    "rr_mean_absolute_error_bpm_30p6s": "lower",
}

DIAGNOSTIC_FIELDS = (
    "final_calibration_gain",
    "final_calibration_offset",
    "epoch0_train_mse_before_calibration",
    "epoch0_train_mse_after_calibration",
    "epoch0_train_abs_correlation_after_calibration",
    "conv2_explained_covariance_fraction",
    "conv3_explained_covariance_fraction",
)


def mean_ci(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    half = float(
        student_t.ppf(0.975, len(values) - 1)
        * values.std(ddof=1)
        / math.sqrt(len(values))
    )
    return mean, half


def load_rows(root: Path):
    with (root / "fold_metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if row.get("max_lag_waveform_correlation", "").strip():
            continue
        fold = int(row["fold"])
        seed = int(row["seed"])
        candidates = list(root.glob(f"*fold{fold:02d}_seed{seed}/waveforms.npz"))
        if len(candidates) != 1:
            raise RuntimeError(f"Cannot resolve waveform for fold {fold} in {root}")
        arrays = np.load(candidates[0])
        correlation, lag = lag_robust_waveform_correlation(
            arrays["prediction"], arrays["target"]
        )
        row["max_lag_waveform_correlation"] = correlation
        row["best_waveform_lag_samples"] = lag
    return {int(row["test_subject"]): row for row in rows}


def load_initialisation_diagnostics(root: Path, experiment: str) -> list[dict]:
    """Read fold-level scale and covariance diagnostics when they are present."""

    rows = []
    for path in sorted(root.glob("*fold*_seed*/final_metrics.json")):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        report = metrics.get("initialisation_report", {})
        calibration = report.get("safe_calibration_head") or {}
        layers = report.get("layers") or {}
        conv2 = layers.get("conv2") or {}
        conv3 = layers.get("conv3") or {}
        values = {
            "final_calibration_gain": metrics.get("final_calibration_gain"),
            "final_calibration_offset": metrics.get("final_calibration_offset"),
            "epoch0_train_mse_before_calibration": calibration.get("train_mse_before"),
            "epoch0_train_mse_after_calibration": calibration.get("train_mse_after"),
            "epoch0_train_abs_correlation_after_calibration": (
                abs(float(calibration["train_correlation_after"]))
                if calibration.get("train_correlation_after") is not None else None
            ),
            "conv2_explained_covariance_fraction": conv2.get(
                "explained_covariance_fraction"
            ),
            "conv3_explained_covariance_fraction": conv3.get(
                "explained_covariance_fraction"
            ),
        }
        if any(value is not None for value in values.values()):
            rows.append(
                {
                    "experiment": experiment,
                    "fold": metrics.get("fold"),
                    "test_subject": metrics.get("test_subject"),
                    **values,
                }
            )
    return rows


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_paired_improvements(path: Path, paired: list[dict]) -> None:
    """Forest plot: positive values always favour the named experiment."""

    metric_titles = {
        "test_mse": "MSE reduction",
        "test_mae": "MAE reduction",
        "waveform_correlation": "Zero-lag correlation gain",
        "max_lag_waveform_correlation": "Max-lag correlation gain",
        "rr_mean_absolute_error_bpm_30p6s": "RR MAE reduction (bpm)",
    }
    comparisons = list(dict.fromkeys(
        f"{row['experiment']} vs {row['reference']}" for row in paired
    ))
    figure, axes = plt.subplots(2, 3, figsize=(15, 7.5), constrained_layout=True)
    for axis, metric in zip(axes.flat, METRICS):
        rows = [row for row in paired if row["metric"] == metric]
        lookup = {
            f"{row['experiment']} vs {row['reference']}": row for row in rows
        }
        means = [float(lookup[label]["oriented_improvement"]) for label in comparisons]
        errors = [
            float(lookup[label]["improvement_ci95_half_width"])
            for label in comparisons
        ]
        y = np.arange(len(comparisons))
        colours = [
            "#167D4A" if bool(lookup[label]["ci_excludes_zero"]) else "#667085"
            for label in comparisons
        ]
        for position, mean, error, colour in zip(y, means, errors, colours):
            axis.errorbar(mean, position, xerr=error, fmt="none", ecolor=colour,
                          elinewidth=2, capsize=4)
        axis.scatter(means, y, c=colours, s=38, zorder=3)
        axis.axvline(0.0, color="black", linewidth=0.8, linestyle="--")
        axis.set_title(metric_titles[metric])
        axis.set_yticks(y)
        axis.set_yticklabels(comparisons if axis in (axes[0, 0], axes[1, 0]) else [])
        axis.grid(axis="x", alpha=0.2)
    axes.flat[-1].axis("off")
    figure.suptitle("Paired 53-subject LOSO improvements (mean and 95% CI)")
    figure.savefig(path.with_suffix(".png"), dpi=220)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", action="append", required=True)
    parser.add_argument("--comparison", action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    experiments = {}
    experiment_roots = {}
    for item in args.experiment:
        label, path = item.split("=", 1)
        root = Path(path)
        experiments[label] = load_rows(root)
        experiment_roots[label] = root
    aggregate = []
    for label, rows in experiments.items():
        for metric in METRICS:
            values = [float(row[metric]) for row in rows.values()]
            mean, ci = mean_ci(values)
            aggregate.append(
                {
                    "experiment": label,
                    "metric": metric,
                    "n": len(values),
                    "mean": mean,
                    "ci95_half_width": ci,
                    "median": float(np.median(values)),
                }
            )
    paired = []
    for item in args.comparison:
        method, reference = item.split("=", 1)
        common = sorted(set(experiments[method]) & set(experiments[reference]))
        for metric, direction in METRICS.items():
            values = np.array([float(experiments[method][s][metric]) for s in common])
            baseline = np.array([float(experiments[reference][s][metric]) for s in common])
            delta = values - baseline
            improvement = delta if direction == "higher" else -delta
            delta_mean, delta_ci = mean_ci(delta)
            improvement_mean, improvement_ci = mean_ci(improvement)
            paired.append(
                {
                    "experiment": method,
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
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_root / "aggregate.csv", aggregate)
    write_csv(args.output_root / "paired_comparisons.csv", paired)
    diagnostics = []
    for label, root in experiment_roots.items():
        diagnostics.extend(load_initialisation_diagnostics(root, label))
    if diagnostics:
        write_csv(args.output_root / "initialisation_diagnostics.csv", diagnostics)
        diagnostic_aggregate = []
        for label in experiments:
            subset = [row for row in diagnostics if row["experiment"] == label]
            for field in DIAGNOSTIC_FIELDS:
                values = np.asarray(
                    [float(row[field]) for row in subset if row[field] is not None],
                    dtype=np.float64,
                )
                if not len(values):
                    continue
                mean, ci = mean_ci(values) if len(values) > 1 else (float(values[0]), 0.0)
                diagnostic_aggregate.append(
                    {
                        "experiment": label,
                        "diagnostic": field,
                        "n": len(values),
                        "mean": mean,
                        "ci95_half_width": ci,
                        "minimum": float(values.min()),
                        "maximum": float(values.max()),
                    }
                )
        write_csv(
            args.output_root / "initialisation_diagnostics_aggregate.csv",
            diagnostic_aggregate,
        )
    plot_paired_improvements(args.output_root / "paired_improvements", paired)
    (args.output_root / "summary.json").write_text(
        json.dumps({"aggregate": aggregate, "paired": paired}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"experiments": list(experiments), "paired_rows": len(paired)}, indent=2))


if __name__ == "__main__":
    main()
