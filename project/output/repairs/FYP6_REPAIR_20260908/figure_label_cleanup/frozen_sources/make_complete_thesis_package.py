"""Build the complete dissertation figure/table package from finished experiments.

The script deliberately excludes smoke tests.  Every plotted estimate comes from
an aggregate or paired table produced by the experiment scripts, and statistical
labels use the pre-computed Holm-adjusted analysis where applicable.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import make_thesis_artifacts


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "thesis_artifacts"
PACKAGE = BASE / "complete_package"
MAIN_FIG = PACKAGE / "figures" / "main"
SUPP_FIG = PACKAGE / "figures" / "supplementary"
MAIN_TAB = PACKAGE / "tables" / "main"
SUPP_TAB = PACKAGE / "tables" / "supplementary"

METHOD_LABELS = {
    "kaiming": "Kaiming",
    "random_stem": "Default random",
    "gabor": "Gabor",
    "pca": "PCA",
    "kmeans": "Class K-means",
    "di_wmf": "Diagonal DI-WMF",
    "lowrank_wmf": "Low-rank DI-WMF",
    "lowrank_wmf_rank16": "Low-rank DI-WMF",
}
COLORS = {
    "kaiming": "#4d4d4d",
    "random_stem": "#969696",
    "gabor": "#1b9e77",
    "pca": "#7570b3",
    "kmeans": "#e6ab02",
    "di_wmf": "#d95f02",
    "lowrank_wmf": "#b2182b",
    "lowrank_wmf_rank16": "#b2182b",
}
DATASET_LABELS = {"fashion": "Fashion-MNIST", "cifar10": "CIFAR-10", "sign": "Sign"}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write(path: Path, data: list[dict]) -> None:
    if not data:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def save(fig: plt.Figure, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.5,
            "figure.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "savefig.transparent": False,
        }
    )


def copy_existing_main_figures() -> None:
    mapping = {
        "figure_cross_task_effects.png": "F01_cross_task_paired_effects.png",
        "figure_learning_curves_50epoch.png": "F02_learning_curves_50_epochs.png",
        "figure_corruption_heatmap.png": "F04_corruption_effect_heatmap.png",
        "figure_parameter_efficiency.png": "F05_parameter_efficiency.png",
        "figure_structured_efficiency.png": "F06_structured_pruning.png",
        "figure_reliability.png": "F07_reliability.png",
        "figure_template_semantics.png": "F08_template_semantics.png",
        "figure_first_layer_cross_task.png": "F16_first_layer_prior_ablation.png",
        "figure_correncoder_depth_softlag_paired.png": "F15_correncoder_depth_lag_forest.png",
    }
    for source, target in mapping.items():
        shutil.copy2(BASE / source, MAIN_FIG / target)
    # Preserve vector originals where already available.
    vector_mapping = {
        ROOT / "first_layer_cross_task_results" / "first_layer_cross_task_main.pdf": MAIN_FIG / "F16_first_layer_prior_ablation.pdf",
        ROOT / "correncoder_completed_extension_results" / "paired_improvements.pdf": MAIN_FIG / "F15_correncoder_depth_lag_forest.pdf",
    }
    for source, target in vector_mapping.items():
        if source.exists():
            shutil.copy2(source, target)


def cross_architecture() -> None:
    data = rows(ROOT / "statistical_corrections" / "all_core_hypotheses_holm.csv")
    selected = []
    for row in data:
        if row["family"] != "cross_architecture" or row["metric"] not in {
            "initial_val_accuracy", "validation_aulc", "test_accuracy"
        }:
            continue
        hypothesis = json.loads(row["hypothesis"])
        if hypothesis.get("method") != "lowrank_wmf_rank16":
            continue
        selected.append({**row, "method": hypothesis["method"]})
    compact = [
        {
            "dataset": row["dataset"],
            "architecture": row["model"],
            "metric": row["metric"],
            "paired_seeds": row["n_paired"],
            "lowrank_minus_kaiming": row["paired_difference"],
            "ci95_low": row["ci95_low"],
            "ci95_high": row["ci95_high"],
            "holm_adjusted_p": row["p_value_holm"],
            "holm_significant_0p05": row["holm_significant_0p05"],
            "paired_cohen_dz": row["cohen_dz"],
        }
        for row in selected
    ]
    write(MAIN_TAB / "T03_cross_architecture_holm.csv", compact)

    metrics = [
        ("initial_val_accuracy", "Epoch-0 accuracy"),
        ("validation_aulc", "Post-update AULC"),
        ("test_accuracy", "Final accuracy"),
    ]
    architectures = ["resnet18", "correncoder"]
    datasets = ["fashion", "cifar10", "sign"]
    offsets = {"fashion": -0.18, "cifar10": 0.0, "sign": 0.18}
    markers = {"fashion": "o", "cifar10": "s", "sign": "^"}
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.7), sharey=True)
    for axis, (metric, title) in zip(axes, metrics):
        subset = [r for r in selected if r["metric"] == metric]
        for dataset in datasets:
            ds = {r["model"]: r for r in subset if r["dataset"] == dataset}
            means = np.array([float(ds[a]["paired_difference"]) for a in architectures])
            lows = np.array([float(ds[a]["ci95_low"]) for a in architectures])
            highs = np.array([float(ds[a]["ci95_high"]) for a in architectures])
            y = np.arange(len(architectures)) + offsets[dataset]
            axis.errorbar(
                means, y, xerr=np.vstack([means - lows, highs - means]), fmt=markers[dataset],
                ms=5, capsize=3, color=COLORS[{"fashion": "di_wmf", "cifar10": "pca", "sign": "lowrank_wmf"}[dataset]],
                label=DATASET_LABELS[dataset], linewidth=1.1,
            )
            for j, arch in enumerate(architectures):
                if ds[arch]["holm_significant_0p05"] == "True":
                    axis.text(highs[j] + 0.025 * max(1, np.ptp(np.r_[lows, highs])), y[j], "*", va="center")
        axis.axvline(0, color="#333333", lw=0.8)
        axis.set_yticks(np.arange(2), ["ResNet18", "Classification\nCorrencoder"])
        axis.set_title(title)
        axis.set_xlabel("Low-rank DI-WMF − Kaiming (percentage points)")
        axis.grid(axis="x", alpha=0.2)
    axes[-1].legend(frameon=False, loc="lower right")
    fig.suptitle("Cross-architecture transfer of domain-informed initialisation")
    fig.tight_layout()
    save(fig, MAIN_FIG, "F03_cross_architecture_transfer")


def explanation_faithfulness() -> None:
    combined = []
    for path in (
        ROOT / "template_semantics_results" / "expanded_500" / "explanation_paired.csv",
        ROOT / "template_semantics_results" / "sign_full_10seed" / "explanation_paired.csv",
    ):
        for row in rows(path):
            if (
                row["reference_explainer"] == "random"
                and row["candidate_explainer"] == "exact_template_cam"
                and row["metric"] in {"target_logit_drop", "prediction_flip_rate"}
            ):
                combined.append(row)
    write(MAIN_TAB / "T07_explanation_faithfulness.csv", combined)

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2), sharex=True)
    for column, dataset in enumerate(("fashion", "cifar10", "sign")):
        for row_index, metric in enumerate(("target_logit_drop", "prediction_flip_rate")):
            axis = axes[row_index, column]
            for method in ("kaiming", "lowrank_wmf"):
                subset = sorted(
                    [r for r in combined if r["dataset"] == dataset and r["metric"] == metric and r["method"] == method],
                    key=lambda r: float(r["fraction"]),
                )
                x = np.array([100 * float(r["fraction"]) for r in subset])
                y = np.array([float(r["mean_candidate_minus_reference"]) for r in subset])
                low = np.array([float(r["ci95_low"]) for r in subset])
                high = np.array([float(r["ci95_high"]) for r in subset])
                axis.errorbar(x, y, yerr=np.vstack([y - low, high - y]), marker="o", capsize=3,
                              color=COLORS[method], label=METHOD_LABELS[method])
            axis.axhline(0, color="#555555", lw=0.7)
            axis.grid(alpha=0.2)
            if row_index == 0:
                axis.set_title(DATASET_LABELS[dataset])
            if row_index == 1:
                axis.set_xlabel("Deleted input pixels (%)")
    axes[0, 0].set_ylabel("Extra target-logit drop\nvs random deletion")
    axes[1, 0].set_ylabel("Extra prediction-flip rate\nvs random deletion")
    axes[0, -1].legend(frameon=False)
    fig.suptitle("Causal faithfulness of exact template explanations")
    fig.tight_layout()
    save(fig, MAIN_FIG, "F09_explanation_faithfulness")


def rotation_figure_and_table() -> None:
    aggregate = rows(ROOT / "rotation_invariance_results" / "rotation_aggregate.csv")
    summary = rows(ROOT / "rotation_invariance_results" / "rotation_summary.csv")
    compact = []
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in summary:
        grouped[(row["dataset"], row["model"], row["method"])].append(row)
    for (dataset, model, method), group in sorted(grouped.items()):
        compact.append(
            {
                "dataset": dataset,
                "model": model,
                "method": method,
                "runs": len(group),
                "mean_clean_accuracy": np.mean([float(r["clean_accuracy"]) for r in group]),
                "mean_worst_angle_accuracy": np.mean([float(r["worst_angle_accuracy"]) for r in group]),
                "mean_c4_prediction_agreement": np.mean([float(r["c4_prediction_agreement"]) for r in group]),
                "mean_c4_probability_l1": np.mean([float(r["c4_probability_l1"]) for r in group]),
            }
        )
    write(MAIN_TAB / "T08_rotation_invariance.csv", compact)

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), sharey=False)
    linestyles = {"standard": "--", "rotation_invariant": "-"}
    for axis, dataset in zip(axes, ("fashion", "cifar10", "sign")):
        for model in ("standard", "rotation_invariant"):
            subset = sorted(
                [r for r in aggregate if r["dataset"] == dataset and r["method"] == "lowrank_wmf" and r["model"] == model],
                key=lambda r: float(r["angle"]),
            )
            x = np.array([float(r["angle"]) for r in subset])
            y = np.array([float(r["accuracy_mean"]) for r in subset])
            ci = np.array([float(r["accuracy_ci95"]) for r in subset])
            label = "Standard CNN" if model == "standard" else "C4 orbit CNN"
            color = COLORS["kaiming"] if model == "standard" else COLORS["lowrank_wmf"]
            axis.plot(x, y, linestyles[model], color=color, marker="o", ms=3.5, label=label)
            axis.fill_between(x, y - ci, y + ci, color=color, alpha=0.12)
        axis.set_title(DATASET_LABELS[dataset])
        axis.set_xlabel("Rotation angle (degrees)")
        axis.set_xticks([-180, -90, 0, 90, 180])
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Accuracy (%)")
    axes[-1].legend(frameon=False)
    fig.suptitle("Rotation sweep: standard convolution versus exact C4 template orbits")
    fig.tight_layout()
    save(fig, MAIN_FIG, "F10_rotation_invariance")


def total_cost() -> None:
    aggregate = rows(ROOT / "total_cost_results" / "cost_aggregate.csv")
    paired = rows(ROOT / "total_cost_results" / "cost_paired.csv")
    write(MAIN_TAB / "T12_total_cost_to_target.csv", aggregate)
    write(SUPP_TAB / "S20_total_cost_paired.csv", paired)
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.5))
    for column, dataset in enumerate(("fashion", "cifar10", "sign")):
        for row_index, metric in enumerate(("mean_total_seconds_among_reached", "mean_net_joules_among_reached")):
            axis = axes[row_index, column]
            ci_key = "ci95_total_seconds_among_reached" if row_index == 0 else "ci95_net_joules_among_reached"
            for method in ("kaiming", "di_wmf", "lowrank_wmf"):
                subset = sorted([r for r in aggregate if r["dataset"] == dataset and r["method"] == method], key=lambda r: float(r["threshold_accuracy"]))
                x = np.array([float(r["threshold_accuracy"]) for r in subset])
                y = np.array([float(r[metric]) for r in subset])
                ci = np.array([float(r[ci_key]) for r in subset])
                axis.errorbar(x, y, yerr=ci, marker="o", capsize=3, color=COLORS[method], label=METHOD_LABELS[method])
            axis.grid(alpha=0.2)
            if row_index == 0:
                axis.set_title(DATASET_LABELS[dataset])
            else:
                axis.set_xlabel("Validation accuracy target (%)")
    axes[0, 0].set_ylabel("Time to target (s)")
    axes[1, 0].set_ylabel("Net GPU energy to target (J)")
    axes[0, -1].legend(frameon=False)
    fig.suptitle("Total compute cost to pre-specified accuracy targets")
    fig.tight_layout()
    save(fig, MAIN_FIG, "F11_time_energy_to_accuracy")


def correncoder_tradeoff() -> None:
    data = rows(ROOT / "correncoder_ablation_results" / "correncoder_aggregate.csv")
    write(MAIN_TAB / "T09_correncoder_classification_ablation.csv", data)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.1), sharex=True)
    for column, dataset in enumerate(("fashion", "cifar10", "sign")):
        for method in ("kaiming", "lowrank_wmf"):
            subset = sorted([r for r in data if r["dataset"] == dataset and r["method"] == method], key=lambda r: float(r["auxiliary_lambda"]))
            x = np.array([float(r["auxiliary_lambda"]) for r in subset])
            acc = np.array([float(r["mean_test_accuracy"]) for r in subset])
            acc_std = np.array([float(r["std_test_accuracy"]) for r in subset])
            rec = np.array([float(r["mean_test_auxiliary_loss"]) for r in subset])
            rec_std = np.array([float(r["std_test_auxiliary_loss"]) for r in subset])
            axes[0, column].errorbar(x, acc, yerr=acc_std, marker="o", capsize=3, color=COLORS[method], label=METHOD_LABELS[method])
            axes[1, column].errorbar(x, rec, yerr=rec_std, marker="o", capsize=3, color=COLORS[method], label=METHOD_LABELS[method])
        axes[0, column].set_title(DATASET_LABELS[dataset])
        axes[1, column].set_xlabel("Auxiliary reconstruction weight λ")
        axes[0, column].grid(alpha=0.2)
        axes[1, column].grid(alpha=0.2)
    axes[0, 0].set_ylabel("Test accuracy (%)")
    axes[1, 0].set_ylabel("Reconstruction loss")
    axes[0, -1].legend(frameon=False)
    fig.suptitle("Classification Correncoder: accuracy–reconstruction trade-off")
    fig.tight_layout()
    save(fig, MAIN_FIG, "F12_correncoder_classification_tradeoff")


def correncoder_regression() -> None:
    aggregate = rows(ROOT / "correncoder_regression_ablation_results" / "aggregate.csv")
    paired = rows(ROOT / "correncoder_regression_ablation_results" / "paired_comparisons.csv")
    write(MAIN_TAB / "T10_correncoder_regression_aggregate.csv", aggregate)
    write(MAIN_TAB / "T11_correncoder_regression_paired.csv", paired)
    chosen = ["random_mse", "pretrained_mse", "matched_1d_mse", "pretrained_corr", "pretrained_corr_spectral"]
    labels = {
        "random_mse": "Random + MSE", "pretrained_mse": "Pretrained + MSE",
        "matched_1d_mse": "Matched stem + MSE", "pretrained_corr": "Pretrained + corr",
        "pretrained_corr_spectral": "Pretrained + corr + spectrum",
    }
    indexed = {(r["experiment"], r["metric"]): r for r in aggregate}
    point_colours = {
        "random_mse": COLORS["kaiming"],
        "pretrained_mse": "#8073ac",
        "matched_1d_mse": COLORS["di_wmf"],
        "pretrained_corr": "#2ca25f",
        "pretrained_corr_spectral": COLORS["lowrank_wmf"],
    }
    fig = plt.figure(figsize=(13.2, 5.35))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.65, 1.0], wspace=0.28)
    pareto = fig.add_subplot(grid[0, 0])
    rr_axis = fig.add_subplot(grid[0, 1])

    # A two-objective view reveals the waveform trade-off directly.
    progression = ["pretrained_mse", "pretrained_corr", "pretrained_corr_spectral"]
    progression_x = [float(indexed[(exp, "test_mae")]["mean"]) for exp in progression]
    progression_y = [float(indexed[(exp, "waveform_correlation")]["mean"]) for exp in progression]
    pareto.plot(progression_x, progression_y, color="#bdbdbd", lw=1.4, zorder=1)
    annotation_offsets = {
        "random_mse": (-12, -27),
        "pretrained_mse": (10, 17),
        "matched_1d_mse": (10, -38),
        "pretrained_corr": (12, -2),
        "pretrained_corr_spectral": (12, 16),
    }
    short_labels = {
        "random_mse": "Random + MSE",
        "pretrained_mse": "Pretrained + MSE",
        "matched_1d_mse": "Matched stem + MSE",
        "pretrained_corr": "Pretrained + correlation",
        "pretrained_corr_spectral": "Pretrained + corr + spectrum",
    }
    for experiment in chosen:
        mae = float(indexed[(experiment, "test_mae")]["mean"])
        mae_ci = float(indexed[(experiment, "test_mae")]["ci95_half_width"])
        correlation = float(indexed[(experiment, "waveform_correlation")]["mean"])
        correlation_ci = float(indexed[(experiment, "waveform_correlation")]["ci95_half_width"])
        colour = point_colours[experiment]
        pareto.errorbar(
            mae,
            correlation,
            xerr=mae_ci,
            yerr=correlation_ci,
            fmt="o",
            ms=8,
            mfc=colour,
            mec="white",
            mew=0.8,
            ecolor=colour,
            elinewidth=1.2,
            capsize=3,
            alpha=0.9,
            zorder=3,
        )
        pareto.annotate(
            short_labels[experiment],
            (mae, correlation),
            xytext=annotation_offsets[experiment],
            textcoords="offset points",
            fontsize=8,
            color=colour,
            fontweight="bold" if experiment == "pretrained_corr_spectral" else "normal",
            arrowprops={"arrowstyle": "-", "color": colour, "lw": 0.7, "alpha": 0.7},
        )
    pareto.set_xlabel("waveform MAE  ← lower is better")
    pareto.set_ylabel("waveform correlation  → higher is better")
    pareto.set_title("Waveform objective plane", loc="left", fontweight="bold")
    pareto.grid(alpha=0.16)
    pareto.text(
        0.03,
        0.96,
        "better waveform region ↖",
        transform=pareto.transAxes,
        va="top",
        color="#2ca25f",
        fontsize=8.5,
        fontweight="bold",
    )

    # Respiratory-rate error tells a different story, so it is shown separately.
    y = np.arange(len(chosen))[::-1]
    rr_means = np.array([
        float(indexed[(experiment, "rr_mean_absolute_error_bpm_30p6s")]["mean"])
        for experiment in chosen
    ])
    rr_ci = np.array([
        float(indexed[(experiment, "rr_mean_absolute_error_bpm_30p6s")]["ci95_half_width"])
        for experiment in chosen
    ])
    best_rr = int(np.argmin(rr_means))
    rr_axis.axhspan(y[best_rr] - 0.42, y[best_rr] + 0.42, color="#f2eef8", zorder=-3)
    rr_axis.axvline(rr_means[best_rr], color="#8073ac", ls="--", lw=0.9, alpha=0.75)
    for position, experiment, mean, ci in zip(y, chosen, rr_means, rr_ci):
        colour = point_colours[experiment]
        rr_axis.errorbar(
            mean,
            position,
            xerr=ci,
            fmt="o",
            ms=7.5,
            mfc=colour,
            mec="white",
            mew=0.8,
            ecolor=colour,
            elinewidth=1.5,
            capsize=3,
            zorder=3,
        )
        rr_axis.text(mean, position + 0.22, f"{mean:.2f}", ha="center", fontsize=7.5, color=colour)
    rr_axis.set_yticks(y, [short_labels[experiment] for experiment in chosen])
    rr_axis.set_xlabel("respiratory-rate MAE (bpm)  ← lower is better")
    rr_axis.set_title("Rate recovery does not share the same optimum", loc="left", fontweight="bold")
    rr_axis.grid(axis="x", alpha=0.16)
    rr_axis.spines[["top", "right", "left"]].set_visible(False)
    rr_axis.tick_params(axis="y", length=0)

    fig.suptitle(
        "Correncoder objectives improve waveform fidelity without guaranteeing better rate estimates",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.018,
        "Means and 95% CIs over 53 BIDMC leave-one-subject-out folds; formal conclusions use paired tests.",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.94))
    save(fig, MAIN_FIG, "F13_correncoder_regression_results")


def representative_waveform() -> None:
    experiment_dirs = {
        "Random + MSE": ROOT / "outputs_correncoder_initialisation" / "random_mse",
        "Pretrained + MSE": ROOT / "outputs_correncoder_regression_full",
        "Pretrained + corr": ROOT / "outputs_correncoder_loss_ablation" / "pretrained_corr",
    }
    candidates = []
    reference_root = experiment_dirs["Pretrained + corr"]
    for metric_path in reference_root.glob("*/final_metrics.json"):
        metric = json.loads(metric_path.read_text(encoding="utf-8"))
        candidates.append((metric["fold"], float(metric["waveform_correlation"]), metric_path.parent.name))
    median = float(np.median([value for _, value, _ in candidates]))
    fold, _, directory_name = min(candidates, key=lambda item: abs(item[1] - median))
    series = {}
    target = None
    for label, directory in experiment_dirs.items():
        waveform_path = directory / directory_name / "waveforms.npz"
        if not waveform_path.exists():
            matches = list(directory.glob(f"bidmc_fold{fold:02d}_seed*/waveforms.npz"))
            if not matches:
                raise FileNotFoundError(f"Waveform for fold {fold} in {directory}")
            waveform_path = matches[0]
        archive = np.load(waveform_path)
        series[label] = np.asarray(archive["prediction"], dtype=float)
        if target is None:
            target = np.asarray(archive["target"], dtype=float)
    # Show a 30.6-second segment and use the highest-variance target window.
    window = min(765, len(target))
    stride = max(1, window // 4)
    starts = range(0, len(target) - window + 1, stride)
    start = max(starts, key=lambda s: float(np.var(target[s : s + window])))
    time = np.arange(window) / 25.0
    fig, axis = plt.subplots(figsize=(11.5, 3.9))
    axis.plot(time, target[start : start + window], color="#111111", lw=1.8, label="Reference respiration")
    for label, color in zip(series, ("#969696", "#8073ac", "#b2182b")):
        axis.plot(time, series[label][start : start + window], lw=1.0, alpha=0.9, color=color, label=label)
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Normalised amplitude")
    axis.set_title(f"Representative BIDMC reconstruction (LOSO fold {fold}, median-correlation subject)")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, ncol=2)
    fig.tight_layout()
    save(fig, MAIN_FIG, "F14_correncoder_representative_waveform")


def supplementary_figures_and_tables() -> None:
    # Existing diagnostics/ablations that are useful but too detailed for the main chapter.
    figure_copies = {
        BASE / "figure_first_layer_selectivity.png": "S01_first_layer_selectivity.png",
        BASE / "figure_first_layer_corruptions.png": "S02_first_layer_corruption_effects.png",
        ROOT / "correncoder_extension_results" / "paired_improvements.png": "S03_correncoder_layerwise_calibration.png",
        ROOT / "correncoder_regression_ablation_results" / "correncoder_ablation.png": "S04_correncoder_objective_ablation.png",
        ROOT / "correncoder_regression_results" / "correncoder_loso_summary.png": "S05_correncoder_loso_distribution.png",
    }
    for source, target in figure_copies.items():
        if source.exists():
            shutil.copy2(source, SUPP_FIG / target)

    # Representative template grids for both methods and all three tasks.
    for dataset, directory in (
        ("fashion", ROOT / "template_semantics_results" / "expanded_500" / "figures"),
        ("cifar10", ROOT / "template_semantics_results" / "expanded_500" / "figures"),
        ("sign", ROOT / "template_semantics_results" / "sign_full_10seed" / "figures"),
    ):
        for method in ("kaiming", "lowrank_wmf"):
            source = directory / f"{dataset}_standard_{method}_seed0.png"
            shutil.copy2(source, SUPP_FIG / f"S06_template_grid_{dataset}_{method}.png")

    # Sign 100-epoch curve.
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for config_path in (ROOT / "outputs_convergence100_sign").glob("*/config.json"):
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        method = cfg.get("init")
        if method not in {"kaiming", "lowrank_wmf"}:
            continue
        history = rows(config_path.parent / "history.csv")
        grouped[method].append(np.array([float(r["val_accuracy"]) for r in history]))
    fig, axis = plt.subplots(figsize=(7.5, 4.1))
    for method in ("kaiming", "lowrank_wmf"):
        values = np.vstack(grouped[method])
        mean = values.mean(0)
        ci = 2.776 * values.std(0, ddof=1) / math.sqrt(values.shape[0])
        epoch = np.arange(values.shape[1])
        axis.plot(epoch, mean, color=COLORS[method], label=METHOD_LABELS[method])
        axis.fill_between(epoch, mean - ci, mean + ci, color=COLORS[method], alpha=0.14)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Validation accuracy (%)")
    axis.set_title("Sign convergence over 100 epochs")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    fig.tight_layout()
    save(fig, SUPP_FIG, "S07_sign_100_epoch_convergence")

    # CIFAR augmentation and unstructured-pruning diagnostics.
    aug = rows(ROOT / "augmentation_results" / "augmentation_aggregate.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.7, 3.7))
    for axis, metric, title in zip(axes, ("mean_test_accuracy", "mean_validation_aulc"), ("Final test accuracy", "Post-update AULC")):
        x = np.arange(2)
        for i, method in enumerate(("kaiming", "lowrank_wmf")):
            indexed = {(r["augmentation"], r["method"]): r for r in aug}
            values = [float(indexed[(a, method)][metric]) for a in ("none", "cifar_standard")]
            axis.bar(x + (i - 0.5) * 0.34, values, 0.34, color=COLORS[method], label=METHOD_LABELS[method])
        axis.set_xticks(x, ["None", "Crop + flip"])
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    axes[-1].legend(frameon=False)
    fig.tight_layout()
    save(fig, SUPP_FIG, "S08_cifar_augmentation")

    pruning = rows(ROOT / "pruning_results" / "full20_standard_5seed" / "pruning_aggregate.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for axis, dataset in zip(axes, ("fashion", "cifar10", "sign")):
        for method in ("kaiming", "lowrank_wmf"):
            subset = sorted([r for r in pruning if r["dataset"] == dataset and r["method"] == method], key=lambda r: float(r["sparsity"]))
            axis.plot([100 * float(r["sparsity"]) for r in subset], [float(r["mean_accuracy"]) for r in subset], marker="o", color=COLORS[method], label=METHOD_LABELS[method])
        axis.set_title(DATASET_LABELS[dataset])
        axis.set_xlabel("Unstructured sparsity (%)")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Test accuracy (%)")
    axes[-1].legend(frameon=False)
    fig.tight_layout()
    save(fig, SUPP_FIG, "S09_unstructured_pruning")

    holm = rows(ROOT / "statistical_corrections" / "family_summary.csv")
    fig, axis = plt.subplots(figsize=(8.5, 5.4))
    holm = sorted(holm, key=lambda r: int(r["hypotheses"]), reverse=True)
    y = np.arange(len(holm))
    total = np.array([int(r["hypotheses"]) for r in holm])
    significant = np.array([int(r["holm_significant"]) for r in holm])
    axis.barh(y, total, color="#d9d9d9", label="All paired hypotheses")
    axis.barh(y, significant, color=COLORS["lowrank_wmf"], label="Holm-significant")
    axis.set_yticks(y, [r["family"].replace("_", " ") for r in holm])
    axis.invert_yaxis()
    axis.set_xlabel("Number of hypotheses")
    axis.set_title("Multiple-comparison audit by research family")
    axis.legend(frameon=False)
    axis.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    save(fig, SUPP_FIG, "S10_holm_family_summary")


def copy_tables() -> None:
    holm_rows = rows(ROOT / "statistical_corrections" / "all_core_hypotheses_holm.csv")
    corruption_compact = []
    for row in holm_rows:
        if row["family"] != "natural_corruption" or row["metric"] != "normalised_accuracy_auc":
            continue
        hypothesis = json.loads(row["hypothesis"])
        if hypothesis.get("method") != "lowrank_wmf":
            continue
        corruption_compact.append(
            {
                "dataset": row["dataset"],
                "corruption": hypothesis["corruption"],
                "paired_seeds": row["n_paired"],
                "lowrank_minus_kaiming_auc": row["paired_difference"],
                "ci95_low": row["ci95_low"],
                "ci95_high": row["ci95_high"],
                "holm_adjusted_p": row["p_value_holm"],
                "holm_significant_0p05": row["holm_significant_0p05"],
                "paired_cohen_dz": row["cohen_dz"],
            }
        )
    write(MAIN_TAB / "T05_natural_corruptions.csv", corruption_compact)

    main = {
        BASE / "table_main_cross_task.csv": "T01_cross_task_standardcnn.csv",
        BASE / "table_first_layer_cross_task.csv": "T02_first_layer_prior_ablation.csv",
        BASE / "table_main_reliability.csv": "T04_reliability.csv",
        BASE / "table_main_structured_efficiency.csv": "T06_structured_efficiency.csv",
        BASE / "table_expanded_template_semantics.csv": "T07_template_semantics.csv",
        BASE / "table_holm_family_summary.csv": "T13_holm_family_summary.csv",
    }
    supplementary = {
        BASE / "table_first_layer_rankings.csv": "S01_first_layer_rankings.csv",
        BASE / "table_first_layer_paired_training.csv": "S02_first_layer_paired_training.csv",
        BASE / "table_first_layer_paired_selectivity.csv": "S03_first_layer_selectivity.csv",
        BASE / "table_first_layer_paired_corruptions.csv": "S04_first_layer_corruptions.csv",
        ROOT / "convergence_results" / "paired.csv": "S05_convergence_50epoch_paired.csv",
        ROOT / "convergence_results" / "sign100_paired.csv": "S06_sign_100epoch_paired.csv",
        ROOT / "efficiency_results" / "aggregate.csv": "S07_width_efficiency_aggregate.csv",
        ROOT / "efficiency_results" / "paired.csv": "S08_width_efficiency_paired.csv",
        ROOT / "pruning_results" / "full20_standard_5seed" / "pruning_aggregate.csv": "S09_unstructured_pruning.csv",
        ROOT / "structured_efficiency_results" / "full20_standard" / "structured_retention_paired.csv": "S10_structured_pruning_paired.csv",
        ROOT / "reliability_results" / "full20_standard" / "calibration_paired.csv": "S11_calibration_paired.csv",
        ROOT / "reliability_results" / "full20_standard" / "ood_paired.csv": "S12_ood_paired.csv",
        ROOT / "reliability_results" / "full20_standard" / "adversarial_paired.csv": "S13_adversarial_paired.csv",
        ROOT / "corruption_results" / "full20_cross_task_paired.csv": "S13b_full_natural_corruptions.csv",
        ROOT / "rotation_invariance_results" / "rotation_aggregate.csv": "S14_rotation_angle_aggregate.csv",
        ROOT / "correncoder_ablation_results" / "correncoder_architecture_and_lambda_paired.csv": "S15_correncoder_lambda_paired.csv",
        ROOT / "correncoder_extension_results" / "aggregate.csv": "S16_correncoder_layerwise_aggregate.csv",
        ROOT / "correncoder_extension_results" / "paired_comparisons.csv": "S17_correncoder_layerwise_paired.csv",
        ROOT / "correncoder_completed_extension_results" / "aggregate.csv": "S18_correncoder_depth_lag_aggregate.csv",
        ROOT / "correncoder_completed_extension_results" / "paired_comparisons.csv": "S19_correncoder_depth_lag_paired.csv",
        ROOT / "augmentation_results" / "augmentation_aggregate.csv": "S21_cifar_augmentation.csv",
        ROOT / "statistical_corrections" / "all_core_hypotheses_holm.csv": "S22_all_holm_corrected_hypotheses.csv",
    }
    for source, target in main.items():
        shutil.copy2(source, MAIN_TAB / target)
    for source, target in supplementary.items():
        shutil.copy2(source, SUPP_TAB / target)


def catalogue() -> None:
    captions = [
        ("F01", "Cross-task paired effects", "Low-rank DI-WMF minus Kaiming for Epoch-0 accuracy, AULC and final accuracy; points show paired means and bars show 95% CIs."),
        ("F02", "Fifty-epoch learning curves", "Mean validation accuracy over ten paired seeds; shaded bands are 95% CIs."),
        ("F03", "Cross-architecture transfer", "Paired low-rank DI-WMF effects on ResNet18 and classification Correncoder. Asterisks denote primary Holm-adjusted p < 0.05."),
        ("F04", "Natural corruption effects", "Difference in normalised corruption AUC between low-rank DI-WMF and Kaiming across eight corruptions."),
        ("F05", "Parameter efficiency", "Accuracy as a function of trainable parameters for full-, half- and quarter-width networks."),
        ("F06", "Structured pruning", "Accuracy versus FLOP reduction after structured channel pruning."),
        ("F07", "Reliability", "Calibrated ECE, OOD error and PGD retention error; lower values are better."),
        ("F08", "Template semantics", "Semantic purity and anchor alignment of learned templates."),
        ("F09", "Explanation faithfulness", "Extra target-logit drop and prediction flips produced by exact template explanations compared with random deletion."),
        ("F10", "Rotation invariance", "Accuracy over a continuous angle sweep for standard and exact C4-orbit matched-template CNNs."),
        ("F11", "Time and energy to target", "Wall-clock time and net GPU energy required to reach pre-specified validation thresholds."),
        ("F12", "Classification Correncoder trade-off", "Test accuracy and reconstruction loss across auxiliary-loss weights."),
        ("F13", "Correncoder regression aggregate", "Waveform and respiratory-rate performance over all 53 BIDMC LOSO folds."),
        ("F14", "Representative waveform", "Reference respiration and three Correncoder reconstructions for a median-correlation subject."),
        ("F15", "Correncoder depth/covariance/lag effects", "Paired subject-level forest plot; positive values consistently indicate improvement."),
        ("F16", "First-layer prior ablation", "Controlled comparison in which only the first convolutional layer differs."),
    ]
    lines = [
        "# Complete dissertation figure and table catalogue",
        "",
        "Generated only from completed non-smoke experiments. PNG files are 300 dpi where regenerated; matching PDF files are vector figures where available.",
        "",
        "## Main figures",
        "",
        "| ID | Title | Suggested caption |",
        "|---|---|---|",
    ]
    lines.extend(f"| {i} | {title} | {caption} |" for i, title, caption in captions)
    lines += [
        "",
        "## Main tables",
        "",
        "| ID | Content |",
        "|---|---|",
        "| T01 | Cross-task StandardCNN comparison |",
        "| T02 | Controlled first-layer prior ablation |",
        "| T03 | Cross-architecture effects with Holm-adjusted inference |",
        "| T04–T05 | Reliability and corruption robustness |",
        "| T06 | Structured efficiency |",
        "| T07 | Template semantics and explanation faithfulness |",
        "| T08 | Rotation invariance |",
        "| T09 | Classification Correncoder loss ablation |",
        "| T10–T11 | Full Correncoder regression aggregates and paired contrasts |",
        "| T12 | Time and energy to target accuracy |",
        "| T13 | Multiple-comparison family summary |",
        "",
        "## Statistical notation",
        "",
        "All uncertainty bars are 95% confidence intervals unless explicitly labelled as standard deviations. Formal claims should use the primary Holm-adjusted p-values in T03/T13 or S22; an unadjusted confidence interval alone is not sufficient after the multiple-comparison audit.",
        "",
        "## Supplement",
        "",
        "Supplementary figures S01–S10 contain detailed ablations, template grids, 100-epoch Sign convergence, augmentation, pruning and the Holm audit. Supplementary tables S01–S22 retain complete aggregate or paired outputs without per-batch smoke-test data.",
    ]
    (PACKAGE / "FIGURE_TABLE_CATALOG.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = {
        "main_figures_png": sorted(p.name for p in MAIN_FIG.glob("*.png")),
        "main_figures_pdf": sorted(p.name for p in MAIN_FIG.glob("*.pdf")),
        "supplementary_figures_png": sorted(p.name for p in SUPP_FIG.glob("*.png")),
        "supplementary_figures_pdf": sorted(p.name for p in SUPP_FIG.glob("*.pdf")),
        "main_tables": sorted(p.name for p in MAIN_TAB.glob("*.csv")),
        "supplementary_tables": sorted(p.name for p in SUPP_TAB.glob("*.csv")),
        "exclusions": ["all outputs_*_smoke directories", "per-batch raw power samples", "duplicate pilot tables"],
    }
    (PACKAGE / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    style()
    make_thesis_artifacts.main()
    for directory in (MAIN_FIG, SUPP_FIG, MAIN_TAB, SUPP_TAB):
        directory.mkdir(parents=True, exist_ok=True)
    copy_existing_main_figures()
    cross_architecture()
    explanation_faithfulness()
    rotation_figure_and_table()
    total_cost()
    correncoder_tradeoff()
    correncoder_regression()
    representative_waveform()
    supplementary_figures_and_tables()
    copy_tables()
    catalogue()
    print(f"Complete thesis package written to {PACKAGE}")


if __name__ == "__main__":
    main()
