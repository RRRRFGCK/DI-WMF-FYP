"""Create dissertation-ready main tables and figures from completed studies."""

import csv
import json
import math
import shutil
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "thesis_artifacts"
T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
}
METHOD_LABELS = {"kaiming": "Kaiming", "lowrank_wmf": "Low-rank DI-WMF"}
COLOURS = {"kaiming": "#555555", "lowrank_wmf": "#b2182b"}


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def interval(values):
    mean = statistics.mean(values)
    if len(values) < 2:
        return mean, mean, mean
    half = T95[len(values) - 1] * statistics.stdev(values) / math.sqrt(len(values))
    return mean, mean - half, mean + half


def learning_curve_figure():
    grouped = defaultdict(lambda: defaultdict(list))
    for config_path in (ROOT / "outputs_convergence50").glob("*/config.json"):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("init") not in METHOD_LABELS:
            continue
        history_path = config_path.parent / "history.csv"
        if not history_path.exists():
            continue
        history = read_csv(history_path)
        grouped[(config["dataset"], config["init"])][config["seed"]] = [
            float(row["val_accuracy"]) for row in history
        ]
    datasets = ["fashion", "cifar10", "sign"]
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.7), sharex=True)
    for axis, dataset in zip(axes, datasets):
        for method in METHOD_LABELS:
            curves = list(grouped[(dataset, method)].values())
            values = np.asarray(curves)
            means = values.mean(0)
            critical = T95.get(values.shape[0] - 1, 1.96)
            half = critical * values.std(0, ddof=1) / math.sqrt(values.shape[0])
            epochs = np.arange(values.shape[1])
            axis.plot(epochs, means, color=COLOURS[method], label=METHOD_LABELS[method])
            axis.fill_between(epochs, means - half, means + half, color=COLOURS[method], alpha=0.16)
        axis.set_title(dataset.replace("fashion", "Fashion-MNIST").replace("cifar10", "CIFAR-10").replace("sign", "Sign"))
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Validation accuracy (%)")
    axes[-1].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(OUTPUT / "figure_learning_curves_50epoch.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def convergence_effect_table_and_figure():
    rows = read_csv(ROOT / "convergence_results" / "paired.csv")
    metrics = ["initial_val_accuracy", "validation_aulc", "test_accuracy"]
    labels = {"initial_val_accuracy": "Epoch-0", "validation_aulc": "Post-update AULC", "test_accuracy": "Final test"}
    selected = [
        row for row in rows
        if row["method"] == "lowrank_wmf_rank16" and row["metric"] in metrics
    ]
    output = [
        {
            "dataset": row["dataset"], "metric": labels[row["metric"]],
            "kaiming_mean": row["reference_mean"], "lowrank_mean": row["method_mean"],
            "paired_difference": row["mean_paired_difference"],
            "ci95_low": row["ci95_low"], "ci95_high": row["ci95_high"],
            "lowrank_wins": row["method_wins"],
        }
        for row in selected
    ]
    write_csv(OUTPUT / "table_main_cross_task.csv", output)
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    datasets = ["fashion", "cifar10", "sign"]
    for axis, metric in zip(axes, metrics):
        subset = {row["dataset"]: row for row in selected if row["metric"] == metric}
        means = np.array([float(subset[dataset]["mean_paired_difference"]) for dataset in datasets])
        lows = np.array([float(subset[dataset]["ci95_low"]) for dataset in datasets])
        highs = np.array([float(subset[dataset]["ci95_high"]) for dataset in datasets])
        axis.errorbar(
            means, np.arange(3), xerr=np.vstack([means - lows, highs - means]),
            fmt="o", color="#b2182b", capsize=4,
        )
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_yticks(np.arange(3), ["Fashion", "CIFAR-10", "Sign"])
        axis.set_title(labels[metric])
        axis.set_xlabel("DI-WMF − Kaiming (points)")
        axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(OUTPUT / "figure_cross_task_effects.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def parameter_efficiency_figure():
    rows = read_csv(ROOT / "efficiency_results" / "aggregate.csv")
    rows = [
        row for row in rows
        if row["method"] in METHOD_LABELS
        and row["model"] in {"standard", "standard_half", "standard_quarter"}
    ]
    datasets = ["fashion", "cifar10", "sign"]
    model_order = ["standard_quarter", "standard_half", "standard"]
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.7))
    for axis, dataset in zip(axes, datasets):
        for method in METHOD_LABELS:
            indexed = {
                row["model"]: row for row in rows
                if row["dataset"] == dataset and row["method"] == method
            }
            xs = [float(indexed[model]["parameter_count"]) for model in model_order]
            ys = [float(indexed[model]["mean_test_accuracy"]) for model in model_order]
            axis.plot(xs, ys, marker="o", color=COLOURS[method], label=METHOD_LABELS[method])
        axis.set_xscale("log")
        axis.set_title(dataset.replace("fashion", "Fashion-MNIST").replace("cifar10", "CIFAR-10").replace("sign", "Sign"))
        axis.set_xlabel("Trainable parameters (log scale)")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Test accuracy (%)")
    axes[-1].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(OUTPUT / "figure_parameter_efficiency.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def corruption_heatmap():
    rows = read_csv(ROOT / "corruption_results" / "full20_cross_task_paired.csv")
    selected = [
        row for row in rows
        if row["method"] == "lowrank_wmf"
        and row["metric"] == "normalised_accuracy_auc"
    ]
    datasets = sorted({row["dataset"] for row in selected})
    corruptions = sorted({row["corruption"] for row in selected})
    indexed = {(row["dataset"], row["corruption"]): float(row["mean_paired_difference"]) for row in selected}
    matrix = np.array([[indexed[(dataset, corruption)] for corruption in corruptions] for dataset in datasets])
    limit = max(abs(matrix.min()), abs(matrix.max()))
    figure, axis = plt.subplots(figsize=(11, 2.8))
    image = axis.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    axis.set_xticks(np.arange(len(corruptions)), [value.replace("_", " ") for value in corruptions], rotation=35, ha="right")
    axis.set_yticks(np.arange(len(datasets)), datasets)
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(column_index, row_index, f"{matrix[row_index, column_index]:+.1f}", ha="center", va="center", fontsize=8)
    axis.set_title("Relative corruption AUC: low-rank DI-WMF − Kaiming")
    figure.colorbar(image, ax=axis, label="percentage points")
    figure.tight_layout()
    figure.savefig(OUTPUT / "figure_corruption_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def semantic_figure():
    expanded_rows = read_csv(
        ROOT / "template_semantics_results" / "expanded_500"
        / "template_semantic_summary.csv"
    )
    sign_rows = read_csv(
        ROOT / "template_semantics_results" / "sign_full_10seed"
        / "template_semantic_summary.csv"
    )
    rows = [row for row in expanded_rows if row["dataset"] in {"fashion", "cifar10"}]
    rows.extend(row for row in sign_rows if row["dataset"] == "sign")
    paired_rows = [
        row
        for row in read_csv(
            ROOT / "template_semantics_results" / "expanded_500"
            / "template_semantic_paired.csv"
        )
        if row["dataset"] in {"fashion", "cifar10"}
    ]
    paired_rows.extend(
        read_csv(
            ROOT / "template_semantics_results" / "sign_full_10seed"
            / "template_semantic_paired.csv"
        )
    )
    datasets = ["fashion", "cifar10", "sign"]
    metrics = [
        (
            "mean_mean_semantic_purity",
            "mean_semantic_purity",
            "Human semantic purity is task-dependent",
        ),
        (
            "mean_mean_anchor_alignment",
            "mean_anchor_alignment",
            "Stored-anchor alignment consistently increases",
        ),
    ]
    figure = plt.figure(figsize=(12.6, 6.7))
    grid = figure.add_gridspec(2, 2, height_ratios=[1.35, 1.0], hspace=0.40, wspace=0.24)
    absolute_axes = [figure.add_subplot(grid[0, index]) for index in range(2)]
    effect_axes = [figure.add_subplot(grid[1, index]) for index in range(2)]
    positions = np.arange(len(datasets))[::-1]
    indexed = {(row["dataset"], row["method"]): row for row in rows}
    paired_indexed = {(row["dataset"], row["metric"]): row for row in paired_rows}
    labels = ["Fashion-MNIST", "CIFAR-10", "Sign"]

    for column, (absolute_metric, paired_metric, title) in enumerate(metrics):
        axis = absolute_axes[column]
        for position, dataset in zip(positions, datasets):
            kaiming = float(indexed[(dataset, "kaiming")][absolute_metric])
            informed = float(indexed[(dataset, "lowrank_wmf")][absolute_metric])
            axis.plot([kaiming, informed], [position, position], color="#c9c9c9", lw=2.2, zorder=1)
            axis.scatter(kaiming, position, s=68, color=COLOURS["kaiming"], edgecolor="white", lw=0.8, zorder=3)
            axis.scatter(informed, position, s=78, color=COLOURS["lowrank_wmf"], edgecolor="white", lw=0.8, zorder=3)
        axis.set_yticks(positions, labels)
        axis.set_xlim((0.36, 0.76) if column == 0 else (0.0, 0.25))
        axis.set_xlabel("mean score")
        axis.set_title(title, loc="left", fontweight="bold")
        axis.grid(axis="x", alpha=0.16)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)

        effect_axis = effect_axes[column]
        for position, dataset in zip(positions, datasets):
            paired = paired_indexed[(dataset, paired_metric)]
            mean = float(paired["mean_candidate_minus_reference"])
            low = float(paired["ci95_low"])
            high = float(paired["ci95_high"])
            colour = COLOURS["lowrank_wmf"] if low > 0 else "#8c8c8c"
            effect_axis.plot([low, high], [position, position], color=colour, lw=2.0)
            effect_axis.scatter(mean, position, s=58, color=colour, edgecolor="white", lw=0.7, zorder=3)
            effect_axis.text(high + 0.006, position, f"{mean:+.3f}", va="center", fontsize=8, color=colour)
        effect_axis.axvline(0, color="#555555", lw=0.9, ls="--")
        effect_axis.set_xlim((-0.07, 0.09) if column == 0 else (-0.02, 0.17))
        effect_axis.set_yticks(positions, labels)
        effect_axis.set_xlabel("paired DI-WMF minus Kaiming (95% CI)")
        effect_axis.set_title("Paired effect", loc="left", fontsize=10)
        effect_axis.grid(axis="x", alpha=0.16)
        effect_axis.spines[["top", "right", "left"]].set_visible(False)
        effect_axis.tick_params(axis="y", length=0)

    figure.suptitle(
        "A known template origin is more consistent than a simple human-semantic label",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    figure.text(
        0.5,
        0.955,
        "Dumbbells show absolute means; lower panels show paired differences with 95% confidence intervals.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    figure.text(0.47, 0.91, "●  Kaiming", ha="right", fontsize=9, color=COLOURS["kaiming"])
    figure.text(0.53, 0.91, "●  Low-rank DI-WMF", ha="left", fontsize=9, color=COLOURS["lowrank_wmf"])
    figure.tight_layout(rect=(0, 0, 1, 0.86))
    figure.savefig(OUTPUT / "figure_template_semantics.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def expanded_explanation_tables():
    expanded = ROOT / "template_semantics_results" / "expanded_500"
    sign = ROOT / "template_semantics_results" / "sign_full_10seed"

    semantic_rows = [
        row for row in read_csv(expanded / "template_semantic_paired.csv")
        if row["dataset"] in {"fashion", "cifar10"}
    ]
    semantic_rows.extend(read_csv(sign / "template_semantic_paired.csv"))
    write_csv(OUTPUT / "table_expanded_template_semantics.csv", semantic_rows)

    explanation_rows = []
    for path in (
        expanded / "explanation_paired.csv",
        sign / "explanation_paired.csv",
    ):
        explanation_rows.extend(
            row for row in read_csv(path)
            if row["reference_explainer"] == "random"
            and row["candidate_explainer"] == "exact_template_cam"
        )
    write_csv(
        OUTPUT / "table_expanded_explanation_faithfulness.csv",
        explanation_rows,
    )


def sign_ten_seed_table():
    rows = []
    root = ROOT / "outputs_full20_sign_bs64"
    for directory in root.iterdir():
        config_path = directory / "config.json"
        metrics_path = directory / "final_metrics.json"
        if not config_path.exists() or not metrics_path.exists():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not (
            config.get("dataset") == "sign"
            and config.get("model") == "standard"
            and config.get("init") in METHOD_LABELS
            and int(config.get("epochs", -1)) == 20
            and 0 <= int(config.get("seed", -1)) < 10
        ):
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "seed": int(config["seed"]),
                "method": config["init"],
                "initial_accuracy": float(metrics["initial_val_accuracy"]),
                "test_accuracy": float(metrics["test_accuracy"]),
                "aulc": float(metrics["validation_aulc"]),
            }
        )

    indexed = {(row["seed"], row["method"]): row for row in rows}
    table = []
    for metric in ("initial_accuracy", "test_accuracy", "aulc"):
        kaiming = [indexed[(seed, "kaiming")][metric] for seed in range(10)]
        lowrank = [indexed[(seed, "lowrank_wmf")][metric] for seed in range(10)]
        differences = [candidate - reference for reference, candidate in zip(kaiming, lowrank)]
        mean, low, high = interval(differences)
        table.append(
            {
                "dataset": "sign",
                "metric": metric,
                "paired_seeds": 10,
                "kaiming_mean": statistics.mean(kaiming),
                "lowrank_mean": statistics.mean(lowrank),
                "mean_lowrank_minus_kaiming": mean,
                "ci95_low": low,
                "ci95_high": high,
            }
        )
    write_csv(OUTPUT / "table_sign_10seed_training.csv", table)


def first_layer_cross_task_artifacts():
    source = ROOT / "first_layer_cross_task_results"
    required = [
        source / "main_metrics.csv",
        source / "method_rankings.csv",
        source / "paired_training.csv",
        source / "paired_selectivity.csv",
        source / "first_layer_cross_task_main.png",
        source / "first_layer_selectivity_effects.png",
        source / "corruption_effect_summary.csv",
        source / "first_layer_corruption_effects.png",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise RuntimeError(
            "Run analyze_first_layer_cross_task.py before artifact generation: "
            f"{missing}"
        )
    copies = {
        "main_metrics.csv": "table_first_layer_cross_task.csv",
        "method_rankings.csv": "table_first_layer_rankings.csv",
        "paired_training.csv": "table_first_layer_paired_training.csv",
        "paired_selectivity.csv": "table_first_layer_paired_selectivity.csv",
        "first_layer_cross_task_main.png": "figure_first_layer_cross_task.png",
        "first_layer_selectivity_effects.png": "figure_first_layer_selectivity.png",
        "corruption_effect_summary.csv": "table_first_layer_corruption_summary.csv",
        "first_layer_corruption_effects.png": "figure_first_layer_corruptions.png",
    }
    for source_name, target_name in copies.items():
        shutil.copy2(source / source_name, OUTPUT / target_name)
    optional_corruption = source / "paired_corruptions.csv"
    if optional_corruption.exists():
        shutil.copy2(optional_corruption, OUTPUT / "table_first_layer_paired_corruptions.csv")


def statistical_correction_artifacts():
    source = ROOT / "statistical_corrections"
    copies = {
        "all_core_hypotheses_holm.csv": "table_all_core_holm.csv",
        "family_summary.csv": "table_holm_family_summary.csv",
        "holm_significant_hypotheses.csv": "table_holm_significant.csv",
        "significance_lost_after_holm.csv": "table_significance_lost_after_holm.csv",
        "REPORT.md": "STATISTICAL_CORRECTION_REPORT.md",
    }
    missing = [source / name for name in copies if not (source / name).exists()]
    if missing:
        raise RuntimeError(
            "Run analyze_multiple_comparisons.py before artifact generation: "
            f"{missing}"
        )
    for source_name, target_name in copies.items():
        shutil.copy2(source / source_name, OUTPUT / target_name)


def new_extension_tables_and_figures():
    reliability = ROOT / "reliability_results" / "full20_standard"
    structured = ROOT / "structured_efficiency_results" / "full20_standard"
    correncoder = ROOT / "correncoder_ablation_results"
    correncoder_extension = ROOT / "correncoder_extension_results"
    correncoder_completed = ROOT / "correncoder_completed_extension_results"
    augmentation = ROOT / "augmentation_results"
    required = [
        reliability / "calibration_aggregate.csv",
        reliability / "ood_aggregate.csv",
        reliability / "adversarial_aggregate.csv",
        structured / "structured_accuracy_aggregate.csv",
        structured / "structured_benchmark_aggregate.csv",
        correncoder / "correncoder_aggregate.csv",
        correncoder_extension / "aggregate.csv",
        correncoder_extension / "paired_comparisons.csv",
        correncoder_extension / "paired_improvements.png",
        correncoder_completed / "aggregate.csv",
        correncoder_completed / "paired_comparisons.csv",
        correncoder_completed / "paired_improvements.png",
        correncoder_completed / "initialisation_diagnostics_aggregate.csv",
        augmentation / "augmentation_aggregate.csv",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Run all extension evaluators before artifact generation: {missing}")

    calibration_rows = read_csv(required[0])
    ood_rows = read_csv(required[1])
    adversarial_rows = read_csv(required[2])
    reliability_table = []
    for dataset in ("fashion", "cifar10", "sign"):
        for method in METHOD_LABELS:
            calibration = next(row for row in calibration_rows if row["dataset"] == dataset and row["method"] == method and row["calibration"] == "temperature")
            ood = next(row for row in ood_rows if row["dataset"] == dataset and row["method"] == method)
            adversarial = next(row for row in adversarial_rows if row["dataset"] == dataset and row["method"] == method and row["attack"] == "pgd")
            reliability_table.append(
                {
                    "dataset": dataset, "method": method,
                    "calibrated_ece": calibration["mean_ece15"],
                    "calibrated_nll": calibration["mean_nll"],
                    "uncertainty_aurc": calibration["mean_aurc"],
                    "ood_msp_auroc": ood["mean_msp_auroc"],
                    "ood_msp_fpr95": ood["mean_msp_fpr95"],
                    "pgd_accuracy": adversarial["mean_accuracy"],
                    "pgd_retention": adversarial["mean_retention"],
                }
            )
    write_csv(OUTPUT / "table_main_reliability.csv", reliability_table)

    structured_accuracy = read_csv(required[3])
    structured_benchmark = read_csv(required[4])
    structured_table = []
    for row in structured_accuracy:
        benchmark = next(
            item for item in structured_benchmark
            if item["dataset"] == row["dataset"] and item["method"] == row["method"]
            and item["sparsity"] == row["sparsity"] and item["batch_size"] == "1"
        )
        structured_table.append(
            {
                "dataset": row["dataset"], "method": row["method"],
                "channel_sparsity": row["sparsity"],
                "accuracy": row["mean_pruned_accuracy"],
                "accuracy_retention": row["mean_accuracy_retention"],
                "parameter_reduction": row["mean_parameter_reduction"],
                "flop_reduction": row["mean_flop_reduction"],
                "batch1_latency_ms": benchmark["mean_latency_median_ms"],
                "parameter_bytes": benchmark["mean_parameter_bytes"],
                "estimated_model_plus_peak_bytes": benchmark["mean_estimated_model_plus_peak_bytes"],
                "peak_incremental_memory_bytes": benchmark["mean_peak_incremental_memory_bytes"],
                "gross_joules_per_batch": benchmark["mean_gross_joules_per_batch"],
            }
        )
    write_csv(OUTPUT / "table_main_structured_efficiency.csv", structured_table)

    corr_rows = read_csv(required[5])
    write_csv(OUTPUT / "table_correncoder_ablation.csv", corr_rows)
    write_csv(
        OUTPUT / "table_correncoder_extension_aggregate.csv", read_csv(required[6])
    )
    write_csv(
        OUTPUT / "table_correncoder_extension_paired.csv", read_csv(required[7])
    )
    shutil.copy2(required[8], OUTPUT / "figure_correncoder_extension_paired.png")
    write_csv(
        OUTPUT / "table_correncoder_depth_softlag_aggregate.csv", read_csv(required[9])
    )
    write_csv(
        OUTPUT / "table_correncoder_depth_softlag_paired.csv", read_csv(required[10])
    )
    shutil.copy2(
        required[11], OUTPUT / "figure_correncoder_depth_softlag_paired.png"
    )
    write_csv(
        OUTPUT / "table_correncoder_initialisation_diagnostics.csv",
        read_csv(required[12]),
    )
    aug_rows = read_csv(required[13])
    write_csv(OUTPUT / "table_cifar_augmentation.csv", aug_rows)

    figure, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for axis, dataset in zip(axes, ("fashion", "cifar10", "sign")):
        for method in METHOD_LABELS:
            subset = sorted(
                [row for row in structured_table if row["dataset"] == dataset and row["method"] == method],
                key=lambda row: float(row["flop_reduction"]),
            )
            axis.plot(
                [100 * float(row["flop_reduction"]) for row in subset],
                [float(row["accuracy"]) for row in subset],
                marker="o", color=COLOURS[method], label=METHOD_LABELS[method],
            )
        axis.set_title(dataset.replace("fashion", "Fashion-MNIST").replace("cifar10", "CIFAR-10").replace("sign", "Sign"))
        axis.set_xlabel("FLOP reduction (%)")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Test accuracy (%)")
    axes[-1].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(OUTPUT / "figure_structured_efficiency.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for axis, dataset in zip(axes, ("fashion", "cifar10", "sign")):
        subset = [row for row in reliability_table if row["dataset"] == dataset]
        x = np.arange(3)
        for index, row in enumerate(subset):
            values = [
                float(row["calibrated_ece"]),
                1.0 - float(row["ood_msp_auroc"]),
                1.0 - float(row["pgd_retention"]),
            ]
            axis.bar(x + (index - 0.5) * 0.35, values, 0.35, color=COLOURS[row["method"]], label=METHOD_LABELS[row["method"]])
        axis.set_xticks(x, ["ECE", "1−OOD AUC", "1−PGD retention"], rotation=15)
        axis.set_title(dataset.replace("fashion", "Fashion-MNIST").replace("cifar10", "CIFAR-10").replace("sign", "Sign"))
        axis.set_ylabel("Error (lower is better)")
        axis.grid(axis="y", alpha=0.2)
    axes[-1].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(OUTPUT / "figure_reliability.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    learning_curve_figure()
    convergence_effect_table_and_figure()
    parameter_efficiency_figure()
    corruption_heatmap()
    semantic_figure()
    expanded_explanation_tables()
    sign_ten_seed_table()
    first_layer_cross_task_artifacts()
    statistical_correction_artifacts()
    new_extension_tables_and_figures()
    print(f"Wrote thesis tables and figures to {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
