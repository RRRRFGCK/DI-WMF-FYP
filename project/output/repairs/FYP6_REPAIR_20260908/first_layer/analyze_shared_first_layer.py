"""Finalize repaired first-layer tables, transparent Holm scopes and figures."""
from __future__ import annotations
import argparse
import csv
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import t, ttest_1samp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
METHODS = ["kaiming", "random_stem", "gabor", "pca", "kmeans", "di_wmf", "lowrank_wmf"]
LABELS = ["Kaiming", "Random stem", "Gabor", "PCA", "K-means", "Diagonal DI-WMF", "Low-rank DI-WMF"]
DATASETS = ["fashion", "cifar10", "sign"]
NAMES = {"fashion": "Fashion-MNIST", "cifar10": "CIFAR-10", "sign": "Sign"}


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def holm(rows, field):
    ordered = sorted(range(len(rows)), key=lambda i: rows[i]["raw_p"])
    running = 0.0
    for rank, i in enumerate(ordered):
        running = max(running, min(1.0, rows[i]["raw_p"] * (len(rows) - rank)))
        rows[i][field] = running


def pair(rows, metrics, family, corruption=False):
    lookup = {(r["dataset"], r["method"], int(r["model_seed"]), r.get("corruption", "")): r for r in rows}
    result = []
    for ds in DATASETS:
        expected = set(range(10 if ds == "sign" else 5))
        corruptions = sorted({key[3] for key in lookup if key[0] == ds})
        for cor in corruptions:
            for method in METHODS[1:]:
                seeds = {key[2] for key in lookup if key[:2] == (ds, method) and key[3] == cor}
                assert seeds == expected
                for metric in metrics:
                    a = np.array([float(lookup[(ds, "kaiming", s, cor)][metric]) for s in sorted(seeds)])
                    b = np.array([float(lookup[(ds, method, s, cor)][metric]) for s in sorted(seeds)])
                    delta = b - a
                    std = float(delta.std(ddof=1))
                    mean = float(delta.mean())
                    half = float(t.ppf(.975, len(seeds) - 1)) * std / math.sqrt(len(seeds))
                    p = float(ttest_1samp(delta, 0).pvalue) if std > 0 else (1.0 if mean == 0 else 0.0)
                    result.append({"study": "shared_downstream_repair_20260908", "dataset": ds, "model": "standard",
                                   "corruption": cor, "reference": "kaiming", "method": method, "metric": metric,
                                   "paired_seeds": len(seeds), "reference_mean": float(a.mean()), "method_mean": float(b.mean()),
                                   "mean_paired_difference": mean, "paired_difference_std": std,
                                   "ci95_low": mean-half, "ci95_high": mean+half, "method_greater_count": int((delta > 0).sum()),
                                   "raw_p": p, "family": family,
                                   "correction_scope": f"{family}|dataset={ds}|metric={metric}",
                                   "holm_p_scope": None, "holm_p_family_wide": None, "holm_p_all_378": None,
                                   "included_in_replacement_registry": (not corruption or metric in {"normalised_accuracy_auc", "mean_retention"}),
                                   "analysis_status": "exploratory_repair; inherited retrospective scope, not preregistered"})
    scoped = defaultdict(list)
    for row in result:
        scoped[row["correction_scope"]].append(row)
    for group in scoped.values():
        holm(group, "holm_p_scope")
    return result


def plot_training(aggregate, output):
    table = {(r["dataset"], r["method"]): r for r in aggregate}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 16})
    fig, axes = plt.subplots(3, 2, figsize=(13.6, 10.6))
    palette = ["#5a6673", "#939da6", "#287e55", "#7762a4", "#bf821a", "#e06435", "#b6213b"]
    for row, ds in enumerate(DATASETS):
        for col, metric in enumerate(["validation_aulc", "test_accuracy"]):
            ax = axes[row, col]
            baseline = float(table[(ds, "kaiming")][f"mean_{metric}"])
            ax.axvline(baseline, ls="--", lw=1.1, color="#a8b0b8", zorder=0)
            for y, method in enumerate(METHODS):
                r = table[(ds, method)]
                mean = float(r[f"mean_{metric}"])
                half = float(r[f"ci95_high_{metric}"]) - mean
                ax.errorbar(mean, y, xerr=half, fmt="o", color=palette[y], capsize=4, ms=7)
            ax.set_yticks(range(7), LABELS if col == 0 else [])
            ax.invert_yaxis()
            ax.grid(axis="x", color="#e4e8ed")
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_title(f"{NAMES[ds]} — {5 if ds != 'sign' else 10} paired seeds", fontweight="bold", fontsize=17)
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.set_xlabel("Post-update AULC 1:10 (%)" if col == 0 else "Test accuracy (%)")
    fig.suptitle("First-layer priors with identical raw downstream weights", fontsize=21, fontweight="bold", y=.985)
    fig.text(.5, .94, "10% training data · 10 epochs · mean ± 95% CI · separate logit-scale calibration", ha="center", fontsize=14)
    fig.subplots_adjust(top=.88, bottom=.085, left=.19, right=.985, hspace=.65, wspace=.25)
    fig.savefig(output / "F16_first_layer_prior_ablation_SHARED.png", dpi=260)
    plt.close(fig)


def plot_selectivity(paired, output):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.9), sharey=True)
    for ax, ds in zip(axes, DATASETS):
        ax.axvline(0, color="#8994a0", ls="--", lw=1)
        for y, method in enumerate(METHODS[1:]):
            r = next(r for r in paired if r["dataset"] == ds and r["method"] == method)
            mean = r["mean_paired_difference"]
            ax.errorbar(mean, y, xerr=[[mean-r["ci95_low"]], [r["ci95_high"]-mean]],
                        fmt="o", color="#067e83", capsize=3)
        ax.set_yticks(range(6), LABELS[1:])
        ax.invert_yaxis()
        ax.set_title(NAMES[ds], fontweight="bold")
        ax.set_xlabel("Best-class selectivity\nmethod − Kaiming", fontsize=14)
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#e6e9ed")
    fig.suptitle("Repaired first-layer comparison: test-set response selectivity", fontsize=19, fontweight="bold")
    fig.text(.5, .025, "Paired means and unadjusted 95% CIs; descriptive score, not a semantic or causal certificate", ha="center", fontsize=13)
    fig.subplots_adjust(left=.19, right=.99, top=.81, bottom=.23, wspace=.20)
    fig.savefig(output / "S01_first_layer_selectivity_SHARED.png", dpi=250)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=ROOT / "outputs_first_layer_shared_20260908")
    parser.add_argument("--results-root", type=Path, default=ROOT / "first_layer_shared_results_20260908")
    args = parser.parse_args()
    output = args.results_root
    train_rows = read(args.runs_root / "rows.csv")
    assert len(train_rows) == 140
    aggregate = read(args.runs_root / "aggregate.csv")
    selectivity_rows, corruption_rows, model_metrics = [], [], []
    folders = sorted((output / "per_checkpoint").glob("*/COMPLETE.json"))
    assert len(folders) == 140
    for completed in folders:
        folder = completed.parent
        selectivity_rows.append(json.loads((folder / "selectivity.json").read_text()))
        corruption_rows.extend(read(folder / "corruption_rows.csv"))
        model_metrics.extend(read(folder / "corruption_model_metrics.csv"))
    assert len(corruption_rows) == 140 * 8 * 13 and len(model_metrics) == 140 * 8
    training = pair(train_rows, ["initial_val_accuracy", "validation_aulc", "test_accuracy", "conv1_drift"], "first_layer_training")
    selectivity_pairs = pair(selectivity_rows, ["mean_best_class_selectivity"], "first_layer_selectivity")
    corruption_pairs = pair(model_metrics, ["normalised_accuracy_auc", "mean_accuracy_drop", "mean_retention", "mean_consistency", "worst_accuracy"], "first_layer_corruption", True)
    registry = [r for r in training + selectivity_pairs + corruption_pairs if r["included_in_replacement_registry"]]
    assert len(registry) == 378
    families = defaultdict(list)
    for row in registry:
        families[row["family"]].append(row)
    for group in families.values():
        holm(group, "holm_p_family_wide")
    holm(registry, "holm_p_all_378")
    write(output / "paired_training.csv", training)
    write(output / "paired_selectivity.csv", selectivity_pairs)
    write(output / "paired_corruptions.csv", corruption_pairs)
    write(output / "statistical_registry_replacement_378.csv", registry)
    write(output / "selectivity_combined_rows.csv", selectivity_rows)
    write(output / "corruption_rows.csv", corruption_rows)
    write(output / "corruption_model_metrics.csv", model_metrics)
    write(output / "training_rows.csv", train_rows)
    write(output / "aggregate.csv", aggregate)
    table = []
    for r in aggregate:
        table.append({"dataset": r["dataset"], "method": r["method"], "runs": r["runs"],
                      "epoch0_mean": r["mean_initial_val_accuracy"], "epoch0_std": r["std_initial_val_accuracy"],
                      "aulc_mean": r["mean_validation_aulc"], "aulc_std": r["std_validation_aulc"],
                      "test_mean": r["mean_test_accuracy"], "test_std": r["std_test_accuracy"],
                      "conv1_drift_mean": r["mean_conv1_drift"], "conv1_drift_std": r["std_conv1_drift"], "device": "cuda"})
    write(output / "main_metrics.csv", table)
    corruption_summary = []
    for ds in DATASETS:
        for method in METHODS[1:]:
            for metric in ["normalised_accuracy_auc", "mean_retention"]:
                group = [r for r in corruption_pairs if r["dataset"] == ds and r["method"] == method and r["metric"] == metric]
                corruption_summary.append({"dataset": ds, "method": method, "metric": metric, "corruptions": len(group),
                                           "mean_paired_difference_across_corruptions": float(np.mean([r["mean_paired_difference"] for r in group])),
                                           "scope_holm_positive_corruptions": sum(r["mean_paired_difference"] > 0 and r["holm_p_scope"] < .05 for r in group),
                                           "scope_holm_negative_corruptions": sum(r["mean_paired_difference"] < 0 and r["holm_p_scope"] < .05 for r in group)})
    write(output / "corruption_effect_summary.csv", corruption_summary)
    plot_training(aggregate, output)
    plot_selectivity(selectivity_pairs, output)
    summary = {"training_runs": 140, "evaluated_checkpoints": len(folders), "corruption_observations": len(corruption_rows),
               "selectivity_observations": len(selectivity_rows), "replacement_registry_hypotheses": 378,
               "scope_significant": sum(r["holm_p_scope"] < .05 for r in registry),
               "family_wide_significant": sum(r["holm_p_family_wide"] < .05 for r in registry),
               "all_378_significant": sum(r["holm_p_all_378"] < .05 for r in registry),
               "scope_policy": "Inherited historical retrospective dataset-by-metric scopes; supplementary whole-family and whole-repair sensitivity",
               "seeds": {"fashion": 5, "cifar10": 5, "sign": 10}, "epochs": 10, "train_fraction": .1}
    (output / "final_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
