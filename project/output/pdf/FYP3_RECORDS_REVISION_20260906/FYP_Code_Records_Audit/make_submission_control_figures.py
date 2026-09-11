"""Create thesis figures for the submission-stage control experiments."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "thesis_overleaf_draft" / "figures" / "main"
GREEN = "#007A78"
NAVY = "#173B57"
ORANGE = "#D97706"
GREY = "#7B8794"
LIGHT = "#D9E2E8"
RED = "#B23A48"


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    half = float(student_t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / math.sqrt(len(values)))
    return mean, mean - half, mean + half


def style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, facecolor="white")
    plt.close(fig)


def main_effect_forest():
    rows = read_csv(ROOT / "fitted_head_training_control_results" / "training_control_rows.csv")
    datasets = ["fashion", "cifar10", "sign"]
    labels = ["Fashion-MNIST", "CIFAR-10", "Sign"]
    metrics = [
        ("initial_validation_accuracy", "Epoch-0 accuracy", (-5, 76)),
        ("post_training_aulc", "Post-update AULC", (-5, 50)),
        ("final_test_accuracy", "Final test accuracy", (-5, 30)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.45))
    for ax, (metric, title, limits) in zip(axes, metrics):
        effects = []
        for dataset in datasets:
            by_condition = {}
            for condition in ("kaiming_conv__kaiming_head", "di_conv__fitted_head"):
                by_condition[condition] = {
                    int(r["seed"]): float(r[metric])
                    for r in rows
                    if r["dataset"] == dataset and r["condition"] == condition
                }
            seeds = sorted(set(by_condition["kaiming_conv__kaiming_head"]) & set(by_condition["di_conv__fitted_head"]))
            delta = [
                by_condition["di_conv__fitted_head"][s] - by_condition["kaiming_conv__kaiming_head"][s]
                for s in seeds
            ]
            effects.append(mean_ci(delta))
        y = np.arange(3)[::-1]
        ax.axvline(0, color=GREY, lw=1, ls="--")
        for yi, (mean, low, high) in zip(y, effects):
            colour = GREEN if low > 0 else (RED if high < 0 else GREY)
            ax.errorbar(mean, yi, xerr=[[mean - low], [high - mean]], fmt="o", color=colour,
                        ecolor=colour, capsize=3, markersize=5, lw=1.5)
            offset = -12 if yi > 0 else 10
            va = "top" if yi > 0 else "bottom"
            ax.annotate(f"{mean:+.2f}", (mean, yi), xytext=(0, offset), textcoords="offset points",
                        ha="center", va=va, fontsize=7.5, color=colour)
        ax.set_yticks(y, labels if ax is axes[0] else ["", "", ""])
        ax.set_xlim(*limits)
        ax.set_title(title)
        ax.set_xlabel("DI-WMF minus Kaiming\n(points)")
        ax.grid(axis="x", color=LIGHT, lw=0.6)
    fig.subplots_adjust(top=0.75, bottom=0.24, wspace=0.23)
    fig.suptitle("Complete initializer: paired ten-seed effects", y=0.98, fontsize=11, fontweight="bold")
    save(fig, "F01_cross_task_paired_effects.png")


def epoch0_factorial():
    aggregate = read_csv(ROOT / "epoch0_factorial_results" / "epoch0_factorial_aggregate.csv")
    datasets = [("fashion", "Fashion-MNIST"), ("cifar10", "CIFAR-10"), ("sign", "Sign")]
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 3.25), sharey=True)
    for ax, (dataset, label) in zip(axes, datasets):
        values = {
            r["condition"]: float(r["mean"])
            for r in aggregate
            if r["dataset"] == dataset and r["metric"] == "validation_accuracy"
        }
        x = np.array([0, 1])
        k = [values["kaiming_conv__kaiming_head"], values["kaiming_conv__fitted_head"]]
        d = [values["di_conv__kaiming_head"], values["di_conv__fitted_head"]]
        ax.plot(x, k, "o-", color=NAVY, lw=2.6, ms=7, label="Kaiming conv")
        ax.plot(x, d, "o-", color=GREEN, lw=2.6, ms=7, label="DI-WMF conv")
        for xs, ys, colour, horizontal_offset, label_offset, vertical_alignment in (
            (x, k, NAVY, -16, 10, "bottom"),
            (x, d, GREEN, 16, 10, "bottom"),
        ):
            for xx, yy in zip(xs, ys):
                ax.annotate(
                    f"{yy:.1f}",
                    (xx, yy),
                    xytext=(horizontal_offset, label_offset),
                    textcoords="offset points",
                    ha="center",
                    va=vertical_alignment,
                    fontsize=13.0,
                    color=colour,
                    zorder=5,
                    bbox=dict(
                        boxstyle="round,pad=0.08",
                        facecolor="white",
                        edgecolor="none",
                        alpha=0.86,
                    ),
                )
        ax.set_xticks(x, ["Random\nhead", "Fitted\nhead"], fontsize=13.0)
        ax.set_ylim(0, 94)
        ax.set_title(label, fontsize=15.5, y=1.04, pad=0)
        ax.grid(axis="y", color=LIGHT, lw=0.6)
        ax.tick_params(axis="y", labelsize=12.5)
        ax.minorticks_off()
    axes[0].set_ylabel("Epoch-0 accuracy (%)", fontsize=14.0)
    handles, labels = axes[-1].get_legend_handles_labels()
    # Keep the legend away from the two-line x-axis labels.  Placing it in the
    # reserved band below the main title also makes the three panels easier to
    # scan at poster distance.
    fig.legend(handles, labels, frameon=False, fontsize=13.5, loc="upper center",
               bbox_to_anchor=(0.5, 0.855), ncol=2, columnspacing=1.8,
               handlelength=1.8)
    fig.subplots_adjust(top=0.61, bottom=0.20, wspace=0.34)
    fig.suptitle("The fitted head, not the convolution bank, dominates Epoch-0", y=0.98,
                 fontsize=17.0, fontweight="bold")
    save(fig, "F21_epoch0_factorial.png")


def trained_conv_effect():
    rows = read_csv(
        ROOT / "fitted_head_training_control_results" / "training_control_rows.csv"
    )
    paired = [
        r for r in read_csv(ROOT / "fitted_head_training_control_results" / "training_control_paired.csv")
        if r["contrast"] == "di_conv_effect_with_fitted_head"
    ]
    datasets = [("fashion", "Fashion-MNIST"), ("cifar10", "CIFAR-10"), ("sign", "Sign")]
    metric_order = ["initial_validation_accuracy", "post_training_aulc", "final_test_accuracy"]
    metric_labels = ["Epoch-0", "AULC 1:50", "Final"]
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.55), sharey=False)
    for ax, (dataset, label) in zip(axes, datasets):
        subset = {r["metric"]: r for r in paired if r["dataset"] == dataset}
        y = np.arange(3)[::-1]
        ax.axvline(0, color=GREY, lw=1, ls="--")
        for yi, metric in zip(y, metric_order):
            row = subset[metric]
            mean = float(row["mean_paired_difference"])
            low = float(row["ci95_low"])
            high = float(row["ci95_high"])
            colour = GREEN if row["holm_significant_0p05"] == "True" and mean > 0 else (
                RED if row["holm_significant_0p05"] == "True" else GREY
            )
            by_condition = {
                condition: {
                    int(item["seed"]): float(item[metric])
                    for item in rows
                    if item["dataset"] == dataset and item["condition"] == condition
                }
                for condition in ("kaiming_conv__fitted_head", "di_conv__fitted_head")
            }
            seeds = sorted(
                set(by_condition["kaiming_conv__fitted_head"])
                & set(by_condition["di_conv__fitted_head"])
            )
            seed_effects = np.array(
                [
                    by_condition["di_conv__fitted_head"][seed]
                    - by_condition["kaiming_conv__fitted_head"][seed]
                    for seed in seeds
                ]
            )
            offsets = np.linspace(-0.11, 0.11, len(seed_effects))
            ax.scatter(
                seed_effects,
                yi + offsets,
                s=13,
                color=colour,
                alpha=0.35,
                edgecolors="none",
                zorder=2,
            )
            ax.errorbar(
                mean,
                yi,
                xerr=[[mean - low], [high - mean]],
                fmt="D",
                color=colour,
                ecolor=colour,
                capsize=3,
                lw=1.6,
                ms=5,
                zorder=4,
            )
            offset = -12 if yi > 0 else 10
            va = "top" if yi > 0 else "bottom"
            ax.annotate(f"{mean:+.2f}", (mean, yi), xytext=(0, offset), textcoords="offset points",
                        ha="center", va=va, fontsize=7.5, color=colour)
        ax.set_yticks(y)
        ax.set_yticklabels(metric_labels if ax is axes[0] else [])
        ax.set_title(label)
        ax.set_xlabel("DI conv minus random conv\n(same fitted head; points)")
        ax.grid(axis="x", color=LIGHT, lw=0.6)
    fig.subplots_adjust(top=0.74, bottom=0.26, wspace=0.30)
    fig.suptitle("Independent contribution of DI-WMF convolutional templates (10 paired seeds)", y=0.98,
                 fontsize=11, fontweight="bold")
    save(fig, "F22_fitted_head_training_control.png")


def trained_conv_effect_poster_compact():
    """Large-format post-update result for the poster hero band.

    Epoch 0 is deliberately excluded here.  It is explained by the separate
    factorial control in Section 3 of the poster; the hero should communicate
    the independent *post-update* convolution effect at viewing distance.
    """
    rows = read_csv(
        ROOT / "fitted_head_training_control_results" / "training_control_rows.csv"
    )
    paired = [
        r for r in read_csv(ROOT / "fitted_head_training_control_results" / "training_control_paired.csv")
        if r["contrast"] == "di_conv_effect_with_fitted_head"
    ]
    datasets = [("fashion", "Fashion-MNIST"), ("cifar10", "CIFAR-10"), ("sign", "Sign")]
    panel_limits = {
        "fashion": (-1.0, 3.0),
        "cifar10": (-1.0, 2.0),
        "sign": (-4.0, 21.0),
    }
    metric_order = ["post_training_aulc", "final_test_accuracy"]
    metric_labels = ["Learning\n(AULC 1:50)", "Final\naccuracy"]
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.55), sharey=True)
    for ax, (dataset, label) in zip(axes, datasets):
        subset = {r["metric"]: r for r in paired if r["dataset"] == dataset}
        y = np.arange(2)[::-1]
        low_limit, high_limit = panel_limits[dataset]
        ax.axvspan(0, high_limit, color="#EAF6EF", alpha=0.55, zorder=0)
        ax.axvline(0, color=GREY, lw=1.2, ls="--")
        for yi, metric in zip(y, metric_order):
            row = subset[metric]
            mean = float(row["mean_paired_difference"])
            low = float(row["ci95_low"])
            high = float(row["ci95_high"])
            colour = GREEN if row["holm_significant_0p05"] == "True" and mean > 0 else (
                RED if row["holm_significant_0p05"] == "True" else GREY
            )
            by_condition = {
                condition: {
                    int(item["seed"]): float(item[metric])
                    for item in rows
                    if item["dataset"] == dataset and item["condition"] == condition
                }
                for condition in ("kaiming_conv__fitted_head", "di_conv__fitted_head")
            }
            seeds = sorted(set(by_condition["kaiming_conv__fitted_head"]) & set(by_condition["di_conv__fitted_head"]))
            effects = np.asarray([
                by_condition["di_conv__fitted_head"][seed]
                - by_condition["kaiming_conv__fitted_head"][seed]
                for seed in seeds
            ])
            ax.scatter(effects, yi + np.linspace(-0.11, 0.11, len(effects)), s=36,
                       color=colour, alpha=0.30, edgecolors="none", zorder=2)
            ax.errorbar(mean, yi, xerr=[[mean - low], [high - mean]], fmt="D",
                        color=colour, ecolor=colour, capsize=5.0, lw=2.8, ms=8.5, zorder=4)
            # Keep the upper estimate below its marker and the lower estimate
            # above its marker so neither label collides with panel headings.
            vertical_offset = -19 if int(yi) == 1 else 18
            ax.annotate(f"{mean:+.2f}", (mean, yi), xytext=(0, vertical_offset),
                        textcoords="offset points", ha="center",
                        va="top" if vertical_offset < 0 else "bottom", fontsize=17.0, color=colour,
                        fontweight="bold")
        supported = dataset in {"fashion", "sign"}
        status = "SUPPORTED" if supported else "SMALL / MIXED"
        ax.set_title(
            f"{label}\n{status}",
            fontsize=20.0,
            fontweight="bold",
            color=GREEN if supported else GREY,
            pad=12,
        )
        ax.set_xlim(low_limit, high_limit)
        ax.set_xlabel("DI-WMF - random conv (points)", fontsize=16.5)
        ax.grid(axis="x", color=LIGHT, lw=0.9)
        ax.tick_params(axis="both", which="major", labelsize=15.0, length=5)
        ax.minorticks_off()
    axes[0].set_yticks(np.arange(2)[::-1], metric_labels, fontsize=17.0)
    fig.suptitle("After training starts, the independent DI-WMF benefit is task-dependent",
                 y=0.99, fontsize=25.0, fontweight="bold")
    fig.text(
        0.5,
        0.885,
        "Same fitted classifier; mean paired effect and 95% CI over 10 seeds",
        ha="center",
        va="top",
        fontsize=17.0,
        color="#4F4F4F",
    )
    fig.subplots_adjust(left=0.105, right=0.995, top=0.66, bottom=0.22, wspace=0.24)
    poster_out = ROOT / "poster_imperial" / "Figures"
    poster_out.mkdir(parents=True, exist_ok=True)
    fig.savefig(poster_out / "poster_hero_result.png", dpi=260, facecolor="white")
    fig.savefig(poster_out / "poster_hero_result.pdf", facecolor="white")
    plt.close(fig)


def sign_user_loso():
    rows = read_csv(ROOT / "sign_user_loso_results" / "sign_user_loso_rows.csv")
    paired = read_csv(ROOT / "sign_user_loso_results" / "sign_user_loso_paired.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.45), gridspec_kw={"width_ratios": [1.18, 1]})
    ax = axes[0]
    users = sorted({int(r["test_user"]) for r in rows})
    di_groups = {}
    for user in users:
        k = float(next(r["final_test_accuracy"] for r in rows if int(r["test_user"]) == user and r["method"] == "kaiming"))
        d = float(next(r["final_test_accuracy"] for r in rows if int(r["test_user"]) == user and r["method"] == "lowrank_wmf"))
        colour = GREEN if d > k else RED
        ax.plot([0, 1], [k, d], color=colour, alpha=0.72, lw=1.5)
        ax.scatter([0, 1], [k, d], color=colour, s=20, zorder=3)
        di_groups.setdefault((d, colour), []).append(user)
    for (value, colour), grouped_users in sorted(di_groups.items()):
        label = ", ".join(f"U{user}" for user in grouped_users)
        ax.text(1.05, value, label, va="center", fontsize=12.0, color=colour)
    ax.set_xticks([0, 1], ["Kaiming", "DI-WMF"], fontsize=12.5)
    ax.set_ylabel("Final accuracy on unseen signer (%)", fontsize=13.0)
    ax.set_xlim(-0.18, 1.3)
    ax.set_ylim(20, 62)
    ax.set_title("A  Held-out signers", loc="left", fontweight="bold", fontsize=14.5)
    ax.grid(axis="y", color=LIGHT, lw=0.6)
    ax.tick_params(axis="y", labelsize=12.0)
    ax.minorticks_off()

    ax = axes[1]
    order = ["initial_test_accuracy", "post_training_aulc", "final_test_accuracy"]
    labels = ["Epoch-0", "AULC 1:50", "Final"]
    subset = {r["metric"]: r for r in paired}
    y = np.arange(3)[::-1]
    ax.axvline(0, color=GREY, lw=1, ls="--")
    for yi, metric in zip(y, order):
        row = subset[metric]
        mean = float(row["mean_paired_difference"])
        low, high = float(row["ci95_low"]), float(row["ci95_high"])
        colour = GREEN if low > 0 else GREY
        ax.errorbar(mean, yi, xerr=[[mean - low], [high - mean]], fmt="o", color=colour,
                    ecolor=colour, capsize=3, lw=1.5, ms=5)
        offset = -12 if yi > 0 else 10
        va = "top" if yi > 0 else "bottom"
        ax.annotate(f"{mean:+.2f}", (mean, yi), xytext=(0, offset), textcoords="offset points",
                    ha="center", va=va, fontsize=12.0, color=colour)
    ax.set_yticks(y, labels, fontsize=12.5)
    ax.set_xlabel("DI-WMF minus Kaiming (points)", fontsize=13.0)
    ax.set_title("B  Paired effects", loc="left", fontweight="bold", fontsize=14.5)
    ax.grid(axis="x", color=LIGHT, lw=0.6)
    ax.tick_params(axis="x", labelsize=12.0)
    ax.minorticks_off()
    fig.subplots_adjust(wspace=0.38)
    save(fig, "F23_sign_user_loso.png")


def classical_rr():
    classical_rows = read_csv(ROOT / "classical_rr_results" / "classical_rr_subject_rows.csv")
    corr_rows = read_csv(ROOT / "outputs_correncoder_regression_full" / "fold_metrics.csv")
    metric = "rr_mean_absolute_error_bpm_30p6s"
    corr = {int(r["fold"]): float(r[metric]) for r in corr_rows}
    methods = ["bandpass_fft", "envelope_fft", "bandpass_autocorrelation"]
    method_labels = ["Band-pass + FFT", "Envelope + FFT", "Band-pass + ACF"]
    by_method = {
        method: {int(r["fold"]): float(r[metric]) for r in classical_rows if r["method"] == method}
        for method in methods
    }
    common = sorted(set(corr) & set(by_method["bandpass_fft"]))
    x = np.array([by_method["bandpass_fft"][fold] for fold in common])
    y = np.array([corr[fold] for fold in common])
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.4))
    ax = axes[0]
    ax.scatter(x, y, s=40, color=GREEN, alpha=0.72, edgecolor="white", linewidth=0.5)
    limit = max(float(x.max()), float(y.max())) * 1.04
    ax.plot([0, limit], [0, limit], ls="--", color=GREY, lw=1.7)
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_xlabel("Band-pass + FFT MAE (bpm)", fontsize=14.0)
    ax.set_ylabel("Correncoder MAE (bpm)", fontsize=14.0)
    ax.set_title("A  Subject-level comparison", loc="left", fontweight="bold", fontsize=15.5)
    ax.text(0.98, 0.06, "Correncoder wins 45/53", transform=ax.transAxes,
            ha="right", color=GREEN, fontweight="bold", fontsize=13.5)
    ax.grid(color=LIGHT, lw=0.5)
    ax.tick_params(labelsize=12.5)
    ax.minorticks_off()

    ax = axes[1]
    series = [("Correncoder", np.array(list(corr.values())), GREEN)]
    series += [(label, np.array(list(by_method[method].values())), colour)
               for method, label, colour in zip(methods, method_labels, [NAVY, ORANGE, GREY])]
    for label, values, colour in series:
        sorted_values = np.sort(values)
        probability = np.arange(1, len(values) + 1) / len(values)
        ax.step(sorted_values, probability, where="post", label=f"{label} ({values.mean():.2f})", color=colour, lw=2.3)
    ax.set_xlabel("Subject mean RR MAE (bpm)", fontsize=14.0)
    ax.set_ylabel("Cumulative fraction of subjects", fontsize=14.0)
    ax.set_title("B  Error distribution", loc="left", fontweight="bold", fontsize=15.5)
    ax.legend(frameon=False, fontsize=11.5, title="Method (mean)", title_fontsize=11.5)
    ax.grid(color=LIGHT, lw=0.5)
    ax.tick_params(labelsize=12.5)
    ax.minorticks_off()
    fig.subplots_adjust(wspace=0.32)
    save(fig, "F24_classical_rr_baselines.png")


def classical_rr_poster():
    """Single large-format physiology panel designed for one A0 poster column."""
    classical_rows = read_csv(ROOT / "classical_rr_results" / "classical_rr_subject_rows.csv")
    corr_rows = read_csv(ROOT / "outputs_correncoder_regression_full" / "fold_metrics.csv")
    metric = "rr_mean_absolute_error_bpm_30p6s"
    corr = {int(r["fold"]): float(r[metric]) for r in corr_rows}
    methods = ["bandpass_fft", "envelope_fft", "bandpass_autocorrelation"]
    method_labels = ["Band-pass + FFT", "Envelope + FFT", "Band-pass + ACF"]
    by_method = {
        method: {int(r["fold"]): float(r[metric]) for r in classical_rows if r["method"] == method}
        for method in methods
    }
    fig, ax = plt.subplots(1, 1, figsize=(7.4, 4.5))
    series = [("Correncoder", np.array(list(corr.values())), GREEN)]
    series += [(label, np.array(list(by_method[method].values())), colour)
               for method, label, colour in zip(methods, method_labels, [NAVY, ORANGE, GREY])]
    for label, values, colour in series:
        sorted_values = np.sort(values)
        probability = np.arange(1, len(values) + 1) / len(values)
        ax.step(sorted_values, probability, where="post",
                label=f"{label}: {values.mean():.2f} bpm", color=colour, lw=3.2)
    ax.set_xlabel("Subject mean respiratory-rate MAE (bpm)", fontsize=18)
    ax.set_ylabel("Cumulative fraction of subjects", fontsize=18)
    ax.set_title("Error distribution on the same 53 held-out subjects",
                 loc="left", fontsize=20, fontweight="bold")
    ax.legend(frameon=False, fontsize=15.5, title="Method: mean error",
              title_fontsize=15.5,
              loc="lower right")
    ax.grid(color=LIGHT, lw=0.9)
    ax.tick_params(labelsize=16)
    ax.minorticks_off()
    fig.subplots_adjust(left=0.15, right=0.98, top=0.88, bottom=0.18)

    poster_out = ROOT / "poster_imperial" / "Figures"
    poster_out.mkdir(parents=True, exist_ok=True)
    fig.savefig(poster_out / "poster_classical_rr_baselines.png", dpi=260, facecolor="white")
    fig.savefig(poster_out / "poster_classical_rr_baselines.pdf", facecolor="white")
    plt.close(fig)


def cross_architecture_post_aulc():
    original = read_csv(ROOT / "thesis_artifacts" / "complete_package" / "tables" / "main" / "T03_cross_architecture_holm.csv")
    post = read_csv(ROOT / "submission_control_results" / "cross_architecture_post_aulc.csv")
    datasets = [("fashion", "Fashion-MNIST"), ("cifar10", "CIFAR-10"), ("sign", "Sign")]
    architectures = [("resnet18", "ResNet18"), ("correncoder", "Classification Correncoder")]
    metrics = [
        ("initial_val_accuracy", "Epoch-0", (-5, 70)),
        ("post_aulc", "AULC 1:10", (-10, 45)),
        ("test_accuracy", "Final", (-10, 45)),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(7.25, 4.0))
    for row_index, (architecture, architecture_label) in enumerate(architectures):
        for column_index, (metric, title, limits) in enumerate(metrics):
            ax = axes[row_index, column_index]
            y = np.arange(3)[::-1]
            ax.axvline(0, color=GREY, lw=1, ls="--")
            for yi, (dataset, _) in zip(y, datasets):
                if metric == "post_aulc":
                    source = next(r for r in post if r["architecture"] == architecture and r["dataset"] == dataset)
                    mean = float(source["mean_paired_difference"])
                    low, high = float(source["ci95_low"]), float(source["ci95_high"])
                    significant = source["holm_significant_0p05"] == "True"
                else:
                    source = next(r for r in original if r["architecture"] == architecture and r["dataset"] == dataset and r["metric"] == metric)
                    mean = float(source["lowrank_minus_kaiming"])
                    low, high = float(source["ci95_low"]), float(source["ci95_high"])
                    significant = source["holm_significant_0p05"] == "True"
                colour = GREEN if significant and mean > 0 else (RED if significant else GREY)
                ax.errorbar(mean, yi, xerr=[[mean - low], [high - mean]], fmt="o", color=colour,
                            ecolor=colour, capsize=2.5, lw=1.35, ms=4.5)
            ax.set_xlim(*limits)
            ax.set_yticks(y)
            ax.set_yticklabels([label for _, label in datasets] if column_index == 0 else [])
            if row_index == 0:
                ax.set_title(title)
            if row_index == 1:
                ax.set_xlabel("DI-WMF minus Kaiming\n(points)")
            ax.grid(axis="x", color=LIGHT, lw=0.5)
            if column_index == 0:
                ax.text(-0.48, 0.5, architecture_label, transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=9, fontweight="bold")
    fig.subplots_adjust(left=0.20, top=0.86, bottom=0.18, hspace=0.42, wspace=0.25)
    fig.suptitle("Cross-architecture transfer after separating Epoch-0 from learning", y=0.98,
                 fontsize=11, fontweight="bold")
    save(fig, "F03_cross_architecture_transfer.png")


def main():
    style()
    main_effect_forest()
    epoch0_factorial()
    trained_conv_effect()
    trained_conv_effect_poster_compact()
    sign_user_loso()
    classical_rr()
    classical_rr_poster()
    cross_architecture_post_aulc()
    print(f"Wrote submission-control figures to {OUT}")


if __name__ == "__main__":
    main()
