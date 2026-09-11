"""Regenerate the Poster 2 hero figure in a wide, shallow aspect ratio."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = (
    ROOT
    / "output"
    / "pdf"
    / "POSTER2_WIDE_HERO_20260903"
    / "overleaf_package"
    / "Figures"
)

GREEN = "#007A78"
RED = "#B23A48"
GREY = "#7B8794"
LIGHT = "#D9E2E8"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    rows = read_csv(
        ROOT / "fitted_head_training_control_results" / "training_control_rows.csv"
    )
    paired = [
        row
        for row in read_csv(
            ROOT
            / "fitted_head_training_control_results"
            / "training_control_paired.csv"
        )
        if row["contrast"] == "di_conv_effect_with_fitted_head"
    ]

    datasets = [
        ("fashion", "Fashion-MNIST"),
        ("cifar10", "CIFAR-10"),
        ("sign", "Sign"),
    ]
    metrics = [
        ("initial_validation_accuracy", "Epoch-0"),
        ("post_training_aulc", "AULC 1:50"),
        ("final_test_accuracy", "Final"),
    ]

    # The A0 LaTeX slot is roughly 6.2:1.  Matching that ratio lets the chart
    # occupy the full 70%-width hero region without changing its height.
    figure, axes = plt.subplots(1, 3, figsize=(21.0, 3.4), sharey=True)
    for axis, (dataset, dataset_label) in zip(axes, datasets):
        subset = {
            row["metric"]: row for row in paired if row["dataset"] == dataset
        }
        y_positions = np.arange(3)[::-1]
        axis.axvline(0, color=GREY, linewidth=1.3, linestyle="--")

        for y_position, (metric, _) in zip(y_positions, metrics):
            summary = subset[metric]
            mean = float(summary["mean_paired_difference"])
            low = float(summary["ci95_low"])
            high = float(summary["ci95_high"])
            significant = summary["holm_significant_0p05"] == "True"
            colour = GREEN if significant and mean > 0 else (
                RED if significant else GREY
            )

            values_by_condition = {
                condition: {
                    int(row["seed"]): float(row[metric])
                    for row in rows
                    if row["dataset"] == dataset and row["condition"] == condition
                }
                for condition in (
                    "kaiming_conv__fitted_head",
                    "di_conv__fitted_head",
                )
            }
            seeds = sorted(
                set(values_by_condition["kaiming_conv__fitted_head"])
                & set(values_by_condition["di_conv__fitted_head"])
            )
            effects = np.asarray(
                [
                    values_by_condition["di_conv__fitted_head"][seed]
                    - values_by_condition["kaiming_conv__fitted_head"][seed]
                    for seed in seeds
                ]
            )

            axis.scatter(
                effects,
                y_position + np.linspace(-0.11, 0.11, len(effects)),
                s=34,
                color=colour,
                alpha=0.30,
                edgecolors="none",
                zorder=2,
            )
            axis.errorbar(
                mean,
                y_position,
                xerr=[[mean - low], [high - mean]],
                fmt="D",
                color=colour,
                ecolor=colour,
                capsize=5,
                linewidth=2.7,
                markersize=8,
                zorder=4,
            )
            label_offsets = {
                2: (0, -17, "center", "top"),
                1: (0, 13, "center", "bottom"),
                0: (14, 0, "left", "center"),
            }
            x_offset, y_offset, horizontal_alignment, vertical_alignment = (
                label_offsets[int(y_position)]
            )
            axis.annotate(
                f"{mean:+.2f}",
                (mean, y_position),
                xytext=(x_offset, y_offset),
                textcoords="offset points",
                ha=horizontal_alignment,
                va=vertical_alignment,
                fontsize=16.5,
                color=colour,
                fontweight="bold",
            )

        axis.set_title(dataset_label, fontsize=20, fontweight="bold", pad=10)
        axis.set_xlabel(
            "DI conv - random conv (points)\nsame fitted head",
            fontsize=15.5,
        )
        axis.grid(axis="x", color=LIGHT, linewidth=0.9)
        axis.tick_params(axis="both", labelsize=14.5, length=5)
        axis.minorticks_off()
        axis.spines[["top", "right"]].set_visible(False)

    axes[0].set_yticks(
        np.arange(3)[::-1],
        [label for _, label in metrics],
        fontsize=16,
    )
    figure.suptitle(
        "Independent convolution effect - 10 paired seeds",
        y=0.985,
        fontsize=23.5,
        fontweight="bold",
    )
    figure.subplots_adjust(
        left=0.055,
        right=0.995,
        top=0.70,
        bottom=0.27,
        wspace=0.22,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUT / "poster_hero_result.png", dpi=260, facecolor="white")
    figure.savefig(OUT / "poster_hero_result.pdf", facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    main()
