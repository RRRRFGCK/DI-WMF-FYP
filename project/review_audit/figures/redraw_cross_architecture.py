"""Rebuild F03 with low-rank, post-update AULC in every architecture/task cell.

Run:
  python redraw_cross_architecture.py --figure-dir PATH_TO_THESIS_FIGURES_MAIN

The two frozen CSV inputs are in inputs/. Epoch-0 and final statistics use
the original T03 table; post-update AULC uses the corrected audit output.
No original validation_aulc column (Epoch 0 included) is plotted.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
INPUT = HERE / "inputs"


def read(name):
    with (INPUT / name).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure-dir", type=Path, default=HERE / "rendered")
    args = parser.parse_args()
    manifest_path = HERE / "F03_input_manifest.json"
    if manifest_path.exists():
        for entry in json.loads(manifest_path.read_text(encoding="utf-8")):
            digest = hashlib.sha256((INPUT / entry["file"]).read_bytes()).hexdigest()
            if digest != entry["sha256"]:
                raise ValueError(f"Frozen F03 input hash differs: {entry['file']}")
    original = read("cross_architecture_original.csv")
    corrected = read("cross_architecture_post_corrected.csv")
    for row in corrected:
        assert row["method"] == "lowrank_wmf" and row["reference"] == "kaiming"
        assert int(row["paired_seeds"]) == 5
    datasets = [("fashion", "Fashion-MNIST"), ("cifar10", "CIFAR-10"), ("sign", "Sign")]
    architectures = [("resnet18", "ResNet18"), ("correncoder", "Classification Correncoder")]
    metrics = [("initial_val_accuracy", "Epoch-0", (-5, 70)),
               ("post_aulc", "AULC 1:10", (-10, 45)),
               ("test_accuracy", "Final", (-10, 45))]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5,
                         "axes.titlesize": 12, "axes.labelsize": 10.5,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(10.7, 5.7))
    audit_rows = []
    for i, (architecture, architecture_label) in enumerate(architectures):
        for j, (metric, title, limits) in enumerate(metrics):
            ax = axes[i, j]
            ax.axvline(0, color="#7B8794", lw=1, ls="--")
            for y, (dataset, _) in zip(np.arange(3)[::-1], datasets):
                if metric == "post_aulc":
                    source = next(r for r in corrected if r["architecture"] == architecture and r["dataset"] == dataset)
                    mean = float(source["mean_paired_difference"])
                    adjusted = float(source["p_value_holm_across_tasks"])
                else:
                    source = next(r for r in original if r["architecture"] == architecture and r["dataset"] == dataset and r["metric"] == metric)
                    mean = float(source["lowrank_minus_kaiming"])
                    adjusted = float(source["holm_adjusted_p"])
                low, high = float(source["ci95_low"]), float(source["ci95_high"])
                significant = source["holm_significant_0p05"] == "True"
                colour = "#007A78" if significant and mean > 0 else ("#B23A48" if significant else "#7B8794")
                ax.errorbar(mean, y, xerr=[[mean-low], [high-mean]], fmt="o", color=colour,
                            capsize=2.5, lw=1.35, ms=5)
                audit_rows.append({"architecture": architecture, "dataset": dataset, "metric": metric,
                                   "method": "lowrank_wmf", "paired_seeds": 5, "mean": mean,
                                   "ci95_low": low, "ci95_high": high, "p_value_holm": adjusted})
            ax.set_xlim(*limits)
            ax.set_ylim(-.25, 2.25)
            ax.set_yticks([2, 1, 0], [label for _, label in datasets] if j == 0 else [])
            ax.grid(axis="x", color="#D9E2E8", lw=.5)
            ax.minorticks_off()
            if i == 0:
                ax.set_title(title, pad=12)
            if i == 1:
                ax.set_xlabel("Low-rank DI-WMF minus Kaiming\n(percentage points)")
        fig.text(.018, .695 if i == 0 else .335, architecture_label, rotation=90,
                 ha="center", va="center", fontsize=11, fontweight="bold")
    fig.subplots_adjust(left=.195, right=.985, top=.85, bottom=.13, hspace=.40, wspace=.23)
    fig.suptitle("Cross-architecture transfer after separating Epoch-0 from learning", y=.975,
                 fontsize=14, fontweight="bold")
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(args.figure_dir / f"F03_cross_architecture_transfer.{extension}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    with (HERE / "F03_plotted_values.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)
    inputs = [{"file": name, "sha256": hashlib.sha256((INPUT/name).read_bytes()).hexdigest()}
              for name in ("cross_architecture_original.csv", "cross_architecture_post_corrected.csv")]
    (HERE / "F03_input_manifest.json").write_text(json.dumps(inputs, indent=2)+"\n", encoding="utf-8")
    print(f"Corrected F03 PNG and PDF saved in {args.figure_dir}")


if __name__ == "__main__":
    main()
