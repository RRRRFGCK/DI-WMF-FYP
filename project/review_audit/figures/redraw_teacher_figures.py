"""Redraw Figures 5.9, 5.16 and 5.21 from frozen CSV records, without overlays.

First preparation (project maintainer):
  python redraw_teacher_figures.py --freeze-root PATH_TO_DOMAIN_MF_V2
Reproduce from the bundled inputs (no model training or PyTorch required):
  python redraw_teacher_figures.py --figure-dir PATH_TO_THESIS_FIGURES_MAIN

The plotted estimates and confidence intervals are read, not recalculated.
The archived input hashes in input_manifest.json identify the source records.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
INPUT = HERE / "inputs"
DATASETS = ("fashion", "cifar10", "sign")
NAMES = {"fashion": "Fashion-MNIST", "cifar10": "CIFAR-10", "sign": "Sign"}
COLOURS = {"kaiming": "#555555", "di_wmf": "#d95f02", "lowrank_wmf": "#b2182b"}
METHODS = {"kaiming": "Kaiming", "di_wmf": "Diagonal DI-WMF", "lowrank_wmf": "Low-rank DI-WMF"}
SOURCE_FILES = {
    "semantic_summary_500.csv": "template_semantics_results/expanded_500/template_semantic_summary.csv",
    "semantic_summary_sign10.csv": "template_semantics_results/sign_full_10seed/template_semantic_summary.csv",
    "semantic_paired_500.csv": "template_semantics_results/expanded_500/template_semantic_paired.csv",
    "semantic_paired_sign10.csv": "template_semantics_results/sign_full_10seed/template_semantic_paired.csv",
    "cost_aggregate.csv": "total_cost_results/cost_aggregate.csv",
    "main_ten_seed.csv": "submission_control_results/final_main_ten_seed_registry.csv",
    "semantic_table.csv": "thesis_artifacts/complete_package/tables/main/T07_template_semantics.csv",
    "corruptions.csv": "thesis_artifacts/complete_package/tables/main/T05_natural_corruptions.csv",
    "regression_aggregate.csv": "correncoder_regression_ablation_results/aggregate.csv",
    "classical_paired.csv": "classical_rr_results/classical_rr_paired.csv",
    "signer_paired.csv": "sign_user_loso_results/sign_user_loso_paired.csv",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    with (INPUT / name).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def freeze(root):
    INPUT.mkdir(parents=True, exist_ok=True)
    records = []
    for destination, source in SOURCE_FILES.items():
        path = root / source
        target = INPUT / destination
        shutil.copy2(path, target)
        records.append({"source": source, "copy": str(target.relative_to(HERE)), "sha256": sha256(path)})
    # The same first test images used by the original generator. These are
    # illustrative dataset icons only; they do not enter any plotted estimate.
    data_root = root.parent / "majorRevision/data"
    fashion_file = data_root / "FashionMNIST/raw/t10k-images-idx3-ubyte"
    with fashion_file.open("rb") as stream:
        stream.read(16)
        fashion = np.frombuffer(stream.read(28 * 28), dtype=np.uint8).reshape(28, 28)
    Image.fromarray(fashion).resize((20, 20), Image.Resampling.BILINEAR).save(INPUT / "icon_fashion.png")
    cifar_file = data_root / "cifar-10-batches-py/test_batch"
    with cifar_file.open("rb") as stream:
        batch = pickle.load(stream, encoding="bytes")
    cifar = batch[b"data"][0].reshape(3, 32, 32).transpose(1, 2, 0)
    Image.fromarray(cifar).save(INPUT / "icon_cifar10.png")
    sign_file = root.parent / "data/sign_language_mnist/test_10/C/0.jpg"
    with Image.open(sign_file) as sign:
        sign.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR).save(INPUT / "icon_sign.png")
    for dataset, source in (("fashion", fashion_file), ("cifar10", cifar_file), ("sign", sign_file)):
        target = INPUT / f"icon_{dataset}.png"
        records.append({"source": str(source), "copy": str(target.relative_to(HERE)), "sha256": sha256(target), "role": "first test-image icon"})
    (HERE / "input_manifest.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")


def verify_inputs():
    manifest = json.loads((HERE / "input_manifest.json").read_text(encoding="utf-8"))
    for entry in manifest:
        actual = sha256(HERE / entry["copy"])
        if actual != entry["sha256"]:
            raise ValueError(f"Frozen input hash differs: {entry['copy']}")


def save(fig, output, stem):
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(output / f"{stem}.{extension}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def semantics(output):
    rows = [r for r in read("semantic_summary_500.csv") if r["dataset"] in {"fashion", "cifar10"}]
    rows += [r for r in read("semantic_summary_sign10.csv") if r["dataset"] == "sign"]
    paired = [r for r in read("semantic_paired_500.csv") if r["dataset"] in {"fashion", "cifar10"}]
    paired += [r for r in read("semantic_paired_sign10.csv") if r["dataset"] == "sign"]
    indexed = {(r["dataset"], r["method"]): r for r in rows}
    contrasts = {(r["dataset"], r["metric"]): r for r in paired}
    metrics = (
        ("mean_mean_semantic_purity", "mean_semantic_purity", "Class-label purity is task-dependent"),
        ("mean_mean_anchor_alignment", "mean_anchor_alignment", "Stored-anchor alignment consistently increases"),
    )
    for dataset in DATASETS:
        expected = 10 if dataset == "sign" else 5
        for _, metric, _ in metrics:
            assert int(contrasts[dataset, metric]["paired_seeds"]) == expected
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 6.7), gridspec_kw={"height_ratios": [1.35, 1]})
    positions = np.arange(3)[::-1]
    for column, (absolute_metric, paired_metric, title) in enumerate(metrics):
        axis = axes[0, column]
        for y, dataset in zip(positions, DATASETS):
            reference = float(indexed[dataset, "kaiming"][absolute_metric])
            candidate = float(indexed[dataset, "lowrank_wmf"][absolute_metric])
            axis.plot([reference, candidate], [y, y], color="#c9c9c9", lw=2.2, zorder=1)
            axis.scatter(reference, y, s=68, color=COLOURS["kaiming"], edgecolor="white", lw=.8, zorder=3)
            axis.scatter(candidate, y, s=78, color=COLOURS["lowrank_wmf"], edgecolor="white", lw=.8, zorder=3)
        axis.set_xlim((.36, .76) if column == 0 else (0, .25))
        axis.set_xlabel("mean score")
        axis.set_title(title, loc="left", fontweight="bold", fontsize=10.5, pad=10)
        effect = axes[1, column]
        for y, dataset in zip(positions, DATASETS):
            row = contrasts[dataset, paired_metric]
            mean, low, high = [float(row[key]) for key in ("mean_candidate_minus_reference", "ci95_low", "ci95_high")]
            colour = COLOURS["lowrank_wmf"] if low > 0 else "#8c8c8c"
            effect.plot([low, high], [y, y], color=colour, lw=2)
            effect.scatter(mean, y, s=58, color=colour, edgecolor="white", lw=.7, zorder=3)
            effect.text(high + .006, y, f"{mean:+.3f}", va="center", fontsize=8, color=colour)
        effect.axvline(0, color="#555555", lw=.9, ls="--")
        effect.set_xlim((-.07, .09) if column == 0 else (-.02, .17))
        effect.set_title("Paired effect", loc="left", fontsize=10, pad=8)
        effect.set_xlabel("paired DI-WMF minus Kaiming (95% CI)")
        for panel in (axis, effect):
            panel.set_yticks(positions, [NAMES[d] for d in DATASETS])
            panel.set_ylim(-.2, 2.2)
            panel.grid(axis="x", alpha=.16)
            panel.spines[["top", "right", "left"]].set_visible(False)
            panel.tick_params(axis="y", length=0)
    fig.suptitle("Template provenance and class-label purity", fontsize=14, fontweight="bold", y=.99)
    fig.text(.5, .945, "Dumbbells show absolute means; lower panels show paired differences with 95% confidence intervals.",
             ha="center", fontsize=9, color="#555555")
    handles = [Line2D([], [], marker="o", linestyle="none", color=COLOURS[m], label=METHODS[m]) for m in ("kaiming", "lowrank_wmf")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .915), ncol=2, frameon=False)
    fig.subplots_adjust(left=.10, right=.98, bottom=.085, top=.79, hspace=.55, wspace=.32)
    save(fig, output, "F08_template_semantics")


def total_cost(output):
    aggregate = read("cost_aggregate.csv")
    assert {int(row["runs"]) for row in aggregate} == {5}
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.5))
    for col, dataset in enumerate(DATASETS):
        for row_index, metric in enumerate(("mean_total_seconds_among_reached", "mean_net_joules_among_reached")):
            axis = axes[row_index, col]
            ci_key = "ci95_total_seconds_among_reached" if row_index == 0 else "ci95_net_joules_among_reached"
            for method in METHODS:
                subset = sorted((r for r in aggregate if r["dataset"] == dataset and r["method"] == method), key=lambda r: float(r["threshold_accuracy"]))
                x = [float(r["threshold_accuracy"]) for r in subset]
                y = [float(r[metric]) for r in subset]
                ci = [float(r[ci_key]) for r in subset]
                axis.errorbar(x, y, yerr=ci, marker="o", capsize=3, color=COLOURS[method], label=METHODS[method])
            axis.grid(alpha=.2)
            if row_index == 0:
                axis.set_title(NAMES[dataset])
            else:
                axis.set_xlabel("Validation accuracy target (%)")
    axes[0, 0].set_ylabel("Time to target (s)")
    axes[1, 0].set_ylabel("Net GPU energy to target (J)")
    axes[0, -1].legend(frameon=False)
    fig.suptitle("Total compute cost to fixed accuracy targets")
    fig.tight_layout()
    save(fig, output, "F11_time_energy_to_accuracy")


def panel(fig, bounds, face, edge):
    x, y, width, height = bounds
    fig.add_artist(patches.FancyBboxPatch((x, y), width, height, transform=fig.transFigure,
                   boxstyle="round,pad=.006,rounding_size=.018", facecolor=face, edgecolor=edge, lw=1, zorder=-1))


def conclusion(output):
    main = {(r["dataset"], r["metric"]): r for r in read("main_ten_seed.csv")}
    assert {int(r["paired_seeds"]) for r in main.values()} == {10}
    semantic = {(r["dataset"], r["metric"]): r for r in read("semantic_table.csv")}
    corruption = read("corruptions.csv")
    regressions = {(r["experiment"], r["metric"]): r for r in read("regression_aggregate.csv")}
    classical = next(r for r in read("classical_paired.csv") if r["comparison"] == "correncoder_improvement_over_bandpass_fft")
    unseen = next(r for r in read("signer_paired.csv") if r["metric"] == "final_test_accuracy")
    titles = (("Complete initializer", "Epoch-0; head-dominated"), ("Post-update learning", "AULC 1:50"),
              ("Final outcome", "final accuracy"), ("Template provenance", "anchor alignment"),
              ("Natural corruptions", "absolute AUC vs retention"))
    fig = plt.figure(figsize=(13.7, 7.25), facecolor="white")
    fig.suptitle("One domain-informed prior, three task-dependent outcomes", y=.978, fontsize=15, fontweight="bold")
    fig.text(.5, .935, "The fitted head explains training-free accuracy; template, final and robustness effects depend on the data regime.",
             ha="center", fontsize=9.3, color="#505050")
    x0, width, gap, row_height = .17, .148, .014, .15
    for i, (heading, subheading) in enumerate(titles):
        center = x0 + i * (width + gap) + width / 2
        fig.text(center, .862, heading, ha="center", fontsize=9.2, fontweight="bold")
        fig.text(center, .824, subheading, ha="center", fontsize=7.7, color="#666666")
    for dataset, y in zip(DATASETS, (.65, .46, .27)):
        ax = fig.add_axes([.035, y + .032, .065, .092])
        icon = np.asarray(Image.open(INPUT / f"icon_{dataset}.png"))
        ax.imshow(icon, cmap="gray" if icon.ndim == 2 else None, vmin=0, vmax=255)
        ax.axis("off")
        name = "Fashion-\nMNIST" if dataset == "fashion" else NAMES[dataset]
        fig.text(.108, y + .085, name, ha="left", fontsize=8.5, fontweight="bold")
        payloads = []
        for metric in ("Epoch-0", "Post-update AULC", "Final test"):
            row = main[dataset, metric]
            mean, low, high = [float(row[k]) for k in ("mean_paired_difference", "ci95_low", "ci95_high")]
            if metric == "Epoch-0":
                payloads.append((f"{mean:+.2f} pp", "fitted head dominates", "control"))
            elif metric == "Final test" and dataset == "sign":
                payloads.append((f"main {mean:+.2f} pp", f"unseen signer: {float(unseen['mean_paired_difference']):+.2f}, inconclusive", "mixed"))
            else:
                payloads.append((f"{mean:+.2f} pp", f"95% CI [{low:+.2f}, {high:+.2f}]", "support" if low > 0 else "mixed"))
        row = semantic[dataset, "mean_anchor_alignment"]
        mean, low, high = [float(row[k]) for k in ("mean_candidate_minus_reference", "ci95_low", "ci95_high")]
        payloads.append((f"{mean:+.3f}", f"95% CI [{low:+.3f}, {high:+.3f}]", "support" if low > 0 else "mixed"))
        if dataset == "fashion":
            payloads.append(("not audited", "full-model corruption family", "boundary"))
        else:
            subset = [r for r in corruption if r["dataset"] == dataset]
            count = sum(float(r["lowrank_minus_kaiming_auc"]) > 0 and float(r["holm_adjusted_p"]) < .05 for r in subset)
            payloads.append((f"absolute AUC: {count}/{len(subset)}", "relative retention: mixed", "mixed"))
        for i, (headline, detail, status) in enumerate(payloads):
            x = x0 + i * (width + gap)
            face, edge, marker = {
                "support": ("#f5d9d5", "#b2182b", "SUPPORTED"),
                "mixed": ("#fff1cf", "#bd7b00", "MIXED"),
                "control": ("#e7f0f6", "#2166ac", "HEAD-DOMINATED"),
                "boundary": ("#efefef", "#9b9b9b", "BOUNDARY"),
            }[status]
            panel(fig, (x, y, width, row_height), face, edge)
            fig.text(x + .012, y + .119, marker, fontsize=6.7, fontweight="bold", color=edge)
            fig.text(x + width / 2, y + .073, headline, ha="center", fontsize=10.3, fontweight="bold", color="#111111")
            fig.text(x + width / 2, y + .031, detail, ha="center", fontsize=7.15, color="#555555")
    rho = float(regressions["pretrained_corr", "waveform_correlation"]["mean"])
    rr = float(regressions["pretrained_mse", "rr_mean_absolute_error_bpm_30p6s"]["mean"])
    panel(fig, (.045, .075, .915, .145), "#eef3f7", "#2166ac")
    fig.text(.065, .175, "CORRENCODER EXTENSION", fontsize=7.2, color="#2166ac", fontweight="bold")
    fig.text(.065, .127, "PPG  \u2192  respiration", fontsize=11, fontweight="bold")
    fig.text(.285, .173, "Waveform correlation\n(Pretrained + correlation)", fontsize=8.2, color="#555555", va="center", linespacing=1.25)
    fig.text(.285, .115, f"{rho:.3f}  (modest morphology recovery)", fontsize=10.2, fontweight="bold", color="#2166ac")
    fig.text(.56, .127, f"{int(classical['correncoder_wins'])}/{int(classical['paired_subjects'])} subjects favour Correncoder", fontsize=9.2, fontweight="bold")
    fig.text(.76, .173, "Respiratory-rate MAE\n(Pretrained + MSE)", fontsize=8.2, color="#555555", va="center", linespacing=1.25)
    fig.text(.76, .115, f"{rr:.2f} vs {float(classical['classical_mean']):.2f} bpm", fontsize=10, fontweight="bold", color="#2166ac")
    fig.text(.76, .087, "Correncoder vs best tested classical", fontsize=7.5, color="#555555")
    save(fig, output, "F20_cross_task_conclusion_map")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-root", type=Path)
    parser.add_argument("--figure-dir", type=Path, default=HERE / "rendered")
    args = parser.parse_args()
    if args.freeze_root:
        freeze(args.freeze_root.resolve())
    verify_inputs()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.titlesize": 10.5,
                         "axes.labelsize": 9.5, "legend.fontsize": 8.5, "figure.titlesize": 12,
                         "axes.spines.top": False, "axes.spines.right": False})
    semantics(args.figure_dir)
    total_cost(args.figure_dir)
    conclusion(args.figure_dir)
    print(f"Redrawn 3 figures as PNG + PDF in {args.figure_dir}")


if __name__ == "__main__":
    main()
