"""Redraw two label-corrected scientific figures from hash-verified frozen CSVs.

This does not fit models, change any CSV value, regenerate PDF assets or patch
LaTeX. Original frozen generator functions identify the exact rows to plot.
"""
from pathlib import Path
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.stats import t, ttest_1samp

ROOT = Path(__file__).resolve().parents[4]
WORK = Path(__file__).parent
PACKAGE = ROOT / "output/pdf/FYP3_RECORDS_REVISION_20260906/FYP_Code_Records_Audit"
FIGURES = ROOT / "output/repairs/FYP6_REPAIR_20260908/thesis/figures/main"
DATASETS = [("fashion", "Fashion-MNIST"), ("cifar10", "CIFAR-10"), ("sign", "Sign")]
GREEN, RED, GREY = "#007A78", "#B23A48", "#7B8794"
CAPTIONS = {
    "F22_fitted_head_training_control": (
        "Fitted-head pipeline contrast: DI-WMF minus Kaiming when the same head-fitting procedure "
        "is applied separately to each representation, with separately fitted coefficients. Small points "
        "show ten paired-seed effects; diamonds and whiskers show means and unadjusted 95% paired-seed "
        "confidence intervals. Green/red indicate positive/negative effects passing the original Holm "
        "correction across three tasks separately for each outcome of this contrast; grey is inconclusive. "
        "This colouring is not the joint 18-outcome sensitivity analysis discussed in the text."
    ),
    "F09_explanation_faithfulness": (
        "Deletion faithfulness of exact-template-derived explanations relative to random deletion "
        "(full-data, 20-epoch tier: five paired seeds and 500 samples for Fashion-MNIST/CIFAR-10; "
        "ten paired seeds and 140 samples for Sign). Deletion rankings use rectified, min--max-normalised "
        "and bilinearly upsampled CAMs, not the raw signed logit decomposition. Curves show paired mean "
        "differences in target-logit drop and prediction-flip rate with unadjusted 95% paired-seed "
        "confidence intervals. Colours identify initialisation, not significance; deletion creates "
        "inputs outside the training distribution."
    ),
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    manifest_path = PACKAGE / "PACKAGE_MANIFEST.csv"
    package_manifest = {r["path"]: r for r in read(manifest_path)}
    sources = []

    def source(relative):
        path = PACKAGE / relative
        expected = package_manifest[relative]["sha256"]
        assert sha(path) == expected, (relative, "frozen package hash mismatch")
        frozen = WORK / "frozen_sources" / relative
        frozen.parent.mkdir(parents=True, exist_ok=True)
        if frozen.exists():
            assert sha(frozen) == expected, (relative, "repair snapshot changed")
        else:
            shutil.copy2(path, frozen)
        assert sha(frozen) == expected
        sources.append({"original_frozen_path": str(path), "repair_snapshot": str(frozen), "sha256": expected})
        return frozen

    generator22 = source("make_submission_control_figures.py")
    generator09 = source("make_complete_thesis_package.py")
    source("analyze_fitted_head_training_control.py")
    explanation_code = source("evaluate_template_semantics.py")
    explanation_text = explanation_code.read_text(encoding="utf-8")
    assert "exact = F.relu((features * weights[:, :, None, None]).sum(dim=1))" in explanation_text
    assert "normalise_heatmap(exact.detach())" in explanation_text
    assert 'mode="bilinear", align_corners=False' in explanation_text
    assert "def trained_conv_effect():" in generator22.read_text(encoding="utf-8")
    assert "def explanation_faithfulness()" in generator09.read_text(encoding="utf-8")
    rows22 = read(source("fitted_head_training_control_results/training_control_rows.csv"))
    paired22 = [r for r in read(source("fitted_head_training_control_results/training_control_paired.csv"))
                if r["contrast"] == "di_conv_effect_with_fitted_head"]
    combined09 = []
    for relative in ("template_semantics_results/expanded_500/explanation_paired.csv",
                     "template_semantics_results/sign_full_10seed/explanation_paired.csv"):
        combined09.extend(r for r in read(source(relative))
                          if r["reference_explainer"] == "random" and r["candidate_explainer"] == "exact_template_cam"
                          and r["metric"] in {"target_logit_drop", "prediction_flip_rate"}
                          and r["method"] in {"kaiming", "lowrank_wmf"})
    assert len(paired22) == 9 and len(combined09) == 36
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 16,
                         "axes.titlesize": 18, "axes.labelsize": 16,
                         "xtick.labelsize": 14, "ytick.labelsize": 15,
                         "axes.spines.top": False, "axes.spines.right": False})
    outputs = []
    plotted22 = []
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.8), layout="constrained")
    fig.set_layout_engine("constrained", rect=(0, .22, 1, .60), h_pad=.15, w_pad=.16, wspace=.04)
    fig.suptitle("Fitted-head pipeline contrast (10 paired seeds)", fontsize=23, fontweight="bold", y=.985).set_in_layout(False)
    legend = [Line2D([0], [0], color=c, marker="D", lw=2, label=label) for c, label in
              ((GREEN, "Positive, scope-Holm < .05"), (RED, "Negative, scope-Holm < .05"), (GREY, "Inconclusive"))]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.54, .915), ncol=3, frameon=False,
               fontsize=14, handlelength=1.6, columnspacing=1.7).set_in_layout(False)
    metric_order = ["initial_validation_accuracy", "post_training_aulc", "final_test_accuracy"]
    for axis, (dataset, label) in zip(axes, DATASETS):
        axis.axvline(0, color=GREY, lw=1.2, ls="--")
        for yi, metric in zip([2, 1, 0], metric_order):
            row = next(r for r in paired22 if r["dataset"] == dataset and r["metric"] == metric)
            conditions = {condition: {int(r["seed"]): float(r[metric]) for r in rows22
                                     if r["dataset"] == dataset and r["condition"] == condition}
                          for condition in ("kaiming_conv__fitted_head", "di_conv__fitted_head")}
            seeds = sorted(conditions["kaiming_conv__fitted_head"])
            assert len(seeds) == 10 and set(seeds) == set(conditions["di_conv__fitted_head"])
            delta = np.array([conditions["di_conv__fitted_head"][s] - conditions["kaiming_conv__fitted_head"][s] for s in seeds])
            mean, low, high = (float(row[k]) for k in ("mean_paired_difference", "ci95_low", "ci95_high"))
            half = t.ppf(.975, 9) * delta.std(ddof=1) / np.sqrt(10)
            assert abs(delta.mean() - mean) < 1e-10
            assert max(abs(mean - half - low), abs(mean + half - high)) < 1e-8
            assert abs(float(ttest_1samp(delta, 0).pvalue) - float(row["p_value_raw"])) < 1e-10
            significant = row["holm_significant_0p05"] == "True"
            assert significant == (float(row["p_value_holm_across_tasks"]) < .05)
            color = GREEN if significant and mean > 0 else RED if significant else GREY
            axis.scatter(delta, yi + np.linspace(-.11, .11, 10), s=36, color=color, alpha=.36, edgecolors="none", zorder=2)
            axis.errorbar(mean, yi, xerr=[[mean - low], [high - mean]], fmt="D", color=color,
                          capsize=4, lw=2.2, ms=8, zorder=4)
            axis.annotate(f"{mean:+.2f}", (mean, yi), xytext=(0, 15), textcoords="offset points",
                          ha="center", va="bottom", fontsize=15, color=color)
            plotted22.append({**row, "color": color, "paired_seed_ids": seeds, "individual_effects": delta.tolist()})
        axis.set_yticks([2, 1, 0], ["Epoch-0", "AULC 1:50", "Final"] if axis is axes[0] else ["", "", ""])
        axis.set_ylim(-.5, 2.63)
        axis.set_title(label, pad=13, fontweight="bold")
        axis.xaxis.set_major_locator(MaxNLocator(4))
        axis.grid(axis="x", color="#D9E2E8", lw=.8)
    fig.supxlabel("DI-WMF pipeline minus Kaiming pipeline (percentage points)\n"
                  "same head-fitting procedure; coefficients fitted separately", y=.045, fontsize=18).set_in_layout(False)
    name22 = "F22_fitted_head_training_control"
    path22 = FIGURES / f"{name22}.png"
    fig.savefig(path22, dpi=220, facecolor="white")
    plt.close(fig)
    outputs.append({"name": name22, "path": str(path22), "sha256": sha(path22),
                    "original_generator_function": "make_submission_control_figures.trained_conv_effect",
                    "plotted_rows": plotted22, "caption": CAPTIONS[name22],
                    "statistical_scope": "Holm over 3 tasks separately per outcome of the fixed fitted-head pipeline contrast",
                    "independent_raw_seed_mean_ci_p_check": "passed; plotted means and CI endpoints are unchanged CSV values"})

    fig, axes = plt.subplots(2, 3, figsize=(14.4, 8.8), sharex=True, layout="constrained")
    fig.set_layout_engine("constrained", rect=(0, .015, 1, .80), h_pad=.16, w_pad=.13, hspace=.05, wspace=.04)
    fig.suptitle("Deletion faithfulness of exact template explanations", fontsize=23, fontweight="bold", y=.985).set_in_layout(False)
    fig.text(.54, .923, "Deletion rankings use rectified, normalized and upsampled CAMs", ha="center", fontsize=17)
    colors = {"kaiming": "#4D4D4D", "lowrank_wmf": "#B2182B"}
    labels = {"kaiming": "Kaiming", "lowrank_wmf": "Low-rank DI-WMF"}
    legend = [Line2D([0], [0], color=colors[m], marker="o", lw=2.3, label=labels[m]) for m in colors]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.54, .89), ncol=2,
               fontsize=16, frameon=False, columnspacing=2.8).set_in_layout(False)
    for column, (dataset, label) in enumerate(DATASETS):
        for row_index, metric in enumerate(("target_logit_drop", "prediction_flip_rate")):
            axis = axes[row_index, column]
            for method in ("kaiming", "lowrank_wmf"):
                subset = sorted([r for r in combined09 if r["dataset"] == dataset and r["metric"] == metric and r["method"] == method],
                                key=lambda r: float(r["fraction"]))
                assert len(subset) == 3 and [float(r["fraction"]) for r in subset] == [.1, .2, .3]
                assert all(int(r["paired_seeds"]) == (10 if dataset == "sign" else 5) for r in subset)
                x = np.array([100 * float(r["fraction"]) for r in subset])
                mean = np.array([float(r["mean_candidate_minus_reference"]) for r in subset])
                low = np.array([float(r["ci95_low"]) for r in subset])
                high = np.array([float(r["ci95_high"]) for r in subset])
                assert np.all(low <= mean) and np.all(mean <= high)
                axis.errorbar(x, mean, yerr=np.vstack([mean - low, high - mean]), marker="o", ms=6,
                              capsize=4, lw=2.1, color=colors[method], zorder=3)
            axis.axhline(0, color="#555555", lw=1.1, ls="--")
            axis.grid(color="#D9E2E8", lw=.8)
            axis.set_xlim(8, 32)
            axis.set_xticks([10, 20, 30])
            axis.yaxis.set_major_locator(MaxNLocator(5))
            if row_index == 0:
                axis.set_title(label, pad=13, fontweight="bold")
            else:
                axis.set_xlabel("Deleted input pixels (%)", labelpad=10)
    axes[0, 0].set_ylabel("Extra target-logit drop\nvs random deletion", labelpad=12)
    axes[1, 0].set_ylabel("Extra prediction-flip rate\nvs random deletion", labelpad=12)
    name09 = "F09_explanation_faithfulness"
    path09 = FIGURES / f"{name09}.png"
    fig.savefig(path09, dpi=220, facecolor="white")
    plt.close(fig)
    outputs.append({"name": name09, "path": str(path09), "sha256": sha(path09),
                    "original_generator_function": "make_complete_thesis_package.explanation_faithfulness",
                    "plotted_rows": combined09, "caption": CAPTIONS[name09],
                    "statistical_scope": "unadjusted paired-seed 95% CI; method colours are not significance labels",
                    "explanation_map_processing": "weighted channel sum -> ReLU -> per-map min-max normalization -> bilinear upsampling -> top-k pixel deletion",
                    "data_preservation_check": "36 plotted means and CI endpoints read directly from hash-verified frozen CSVs"})
    metadata = {"created_utc": datetime.now(timezone.utc).isoformat(), "renderer": str(Path(__file__)),
                "renderer_sha256": sha(__file__), "package_manifest": str(manifest_path),
                "package_manifest_sha256": sha(manifest_path), "sources": sources, "outputs": outputs,
                "training_performed": False, "pdf_written": False, "old_csvs_modified": False}
    (WORK / "figure_source_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"outputs": [{k: r[k] for k in ("name", "path", "sha256", "caption")} for r in outputs],
                      "source_files_verified": len(sources), "plot_rows": [len(plotted22), len(combined09)]}, indent=2))


if __name__ == "__main__":
    main()
