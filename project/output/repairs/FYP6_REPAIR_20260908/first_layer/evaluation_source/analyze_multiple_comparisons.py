"""Apply family-wise Holm correction and compute paired Cohen's dz.

The project stores paired means and either the paired standard deviation or a
95% paired-t interval.  This script reconstructs the two-sided paired t test,
then applies Holm--Bonferroni within explicitly encoded experimental scopes.
The mapping is reproducible, but it was assembled after result generation and
is therefore a retrospective multiplicity audit, not a preregistration or a
confirmatory analysis plan.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

from scipy.stats import t as student_t


@dataclass(frozen=True)
class Study:
    family: str
    path: str
    metrics: tuple[str, ...] | None = None


STUDIES = (
    Study("main_cross_task", "convergence_results/paired.csv"),
    Study("first_layer_training", "first_layer_cross_task_results/paired_training.csv"),
    Study("first_layer_selectivity", "first_layer_cross_task_results/paired_selectivity.csv"),
    Study(
        "first_layer_corruption",
        "first_layer_cross_task_results/paired_corruptions.csv",
        ("normalised_accuracy_auc", "mean_retention"),
    ),
    Study("cross_architecture", "resnet18_paired_comparisons.csv"),
    Study("cross_architecture", "correncoder_paired_comparisons.csv"),
    Study(
        "natural_corruption",
        "corruption_results/full20_cross_task_paired.csv",
        ("normalised_accuracy_auc", "mean_retention"),
    ),
    Study("interpretability_semantics", "template_semantics_results/expanded_500/template_semantic_paired.csv"),
    Study("interpretability_semantics", "template_semantics_results/sign_full_10seed/template_semantic_paired.csv"),
    Study("interpretability_faithfulness", "template_semantics_results/expanded_500/explanation_paired.csv"),
    Study("interpretability_faithfulness", "template_semantics_results/sign_full_10seed/explanation_paired.csv"),
    Study("reliability_calibration", "reliability_results/full20_standard/calibration_paired.csv"),
    Study("reliability_ood", "reliability_results/full20_standard/ood_paired.csv"),
    Study("reliability_adversarial", "reliability_results/full20_standard/adversarial_paired.csv"),
    Study("parameter_efficiency", "efficiency_results/paired.csv"),
    Study("correncoder_initialisation_objective", "correncoder_regression_ablation_results/paired_comparisons.csv"),
    Study("correncoder_layerwise_lag", "correncoder_extension_results/paired_comparisons.csv"),
    Study("correncoder_depth_covariance", "correncoder_completed_extension_results/paired_comparisons.csv"),
)


N_COLUMNS = ("paired_seeds", "n", "n_paired_subjects")
DIFF_COLUMNS = (
    "mean_paired_difference",
    "mean_candidate_minus_reference",
    "mean_lowrank_minus_kaiming",
    "oriented_improvement",
    "mean_lowrank_minus_kaiming_retention",
    "method_minus_reference",
)
STD_COLUMNS = ("paired_difference_std",)
HALF_WIDTH_COLUMNS = ("improvement_ci95_half_width", "delta_ci95_half_width")
SCOPE_FIELDS = {
    "main_cross_task": ("metric",),
    "first_layer_training": ("dataset", "metric"),
    "first_layer_selectivity": ("dataset", "metric"),
    "first_layer_corruption": ("dataset", "metric"),
    "cross_architecture": ("model", "metric"),
    "natural_corruption": ("dataset", "metric"),
    "interpretability_semantics": ("dataset", "metric"),
    "interpretability_faithfulness": ("dataset", "metric"),
    "reliability_calibration": ("dataset", "metric"),
    "reliability_ood": ("dataset", "metric"),
    "reliability_adversarial": ("dataset", "metric"),
    "parameter_efficiency": ("dataset", "metric"),
    "correncoder_initialisation_objective": ("metric",),
    "correncoder_layerwise_lag": ("metric",),
    "correncoder_depth_covariance": ("metric",),
}

# This is a submission-stage description of how each family is used in the
# dissertation.  It is not, and must never be interpreted as, an experiment-
# time registration or planned/exploratory label from before results existed.
SUBMISSION_STAGE_ROLE = {
    "main_cross_task": "central_descriptive_result",
    "first_layer_training": "controlled_secondary_analysis",
    "first_layer_selectivity": "controlled_secondary_analysis",
    "first_layer_corruption": "exploratory_stress_test",
    "cross_architecture": "controlled_secondary_analysis",
    "natural_corruption": "controlled_secondary_analysis",
    "interpretability_semantics": "controlled_secondary_analysis",
    "interpretability_faithfulness": "controlled_secondary_analysis",
    "reliability_calibration": "controlled_secondary_analysis",
    "reliability_ood": "controlled_secondary_analysis",
    "reliability_adversarial": "controlled_secondary_analysis",
    "parameter_efficiency": "controlled_secondary_analysis",
    "correncoder_initialisation_objective": "exploratory_extension",
    "correncoder_layerwise_lag": "exploratory_extension",
    "correncoder_depth_covariance": "exploratory_extension",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def first_value(row: dict[str, str], names: tuple[str, ...]) -> tuple[str, float]:
    for name in names:
        value = row.get(name, "")
        if value not in ("", None):
            return name, float(value)
    raise ValueError(f"Missing one of {names}: {row}")


def holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [1.0] * len(p_values)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def magnitude(value: float) -> str:
    absolute = abs(value)
    if not math.isfinite(absolute):
        return "infinite"
    if absolute < 0.2:
        return "negligible"
    if absolute < 0.5:
        return "small"
    if absolute < 0.8:
        return "medium"
    return "large"


def standardise(root: Path, study: Study) -> list[dict]:
    path = root / study.path
    if not path.exists():
        raise FileNotFoundError(path)
    output: list[dict] = []
    for source_index, row in enumerate(read_csv(path), start=1):
        if study.metrics is not None and row.get("metric") not in study.metrics:
            continue
        _, n_float = first_value(row, N_COLUMNS)
        n = int(n_float)
        if n < 2:
            raise ValueError(f"Paired inference requires n>=2: {path} row {source_index}")
        diff_column, difference = first_value(row, DIFF_COLUMNS)
        t_critical = float(student_t.ppf(0.975, n - 1))

        if row.get("ci95_low", "") not in ("", None) and row.get("ci95_high", "") not in ("", None):
            ci_low = float(row["ci95_low"])
            ci_high = float(row["ci95_high"])
            half_width = (ci_high - ci_low) / 2.0
        else:
            _, half_width = first_value(row, HALF_WIDTH_COLUMNS)
            ci_low = difference - half_width
            ci_high = difference + half_width

        paired_sd = None
        for std_column in STD_COLUMNS:
            if row.get(std_column, "") not in ("", None):
                paired_sd = float(row[std_column])
                break
        if paired_sd is None:
            paired_sd = half_width * math.sqrt(n) / t_critical

        if paired_sd == 0.0:
            if difference == 0.0:
                t_statistic, raw_p, effect = 0.0, 1.0, 0.0
            else:
                t_statistic = math.copysign(math.inf, difference)
                raw_p = 0.0
                effect = math.copysign(math.inf, difference)
        else:
            effect = difference / paired_sd
            t_statistic = effect * math.sqrt(n)
            raw_p = float(2.0 * student_t.sf(abs(t_statistic), n - 1))

        identity_keys = (
            "study", "experiment", "contrast", "dataset", "model", "method",
            "reference", "candidate", "corruption", "attack", "calibration",
            "ood_dataset", "epsilon", "fraction", "sparsity", "metric",
        )
        identity = {key: row[key] for key in identity_keys if row.get(key, "") != ""}
        output.append(
            {
                "family": study.family,
                "submission_stage_role": SUBMISSION_STAGE_ROLE[study.family],
                "source": study.path,
                "source_row": source_index,
                "hypothesis": json.dumps(identity, ensure_ascii=False, sort_keys=True),
                "dataset": row.get("dataset", ""),
                "model": row.get("model", ""),
                "metric": row.get("metric", ""),
                "correction_scope": "|".join(
                    [study.family]
                    + [f"{key}={row.get(key, '')}" for key in SCOPE_FIELDS[study.family]]
                ),
                "n_paired": n,
                "difference_column": diff_column,
                "paired_difference": difference,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "paired_sd": paired_sd,
                "t_statistic": t_statistic,
                "p_value_raw": raw_p,
                "cohen_dz": effect,
                "effect_magnitude": magnitude(effect),
                "original_ci_excludes_zero": bool(ci_low > 0 or ci_high < 0),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, default=Path("statistical_corrections"))
    args = parser.parse_args()
    output_dir = args.output if args.output.is_absolute() else args.root / args.output

    rows: list[dict] = []
    for study in STUDIES:
        rows.extend(standardise(args.root, study))

    families = sorted({row["family"] for row in rows})
    scopes = sorted({row["correction_scope"] for row in rows})
    for scope in scopes:
        indices = [index for index, row in enumerate(rows) if row["correction_scope"] == scope]
        adjusted = holm_adjust([float(rows[index]["p_value_raw"]) for index in indices])
        for index, adjusted_p in zip(indices, adjusted):
            rows[index]["scope_hypotheses"] = len(indices)
            rows[index]["p_value_holm"] = adjusted_p
            rows[index]["holm_significant_0p05"] = bool(adjusted_p < 0.05)
            rows[index]["significance_lost_after_holm"] = bool(
                rows[index]["original_ci_excludes_zero"] and adjusted_p >= 0.05
            )

    # Retain a deliberately conservative family-wide sensitivity analysis.
    for family in families:
        indices = [index for index, row in enumerate(rows) if row["family"] == family]
        adjusted = holm_adjust([float(rows[index]["p_value_raw"]) for index in indices])
        for index, adjusted_p in zip(indices, adjusted):
            rows[index]["family_hypotheses"] = len(indices)
            rows[index]["p_value_holm_family_wide"] = adjusted_p
            rows[index]["holm_family_wide_significant_0p05"] = bool(adjusted_p < 0.05)

    ordered_columns = [
        "family", "submission_stage_role", "correction_scope",
        "scope_hypotheses", "family_hypotheses",
        "source", "source_row", "hypothesis",
        "dataset", "model", "metric", "n_paired", "difference_column",
        "paired_difference", "ci95_low", "ci95_high", "paired_sd",
        "t_statistic", "p_value_raw", "p_value_holm", "holm_significant_0p05",
        "p_value_holm_family_wide", "holm_family_wide_significant_0p05",
        "original_ci_excludes_zero", "significance_lost_after_holm",
        "cohen_dz", "effect_magnitude",
    ]
    rows = [{key: row[key] for key in ordered_columns} for row in rows]
    write_csv(output_dir / "all_core_hypotheses_holm.csv", rows)

    summary: list[dict] = []
    for family in families:
        subset = [row for row in rows if row["family"] == family]
        summary.append(
            {
                "family": family,
                "hypotheses": len(subset),
                "raw_significant": sum(row["original_ci_excludes_zero"] for row in subset),
                "holm_significant": sum(row["holm_significant_0p05"] for row in subset),
                "lost_after_holm": sum(row["significance_lost_after_holm"] for row in subset),
                "family_wide_holm_significant": sum(
                    row["holm_family_wide_significant_0p05"] for row in subset
                ),
                "large_effects": sum(row["effect_magnitude"] == "large" for row in subset),
            }
        )
    write_csv(output_dir / "family_summary.csv", summary)
    significant_rows = [row for row in rows if row["holm_significant_0p05"]]
    lost_rows = [row for row in rows if row["significance_lost_after_holm"]]
    write_csv(output_dir / "holm_significant_hypotheses.csv", significant_rows)
    write_csv(output_dir / "significance_lost_after_holm.csv", lost_rows)
    manifest = {
        "method": "two-sided paired t tests with Holm--Bonferroni correction",
        "alpha": 0.05,
        "effect_size": "paired Cohen's dz = mean paired difference / paired-difference SD",
        "primary_scope": "retrospectively encoded research family crossed with dataset/model and outcome as specified in SCOPE_FIELDS",
        "sensitivity_scope": "all hypotheses in each broad experimental family",
        "prospective_or_preregistered": False,
        "role_field_status": (
            "submission_stage_role is a retrospective manuscript-use label, "
            "not an original planned/exploratory registration"
        ),
        "families": {row["family"]: row["hypotheses"] for row in summary},
        "total_hypotheses": len(rows),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report_lines = [
        "# Multiple-comparison correction report",
        "",
        "All tests are two-sided paired t tests. The scope-level audit applies ",
        "Holm--Bonferroni correction within a retrospectively encoded research family and ",
        "outcome scope (for example, the six first-layer alternatives within one ",
        "dataset and metric). This controls multiplicity for the recorded mapping, ",
        "but it is not evidence of prospective preregistration. ",
        "`submission_stage_role` records how a family is used in the submitted ",
        "dissertation; it is a retrospective editorial label, not an original ",
        "planned/exploratory designation. ",
        "`p_value_holm_family_wide` is a deliberately more ",
        "conservative sensitivity analysis across every hypothesis in a broad ",
        "experimental family. Effect size is paired Cohen's dz.",
        "",
        "| Family | Tests | Raw significant | Holm significant | Lost after Holm | Family-wide sensitivity |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        report_lines.append(
            f"| {row['family']} | {row['hypotheses']} | {row['raw_significant']} | "
            f"{row['holm_significant']} | {row['lost_after_holm']} | "
            f"{row['family_wide_holm_significant']} |"
        )
    report_lines.extend(
        [
            "",
            "Use `all_core_hypotheses_holm.csv` for exact p values and effect sizes. ",
            "Claims whose primary adjusted p value is at least 0.05 must be described ",
            "as inconclusive or exploratory even if their unadjusted 95% interval ",
            "excluded zero.",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
