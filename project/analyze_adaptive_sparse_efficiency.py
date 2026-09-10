"""Summarise the bounded adaptive, sparse-concept and pruning pilots."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

import torch
from scipy.stats import t as student_t

from domain_mf.metrics import validation_aulcs


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "extension_results" / "adaptive_sparse_efficiency_pilot"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def adaptive_rows() -> list[dict]:
    rows = []
    root = ROOT / "outputs_adaptive_pilot"
    for config_path in sorted(root.glob("*/config.json")):
        config = read_json(config_path)
        selection = config.get("adaptive_selection")
        if not selection:
            continue
        metrics = read_json(config_path.parent / "final_metrics.json")
        for candidate in selection["candidates"]:
            rows.append(
                {
                    "dataset": config["dataset"],
                    "seed": config["seed"],
                    "candidate": candidate["candidate"],
                    "selected": candidate["candidate"]
                    == selection["selected_candidate"],
                    "validation_accuracy": candidate["validation_accuracy"],
                    "validation_loss": candidate["validation_loss"],
                    "candidate_initialisation_seconds": candidate[
                        "initialisation_seconds"
                    ],
                    "adaptive_total_seconds": selection["total_seconds"],
                    "selected_test_accuracy": metrics["test_accuracy"],
                    "selection_criterion": selection["criterion"],
                }
            )
    return rows


def concept_runs(root: Path, mode: str) -> dict[int, dict]:
    runs = {}
    for metrics_path in sorted(root.glob("*/final_metrics.json")):
        config = read_json(metrics_path.parent / "config.json")
        if (
            config.get("dataset") != "fashion"
            or config.get("model") != "concept"
            or config.get("init") != "di_wmf"
        ):
            continue
        metrics = read_json(metrics_path)
        _, post_update_aulc = validation_aulcs(metrics_path.parent / "history.csv")
        sparsity = dict(metrics.get("sparsity_metrics") or {})
        if "fraction_exactly_zero" not in sparsity:
            state = torch.load(
                metrics_path.parent / "checkpoint_best.pt",
                map_location="cpu",
                weights_only=True,
            )
            weights = state["classifier.weight"]
            sparsity["fraction_exactly_zero"] = float(
                (weights == 0).float().mean()
            )
        runs[int(config["seed"])] = {
            "mode": mode,
            "test_accuracy": float(metrics["test_accuracy"]),
            "validation_aulc": post_update_aulc,
            "fraction_exactly_zero": float(sparsity["fraction_exactly_zero"]),
            "fraction_below_5pct": float(
                sparsity["fraction_below_5pct_class_max"]
            ),
            "active_concepts_5pct": float(sparsity["mean_active_concepts_5pct"]),
        }
    return runs


def paired_summary(reference: dict[int, dict], candidate: dict[int, dict]):
    metrics = [
        "test_accuracy",
        "validation_aulc",
        "fraction_exactly_zero",
        "fraction_below_5pct",
        "active_concepts_5pct",
    ]
    seeds = sorted(set(reference) & set(candidate))
    rows = []
    for metric in metrics:
        before = [reference[seed][metric] for seed in seeds]
        after = [candidate[seed][metric] for seed in seeds]
        differences = [b - a for a, b in zip(before, after)]
        mean = statistics.mean(differences)
        std = statistics.stdev(differences) if len(differences) > 1 else 0.0
        se = std / math.sqrt(len(differences)) if differences else float("nan")
        half = (
            float(student_t.ppf(0.975, len(differences) - 1)) * se
            if len(differences) > 1
            else 0.0
        )
        statistic = mean / se if se > 0 else float("inf") if mean else 0.0
        p_value = (
            float(2 * student_t.sf(abs(statistic), len(differences) - 1))
            if len(differences) > 1 and se > 0
            else 0.0 if mean else 1.0
        )
        rows.append(
            {
                "metric": metric,
                "paired_seeds": len(seeds),
                "l1_mean": statistics.mean(before),
                "proximal_mean": statistics.mean(after),
                "proximal_minus_l1": mean,
                "ci95_low": mean - half,
                "ci95_high": mean + half,
                "paired_p": p_value,
                "paired_dz": mean / std if std > 0 else "",
            }
        )
    return rows


def pruning_rows() -> list[dict]:
    path = (
        ROOT
        / "structured_efficiency_results"
        / "finetune_pilot"
        / "structured_accuracy_rows.csv"
    )
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    adaptive = adaptive_rows()
    write_csv(OUTPUT / "adaptive_candidate_scores.csv", adaptive)

    l1 = concept_runs(ROOT / "outputs_frontier_arch", "l1_lambda0.05")
    proximal = concept_runs(
        ROOT / "outputs_concept_proximal_5seed", "proximal_lambda10"
    )
    concept_rows = []
    for seed in sorted(set(l1) & set(proximal)):
        concept_rows.extend(
            [
                {"seed": seed, **l1[seed]},
                {"seed": seed, **proximal[seed]},
            ]
        )
    write_csv(OUTPUT / "concept_sparsity_runs.csv", concept_rows)
    paired = paired_summary(l1, proximal)
    write_csv(OUTPUT / "concept_sparsity_paired.csv", paired)

    pruning = pruning_rows()
    write_csv(OUTPUT / "pruning_finetune_pilot.csv", pruning)

    selected = {
        row["dataset"]: row["candidate"]
        for row in adaptive
        if str(row["selected"]).lower() == "true" or row["selected"] is True
    }
    paired_by_metric = {row["metric"]: row for row in paired}
    accuracy = paired_by_metric["test_accuracy"]
    zeros = paired_by_metric["fraction_exactly_zero"]
    active = paired_by_metric["active_concepts_5pct"]
    pruning_note = "No pruning result was found."
    if pruning:
        row = pruning[0]
        pruning_note = (
            f"The 50% channel-pruned model recovered from "
            f"{float(row['pre_finetune_accuracy']):.2f}% to "
            f"{float(row['pruned_accuracy']):.2f}% after "
            f"{row['finetune_epochs']} epochs, with "
            f"{100*float(row['parameter_reduction']):.1f}% fewer parameters."
        )
    report = f"""# Adaptive, sparse-concept and pruning pilot

These are bounded extension experiments. They are not mixed into the primary
Holm-confirmed result families.

## Adaptive DI-WMF

The Epoch-0 validation selector chose `{selected.get('fashion')}` on Fashion,
`{selected.get('cifar10')}` on CIFAR-10 and `{selected.get('sign')}` on Sign.
It therefore reproduced the fixed rank-16 choice in this one-seed pilot and did
not demonstrate task-dependent selection. A larger run is not justified under
the current criterion. A future selector should use a short training-probe AULC
or a nested task-specific utility, which would require substantially more
candidate training.

## Proximal shared-concept head

Across five paired Fashion seeds, proximal shrinkage changed test accuracy by
{accuracy['proximal_minus_l1']:.3f} percentage points
[{accuracy['ci95_low']:.3f}, {accuracy['ci95_high']:.3f}] relative to the
existing L1 run. Exact-zero weight fraction changed by
{zeros['proximal_minus_l1']:.3f}, and the mean number of concepts above 5% of
the per-class maximum changed by {active['proximal_minus_l1']:.2f}.
This is a real sparsity--accuracy trade-off, not a free accuracy improvement.
Lambda 10 was chosen after a seed-0 smoke sweep, so this remains exploratory.

## Pruning recovery

{pruning_note} This is a one-seed feasibility result and does not yet establish
accuracy-preserving compression.
"""
    (OUTPUT / "PILOT_RESULTS.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
