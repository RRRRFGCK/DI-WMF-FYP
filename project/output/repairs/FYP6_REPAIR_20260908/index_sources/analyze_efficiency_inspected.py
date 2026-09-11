"""Aggregate width-scaling experiments and compare compact DI-WMF models.

The primary comparison is deliberately demanding: each compact low-rank
DI-WMF model is seed-paired against the full-width Kaiming model on the same
dataset and training split. This distinguishes optimisation efficiency from a
mere within-width initialisation comparison.
"""

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

from domain_mf.models import build_model
from domain_mf.metrics import validation_aulcs


T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}


def parse_args():
    parser = argparse.ArgumentParser(description="Analyse parameter-efficiency runs")
    parser.add_argument("--roots", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("efficiency_aggregate.csv"))
    parser.add_argument(
        "--paired-output", type=Path, default=Path("efficiency_paired.csv")
    )
    return parser.parse_args()


def read_rows(roots):
    rows = []
    for root in roots:
        summary = root / "summary.csv"
        if not summary.exists():
            continue
        with summary.open(newline="", encoding="utf-8") as handle:
            root_rows = list(csv.DictReader(handle))
        for row in root_rows:
            history = root / row["run"] / "history.csv"
            if history.exists():
                row["validation_aulc_0T"], row["validation_aulc"] = (
                    str(value) for value in validation_aulcs(history)
                )
        rows.extend(root_rows)
    if not rows:
        raise ValueError("No summary rows found")
    for row in rows:
        if not row.get("parameter_count"):
            row["parameter_count"] = str(
                sum(
                    parameter.numel()
                    for parameter in build_model(row["model"], row["dataset"]).parameters()
                )
            )
    return rows


def mean(rows, key):
    return statistics.mean(float(row[key]) for row in rows)


def stdev(rows, key):
    values = [float(row[key]) for row in rows]
    return statistics.stdev(values) if len(values) > 1 else 0.0


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["model"], row["method"])].append(row)
    full_parameters = {
        (dataset, method): mean(group, "parameter_count")
        for (dataset, model, method), group in groups.items()
        if model == "standard"
    }
    output = []
    for (dataset, model, method), group in sorted(groups.items()):
        parameters = mean(group, "parameter_count")
        reference_parameters = full_parameters.get((dataset, method), parameters)
        output.append(
            {
                "dataset": dataset,
                "model": model,
                "method": method,
                "runs": len(group),
                "parameter_count": parameters,
                "parameter_reduction_vs_full": 1.0 - parameters / reference_parameters,
                "mean_initial_val_accuracy": mean(group, "initial_val_accuracy"),
                "std_initial_val_accuracy": stdev(group, "initial_val_accuracy"),
                "mean_test_accuracy": mean(group, "test_accuracy"),
                "std_test_accuracy": stdev(group, "test_accuracy"),
                "mean_validation_aulc": mean(group, "validation_aulc"),
                "std_validation_aulc": stdev(group, "validation_aulc"),
                "mean_training_seconds": mean(group, "training_seconds"),
                "accuracy_per_1000_parameters": 1000.0 * mean(group, "test_accuracy") / parameters,
            }
        )
    return output


def paired_summary(reference, candidate, metric):
    ref = {int(row["seed"]): float(row[metric]) for row in reference}
    comp = {int(row["seed"]): float(row[metric]) for row in candidate}
    seeds = sorted(set(ref) & set(comp))
    differences = [comp[seed] - ref[seed] for seed in seeds]
    if len(differences) < 2:
        raise ValueError("At least two paired seeds are required")
    difference_mean = statistics.mean(differences)
    difference_std = statistics.stdev(differences)
    half_width = T95.get(len(seeds) - 1, 1.96) * difference_std / math.sqrt(len(seeds))
    return {
        "metric": metric,
        "paired_seeds": len(seeds),
        "reference_mean": statistics.mean(ref[seed] for seed in seeds),
        "candidate_mean": statistics.mean(comp[seed] for seed in seeds),
        "mean_candidate_minus_reference": difference_mean,
        "ci95_low": difference_mean - half_width,
        "ci95_high": difference_mean + half_width,
        "candidate_wins": sum(comp[seed] > ref[seed] for seed in seeds),
    }


def paired(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["model"], row["method"])].append(row)
    output = []
    datasets = sorted({row["dataset"] for row in rows})
    comparisons = []
    for dataset in datasets:
        for model in ("standard_half", "standard_quarter"):
            # The first contrast asks whether prior knowledge can compensate for
            # a smaller network. The second measures the pure compression cost,
            # while the third isolates initialization within a fixed width.
            comparisons.extend(
                [
                    (dataset, "standard", "kaiming", model, "lowrank_wmf", "compact_DI_vs_full_random"),
                    (dataset, "standard", "lowrank_wmf", model, "lowrank_wmf", "compact_vs_full_DI"),
                    (dataset, model, "kaiming", model, "lowrank_wmf", "DI_vs_random_same_width"),
                ]
            )
    for dataset, ref_model, ref_method, cand_model, cand_method, contrast in comparisons:
        reference = groups.get((dataset, ref_model, ref_method), [])
        candidate = groups.get((dataset, cand_model, cand_method), [])
        if not reference or not candidate:
            continue
        reference_parameters = float(reference[0]["parameter_count"])
        candidate_parameters = float(candidate[0]["parameter_count"])
        for metric in (
            "initial_val_accuracy", "test_accuracy", "validation_aulc",
            "training_seconds",
        ):
            item = paired_summary(reference, candidate, metric)
            item.update(
                {
                    "contrast": contrast,
                    "dataset": dataset,
                    "reference_model": ref_model,
                    "reference_method": ref_method,
                    "candidate_model": cand_model,
                    "candidate_method": cand_method,
                    "parameter_ratio": candidate_parameters / reference_parameters,
                    "parameter_reduction": 1.0 - candidate_parameters / reference_parameters,
                }
            )
            output.append(item)
    return output


def write_csv(path, rows):
    if not rows:
        raise ValueError(
            f"No rows available for {path}; check that every --roots directory "
            "contains a summary.csv with matching datasets, models and seeds"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    rows = read_rows(args.roots)
    aggregate_rows = aggregate(rows)
    paired_rows = paired(rows)
    write_csv(args.output, aggregate_rows)
    write_csv(args.paired_output, paired_rows)
    print(f"Wrote {len(aggregate_rows)} aggregate rows to {args.output.resolve()}")
    print(f"Wrote {len(paired_rows)} paired rows to {args.paired_output.resolve()}")


if __name__ == "__main__":
    main()
