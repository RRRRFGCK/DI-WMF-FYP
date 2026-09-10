import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

from domain_mf.metrics import validation_aulcs


ROOT = Path(__file__).resolve().parent
T95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute seed-paired comparisons for the frontier studies"
    )
    parser.add_argument("--experiment-a", type=Path, default=ROOT / "outputs_frontier_a")
    parser.add_argument("--experiment-b", type=Path, default=ROOT / "outputs_frontier_b")
    parser.add_argument("--architectures", type=Path, default=ROOT / "outputs_frontier_arch")
    parser.add_argument(
        "--corruption-roots", nargs="*", type=Path, default=[]
    )
    parser.add_argument("--cross-task-b", type=Path)
    parser.add_argument("--cross-task-architectures", type=Path)
    parser.add_argument("--long-horizon-roots", nargs="*", type=Path, default=[])
    parser.add_argument(
        "--skip-default-studies",
        action="store_true",
        help="Analyse only the explicitly supplied study roots.",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "frontier_paired_comparisons.csv"
    )
    return parser.parse_args()


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def method_label(row):
    label = row["method"]
    if label == "lowrank_wmf":
        label += f"_rank{int(float(row['covariance_rank']))}"
    return label


def paired_summary(
    study,
    dataset,
    model,
    reference,
    method,
    metric,
    reference_rows,
    method_rows,
    corruption="",
):
    reference_by_seed = {int(row["seed"]): float(row[metric]) for row in reference_rows}
    method_by_seed = {int(row["seed"]): float(row[metric]) for row in method_rows}
    seeds = sorted(set(reference_by_seed) & set(method_by_seed))
    if len(seeds) < 2:
        raise ValueError(f"Need at least two paired seeds for {study}: {method}")
    reference_values = [reference_by_seed[seed] for seed in seeds]
    method_values = [method_by_seed[seed] for seed in seeds]
    differences = [value - baseline for value, baseline in zip(method_values, reference_values)]
    difference_mean = statistics.mean(differences)
    difference_std = statistics.stdev(differences)
    critical = T95.get(len(seeds) - 1, 1.96)
    half_width = critical * difference_std / math.sqrt(len(seeds))
    return {
        "study": study,
        "dataset": dataset,
        "model": model,
        "corruption": corruption,
        "reference": reference,
        "method": method,
        "metric": metric,
        "paired_seeds": len(seeds),
        "reference_mean": statistics.mean(reference_values),
        "method_mean": statistics.mean(method_values),
        "mean_paired_difference": difference_mean,
        "paired_difference_std": difference_std,
        "ci95_low": difference_mean - half_width,
        "ci95_high": difference_mean + half_width,
        "method_wins": sum(value > baseline for value, baseline in zip(method_values, reference_values)),
    }


def comparisons_from_training(study, path, reference="kaiming"):
    rows = read_csv(path / "summary.csv")
    for row in rows:
        history = path / row["run"] / "history.csv"
        if history.exists():
            _, post_update_aulc = validation_aulcs(history)
            row["validation_aulc"] = str(post_update_aulc)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["model"], method_label(row))].append(row)
    output = []
    dataset_models = sorted({(key[0], key[1]) for key in grouped})
    for dataset, model in dataset_models:
        reference_rows = grouped[(dataset, model, reference)]
        candidates = sorted(
            key[2]
            for key in grouped
            if key[0] == dataset and key[1] == model
        )
        for candidate in candidates:
            if candidate == reference:
                continue
            for metric in (
                "initial_val_accuracy",
                "test_accuracy",
                "validation_aulc",
                "conv1_drift",
            ):
                output.append(
                    paired_summary(
                        study,
                        dataset,
                        model,
                        reference,
                        candidate,
                        metric,
                        reference_rows,
                        grouped[(dataset, model, candidate)],
                    )
                )
    return output


def comparisons_from_corruptions(path, reference="kaiming"):
    rows = read_csv(path / "corruption_model_metrics.csv")
    grouped = defaultdict(list)
    for row in rows:
        grouped[
            (row["dataset"], row["model"], row["corruption"], row["method"])
        ].append(row)
    output = []
    for dataset, model, corruption, method in sorted(grouped):
        if method == reference:
            continue
        reference_rows = grouped[(dataset, model, corruption, reference)]
        # Reuse the training comparison helper by normalising the seed field.
        method_rows = grouped[(dataset, model, corruption, method)]
        for row in reference_rows + method_rows:
            row["seed"] = row["model_seed"]
        for metric in (
            "normalised_accuracy_auc",
            "worst_accuracy",
            "mean_accuracy_drop",
            "mean_retention",
            "mean_consistency",
        ):
            output.append(
                paired_summary(
                    f"corruption:{corruption}",
                    dataset,
                    model,
                    reference,
                    method,
                    metric,
                    reference_rows,
                    method_rows,
                    corruption,
                )
            )
    return output


def main():
    args = parse_args()
    rows = []
    if not args.skip_default_studies:
        rows.extend(comparisons_from_training("A:stem-only", args.experiment_a))
        rows.extend(
            comparisons_from_training("B:layer-wise", args.experiment_b, "di_wmf")
        )
        rows.extend(comparisons_from_training("architecture", args.architectures))
    if args.cross_task_b:
        rows.extend(
            comparisons_from_training(
                "cross-task:B:layer-wise", args.cross_task_b, "di_wmf"
            )
        )
    if args.cross_task_architectures:
        rows.extend(
            comparisons_from_training(
                "cross-task:architecture", args.cross_task_architectures
            )
        )
    for root in args.long_horizon_roots:
        rows.extend(
            comparisons_from_training(f"long-horizon:{root.name}", root)
        )
    for root in args.corruption_roots:
        rows.extend(comparisons_from_corruptions(root))
    if not rows:
        raise ValueError("No comparisons were requested")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} paired comparisons to {args.output.resolve()}")


if __name__ == "__main__":
    main()
