"""Compare clean and modern CIFAR-10 training augmentation protocols."""

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from domain_mf.metrics import validation_aulcs


T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}
METRICS = ["initial_val_accuracy", "test_accuracy", "validation_aulc"]


def parse_args():
    parser = argparse.ArgumentParser(description="Analyse CIFAR augmentation")
    parser.add_argument("--output-roots", nargs="+", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, default=Path("augmentation_results"))
    return parser.parse_args()


def load_rows(roots):
    rows = []
    seen = set()
    for root in roots:
        for config_path in root.glob("*/config.json"):
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if (
                config.get("dataset") != "cifar10"
                or config.get("model") != "standard"
                or config.get("init") not in {"kaiming", "lowrank_wmf"}
                or int(config.get("epochs", -1)) != 20
                or float(config.get("train_fraction", -1)) != 1.0
            ):
                continue
            metrics_path = config_path.parent / "final_metrics.json"
            if not metrics_path.exists():
                continue
            augmentation = config.get("augmentation", "none")
            key = (augmentation, config["init"], int(config["seed"]))
            if key in seen:
                raise RuntimeError(f"Duplicate augmentation run: {key}")
            seen.add(key)
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            _, post_update_aulc = validation_aulcs(config_path.parent / "history.csv")
            metrics["validation_aulc"] = post_update_aulc
            rows.append(
                {
                    "augmentation": augmentation, "method": config["init"],
                    "seed": int(config["seed"]),
                    **{metric: metrics[metric] for metric in METRICS},
                    "device_resolved": config.get("device_resolved", ""),
                }
            )
    expected = {
        (augmentation, method, seed)
        for augmentation in ("none", "cifar_standard")
        for method in ("kaiming", "lowrank_wmf")
        for seed in range(5)
    }
    found = {(row["augmentation"], row["method"], row["seed"]) for row in rows}
    if expected - found:
        raise RuntimeError(f"Missing augmentation runs: {sorted(expected - found)[:5]}")
    return rows


def interval(differences):
    mean = statistics.mean(differences)
    std = statistics.stdev(differences) if len(differences) > 1 else 0.0
    half = T95[len(differences) - 1] * std / math.sqrt(len(differences)) if len(differences) > 1 else 0.0
    return mean, mean - half, mean + half


def main():
    args = parse_args()
    rows = load_rows(args.output_roots)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["augmentation"], row["method"])].append(row)
    aggregates = []
    for key, group in sorted(grouped.items()):
        item = {"augmentation": key[0], "method": key[1], "runs": len(group)}
        for metric in METRICS:
            values = [float(row[metric]) for row in group]
            item[f"mean_{metric}"] = statistics.mean(values)
            item[f"std_{metric}"] = statistics.stdev(values)
        aggregates.append(item)
    indexed = {
        (row["augmentation"], row["method"], row["seed"]): row for row in rows
    }
    paired = []
    comparisons = [
        ("lowrank_minus_kaiming_clean", ("none", "kaiming"), ("none", "lowrank_wmf")),
        ("lowrank_minus_kaiming_augmented", ("cifar_standard", "kaiming"), ("cifar_standard", "lowrank_wmf")),
        ("augmentation_effect_kaiming", ("none", "kaiming"), ("cifar_standard", "kaiming")),
        ("augmentation_effect_lowrank", ("none", "lowrank_wmf"), ("cifar_standard", "lowrank_wmf")),
    ]
    for name, reference, candidate in comparisons:
        for metric in METRICS:
            differences = [
                float(indexed[(*candidate, seed)][metric])
                - float(indexed[(*reference, seed)][metric])
                for seed in range(5)
            ]
            mean, low, high = interval(differences)
            paired.append(
                {
                    "contrast": name, "metric": metric, "paired_seeds": 5,
                    "mean_candidate_minus_reference": mean,
                    "ci95_low": low, "ci95_high": high,
                    "candidate_wins": sum(value > 0 for value in differences),
                }
            )
    args.results_root.mkdir(parents=True, exist_ok=True)
    for filename, output in (("augmentation_rows.csv", rows), ("augmentation_aggregate.csv", aggregates), ("augmentation_paired.csv", paired)):
        with (args.results_root / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(output[0]))
            writer.writeheader()
            writer.writerows(output)
    print(f"Wrote augmentation results to {args.results_root.resolve()}")


if __name__ == "__main__":
    main()
