import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from domain_mf.metrics import validation_aulcs


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Summarise completed v2 runs")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--csv", type=Path, default=ROOT / "outputs" / "summary.csv")
    parser.add_argument(
        "--aggregate-csv", type=Path, default=ROOT / "outputs" / "aggregate_summary.csv"
    )
    return parser.parse_args()


def _mean(rows, key):
    values = [float(row[key]) for row in rows if row[key] != ""]
    return statistics.mean(values) if values else ""


def _std(rows, key):
    values = [float(row[key]) for row in rows if row[key] != ""]
    return statistics.stdev(values) if len(values) > 1 else 0.0


def main():
    args = parse_args()
    rows = []
    for metrics_path in sorted(args.output_root.glob("*/final_metrics.json")):
        config_path = metrics_path.parent / "config.json"
        if not config_path.exists():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        aulc_0t, aulc_1t = validation_aulcs(metrics_path.parent / "history.csv")
        evidence_gap = metrics.get(
            "class_group_evidence_gap", metrics.get("class_selectivity", {})
        )
        rows.append(
            {
                "run": metrics_path.parent.name,
                "dataset": config["dataset"],
                "model": config["model"],
                "method": config["init"],
                "init_layers": config.get("init_layers", ""),
                "seed": config["seed"],
                "epochs": config["epochs"],
                "train_fraction": config["train_fraction"],
                "val_fraction": config["val_fraction"],
                "regularisation": config["reg_schedule"],
                "reg_lambda": config["reg_lambda"],
                "shrinkage": config["shrinkage"],
                "covariance_rank": config.get("covariance_rank", ""),
                "sparsity_lambda": config.get("sparsity_lambda", 0.0),
                "auxiliary_lambda": config.get("auxiliary_lambda", 0.0),
                "augmentation": config.get("augmentation", "none"),
                "lr_scheduler": config.get("lr_scheduler", "none"),
                "min_learning_rate": config.get("min_learning_rate", ""),
                "logit_calibration": not config.get("no_logit_calibration", True),
                "logit_std": config.get("logit_std", ""),
                "device_resolved": config.get("device_resolved", ""),
                "initial_val_accuracy": metrics["initial_val_accuracy"],
                "best_val_accuracy": metrics["best_val_accuracy"],
                "test_accuracy": metrics["test_accuracy"],
                # Canonical AULC is post-update; retain the inclusive value with
                # an explicit name for audits of the original analysis.
                "validation_aulc": aulc_1t,
                "validation_aulc_0T": aulc_0t,
                "validation_aulc_1T": aulc_1t,
                "initial_val_auxiliary_loss": metrics.get(
                    "initial_val_auxiliary_loss", ""
                ),
                "test_auxiliary_loss": metrics.get("test_auxiliary_loss", ""),
                "epoch_to_95pct_best": metrics["epoch_to_95pct_best"],
                "conv1_drift": metrics["kernel_drift"]["conv1"]["mean"],
                "conv2_drift": metrics["kernel_drift"].get("conv2", {}).get("mean", ""),
                "conv1_assignment_stability": metrics["assignment_stability"]["conv1"],
                "conv2_assignment_stability": metrics["assignment_stability"].get("conv2", ""),
                "mean_class_group_evidence_gap": evidence_gap.get("mean", ""),
                "training_seconds": metrics["training_seconds"],
                "initialisation_seconds": config.get("initialisation_seconds", 0.0),
                "parameter_count": metrics.get("parameter_count", ""),
                "trainable_parameter_count": metrics.get(
                    "trainable_parameter_count", ""
                ),
                "state_size_bytes": metrics.get("state_size_bytes", ""),
                "effective_concept_sparsity": metrics.get(
                    "sparsity_metrics", {}
                ).get("fraction_below_5pct_class_max", ""),
                "mean_active_concepts": metrics.get("sparsity_metrics", {}).get(
                    "mean_active_concepts_5pct", ""
                ),
            }
        )
    if not rows:
        raise SystemExit(f"No completed runs under {args.output_root}")
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    group_fields = [
        "dataset", "model", "method", "init_layers", "epochs", "train_fraction",
        "val_fraction", "regularisation", "reg_lambda", "shrinkage",
        "covariance_rank", "sparsity_lambda",
        "auxiliary_lambda", "augmentation", "lr_scheduler", "min_learning_rate",
        "logit_calibration", "logit_std", "device_resolved"
    ]
    metric_fields = [
        "initial_val_accuracy", "best_val_accuracy", "test_accuracy",
        "validation_aulc", "validation_aulc_0T", "validation_aulc_1T",
        "epoch_to_95pct_best", "conv1_drift",
        "initial_val_auxiliary_loss", "test_auxiliary_loss",
        "conv2_drift", "conv1_assignment_stability", "conv2_assignment_stability",
        "mean_class_group_evidence_gap", "training_seconds", "initialisation_seconds",
        "parameter_count", "trainable_parameter_count", "state_size_bytes",
        "effective_concept_sparsity", "mean_active_concepts"
    ]
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in group_fields)].append(row)
    aggregate_rows = []
    for key, group in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        aggregate = dict(zip(group_fields, key))
        aggregate["runs"] = len(group)
        for metric in metric_fields:
            aggregate[f"mean_{metric}"] = _mean(group, metric)
            aggregate[f"std_{metric}"] = _std(group, metric)
        aggregate_rows.append(aggregate)
    aggregate_fields = group_fields + ["runs"] + [
        value for metric in metric_fields for value in (f"mean_{metric}", f"std_{metric}")
    ]
    args.aggregate_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.aggregate_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=aggregate_fields)
        writer.writeheader()
        writer.writerows(aggregate_rows)
    for row in rows:
        print(
            f"{row['method']:10s} {row['regularisation']:6s} "
            f"init={row['initial_val_accuracy']:6.2f}% "
            f"test={row['test_accuracy']:6.2f}% "
            f"AULC={row['validation_aulc']:6.2f}"
        )
    print(f"Wrote {len(rows)} rows to {args.csv.resolve()}")
    print(f"Wrote {len(aggregate_rows)} aggregate rows to {args.aggregate_csv.resolve()}")


if __name__ == "__main__":
    main()
