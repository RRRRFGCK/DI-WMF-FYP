import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose whether validation curves have plateaued near the horizon"
    )
    parser.add_argument("--roots", nargs="+", type=Path, required=True)
    parser.add_argument("--tail-epochs", type=int, default=5)
    parser.add_argument("--slope-threshold", type=float, default=0.15)
    parser.add_argument("--gap-threshold", type=float, default=0.5)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "convergence_diagnostics.csv"
    )
    return parser.parse_args()


def _linear_slope(points):
    x_values = [float(point[0]) for point in points]
    y_values = [float(point[1]) for point in points]
    x_mean = statistics.mean(x_values)
    y_mean = statistics.mean(y_values)
    denominator = sum((value - x_mean) ** 2 for value in x_values)
    if denominator == 0:
        return 0.0
    return sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x_values, y_values)
    ) / denominator


def _method_label(config):
    label = config["init"]
    if label == "lowrank_wmf":
        label += f"_rank{int(config['covariance_rank'])}"
    return label


def read_runs(roots, tail_epochs, slope_threshold, gap_threshold):
    rows = []
    for root in roots:
        for config_path in sorted(root.glob("*/config.json")):
            run_dir = config_path.parent
            history_path = run_dir / "history.csv"
            metrics_path = run_dir / "final_metrics.json"
            if not history_path.exists() or not metrics_path.exists():
                continue
            config = json.loads(config_path.read_text(encoding="utf-8"))
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            with history_path.open(newline="", encoding="utf-8") as handle:
                history = list(csv.DictReader(handle))
            trained = [row for row in history if int(row["epoch"]) > 0]
            tail = trained[-tail_epochs:]
            if len(tail) < 2:
                raise ValueError(f"Not enough tail epochs in {run_dir}")
            slope = _linear_slope(
                [(int(row["epoch"]), float(row["val_accuracy"])) for row in tail]
            )
            final_accuracy = float(trained[-1]["val_accuracy"])
            best_accuracy = float(metrics["best_val_accuracy"])
            gap = best_accuracy - final_accuracy
            converged = abs(slope) <= slope_threshold and gap <= gap_threshold
            rows.append(
                {
                    "dataset": config["dataset"],
                    "model": config["model"],
                    "method": _method_label(config),
                    "seed": int(config["seed"]),
                    "epochs": int(config["epochs"]),
                    "best_epoch": int(metrics["best_epoch"]),
                    "best_val_accuracy": best_accuracy,
                    "final_val_accuracy": final_accuracy,
                    "tail_slope_points_per_epoch": slope,
                    "tail_gain": final_accuracy - float(tail[0]["val_accuracy"]),
                    "best_final_gap": gap,
                    "converged": converged,
                }
            )
    return rows


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["model"], row["method"])].append(row)
    output = []
    for (dataset, model, method), group in sorted(grouped.items()):
        output.append(
            {
                "dataset": dataset,
                "model": model,
                "method": method,
                "runs": len(group),
                "mean_best_epoch": statistics.mean(row["best_epoch"] for row in group),
                "mean_tail_slope_points_per_epoch": statistics.mean(
                    row["tail_slope_points_per_epoch"] for row in group
                ),
                "mean_tail_gain": statistics.mean(row["tail_gain"] for row in group),
                "mean_best_final_gap": statistics.mean(
                    row["best_final_gap"] for row in group
                ),
                "converged_runs": sum(bool(row["converged"]) for row in group),
            }
        )
    return output


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    rows = read_runs(
        args.roots,
        args.tail_epochs,
        args.slope_threshold,
        args.gap_threshold,
    )
    if not rows:
        raise ValueError("No completed runs found")
    aggregates = aggregate(rows)
    write_csv(args.output, rows)
    aggregate_path = args.output.with_name(f"{args.output.stem}_aggregate.csv")
    write_csv(aggregate_path, aggregates)
    print(f"Wrote {len(rows)} run diagnostics to {args.output.resolve()}")
    print(f"Wrote {len(aggregates)} aggregate rows to {aggregate_path.resolve()}")


if __name__ == "__main__":
    main()
