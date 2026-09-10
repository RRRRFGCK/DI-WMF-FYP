"""Angle-sweep evaluation and exact C4-invariance diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from domain_mf.data import build_test_dataset
from domain_mf.models import build_model


ROOT = Path(__file__).resolve().parent
DEFAULT_ANGLES = (-180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180)


def parse_args():
    parser = argparse.ArgumentParser(description="Measure rotation invariance across angles")
    parser.add_argument(
        "--roots", nargs="+", type=Path, default=[ROOT / "outputs_rotation"]
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "rotation_invariance_results"
    )
    parser.add_argument(
        "--models", nargs="+", default=["standard", "rotation_invariant"]
    )
    parser.add_argument(
        "--methods", nargs="+", default=["kaiming", "lowrank_wmf"]
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["fashion", "cifar10", "sign"]
    )
    parser.add_argument("--angles", nargs="+", type=float, default=list(DEFAULT_ANGLES))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def rotate_batch(inputs: torch.Tensor, angle: float) -> torch.Tensor:
    quarter_turn = angle / 90.0
    if abs(quarter_turn - round(quarter_turn)) < 1e-9:
        return torch.rot90(inputs, int(round(quarter_turn)) % 4, dims=(-2, -1))
    return TF.rotate(
        inputs,
        float(angle),
        interpolation=InterpolationMode.BILINEAR,
        fill=0.0,
    )


def discover_runs(args):
    runs = []
    for root in args.roots:
        for metrics_path in root.rglob("final_metrics.json"):
            run_dir = metrics_path.parent
            config_path = run_dir / "config.json"
            checkpoint = run_dir / "checkpoint_best.pt"
            if not config_path.exists() or not checkpoint.exists():
                continue
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if (
                config.get("dataset") in args.datasets
                and config.get("model") in args.models
                and config.get("init") in args.methods
            ):
                runs.append((run_dir, config, checkpoint))
    return runs


def evaluate_run(model, loader, angles, device):
    totals = {angle: {"correct": 0, "agreement": 0, "prob_l1": 0.0, "count": 0, "max_logit": 0.0} for angle in angles}
    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")
            clean_logits = model(inputs)
            clean_probabilities = F.softmax(clean_logits, dim=1)
            clean_predictions = clean_logits.argmax(1)
            for angle in angles:
                rotated_logits = model(rotate_batch(inputs, angle))
                probabilities = F.softmax(rotated_logits, dim=1)
                predictions = rotated_logits.argmax(1)
                item = totals[angle]
                item["correct"] += int((predictions == labels).sum())
                item["agreement"] += int((predictions == clean_predictions).sum())
                item["prob_l1"] += float(
                    (probabilities - clean_probabilities).abs().mean(dim=1).sum()
                )
                item["max_logit"] = max(
                    item["max_logit"],
                    float((rotated_logits - clean_logits).abs().max()),
                )
                item["count"] += labels.numel()
    return {
        angle: {
            "accuracy": 100.0 * item["correct"] / item["count"],
            "prediction_agreement": item["agreement"] / item["count"],
            "mean_probability_l1": item["prob_l1"] / item["count"],
            "max_absolute_logit_difference": item["max_logit"],
        }
        for angle, item in totals.items()
    }


def _write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean_ci(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    ci = (
        float(student_t.ppf(0.975, values.size - 1) * values.std(ddof=1) / math.sqrt(values.size))
        if values.size > 1
        else 0.0
    )
    return mean, ci


def aggregate_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["model"], row["method"], row["angle"])].append(row)
    aggregates = []
    for key, group in sorted(grouped.items()):
        result = {
            "dataset": key[0],
            "model": key[1],
            "method": key[2],
            "angle": key[3],
            "runs": len(group),
        }
        for metric in (
            "accuracy",
            "prediction_agreement",
            "mean_probability_l1",
            "max_absolute_logit_difference",
        ):
            mean, ci = _mean_ci([row[metric] for row in group])
            result[f"{metric}_mean"] = mean
            result[f"{metric}_ci95"] = ci
        aggregates.append(result)
    return aggregates


def summary_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["model"], row["method"], row["seed"])].append(row)
    summaries = []
    for key, group in sorted(grouped.items()):
        arbitrary = [row for row in group if row["angle"] != 0]
        c4 = [row for row in group if row["angle"] != 0 and row["angle"] % 90 == 0]
        clean = next(row for row in group if row["angle"] == 0)
        summaries.append(
            {
                "dataset": key[0],
                "model": key[1],
                "method": key[2],
                "seed": key[3],
                "clean_accuracy": clean["accuracy"],
                "mean_rotated_accuracy": float(np.mean([row["accuracy"] for row in arbitrary])),
                "worst_angle_accuracy": min(row["accuracy"] for row in arbitrary),
                "c4_prediction_agreement": float(np.mean([row["prediction_agreement"] for row in c4])),
                "c4_probability_l1": float(np.mean([row["mean_probability_l1"] for row in c4])),
                "c4_max_absolute_logit_difference": max(row["max_absolute_logit_difference"] for row in c4),
            }
        )
    return summaries


def plot_results(aggregates, output_dir):
    datasets = sorted({row["dataset"] for row in aggregates})
    figure, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4), squeeze=False)
    for column, dataset in enumerate(datasets):
        selected_dataset = [row for row in aggregates if row["dataset"] == dataset]
        for model, method in sorted({(row["model"], row["method"]) for row in selected_dataset}):
            selected = sorted(
                [row for row in selected_dataset if row["model"] == model and row["method"] == method],
                key=lambda row: row["angle"],
            )
            axes[0, column].plot(
                [row["angle"] for row in selected],
                [row["accuracy_mean"] for row in selected],
                marker="o",
                label=f"{model}/{method}",
            )
        axes[0, column].set_title(dataset)
        axes[0, column].set_xlabel("Rotation angle (degrees)")
        axes[0, column].set_ylabel("Accuracy (%)")
        axes[0, column].grid(alpha=0.25)
        axes[0, column].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "rotation_angle_sweep.png", dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = discover_runs(args)
    if not runs:
        raise RuntimeError("No compatible checkpoints were found")
    loader_cache = {}
    rows = []
    for run_dir, config, checkpoint in runs:
        dataset = config["dataset"]
        cache_key = (dataset, config["data_root"], config["sign_root"])
        if cache_key not in loader_cache:
            test_dataset = build_test_dataset(
                dataset, Path(config["data_root"]), Path(config["sign_root"])
            )
            loader_cache[cache_key] = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
            )
        model = build_model(config["model"], dataset).to(device)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        results = evaluate_run(model, loader_cache[cache_key], args.angles, device)
        for angle, metrics in results.items():
            rows.append(
                {
                    "dataset": dataset,
                    "model": config["model"],
                    "method": config["init"],
                    "seed": int(config["seed"]),
                    "angle": float(angle),
                    **metrics,
                    "run_dir": str(run_dir.resolve()),
                }
            )
        print(f"evaluated {dataset}/{config['model']}/{config['init']}/seed{config['seed']}")
    aggregates = aggregate_rows(rows)
    summaries = summary_rows(rows)
    _write_csv(args.output_dir / "rotation_rows.csv", rows)
    _write_csv(args.output_dir / "rotation_aggregate.csv", aggregates)
    _write_csv(args.output_dir / "rotation_summary.csv", summaries)
    plot_results(aggregates, args.output_dir)
    metadata = {
        "runs": len(runs),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "angles": args.angles,
        "exact_group": "C4 = {0, 90, 180, 270 degrees}",
        "arbitrary_angles_use": "bilinear interpolation with zero fill",
        "confidence_interval": "two-sided 95% Student-t interval across seeds",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
