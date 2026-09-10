import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from domain_mf.data import build_test_dataset
from domain_mf.models import build_model


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate assigned- and best-class selectivity of conv1 filters"
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=["fashion"])
    parser.add_argument("--models", nargs="+", default=["standard"])
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--init-layers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def discover_runs(args):
    expected = {
        (dataset, model, method, seed)
        for dataset in args.datasets
        for model in args.models
        for method in args.methods
        for seed in args.seeds
    }
    found = {}
    for config_path in sorted(args.output_root.glob("*/config.json")):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        key = (
            config.get("dataset"),
            config.get("model"),
            config.get("init"),
            int(config.get("seed", -1)),
        )
        if key not in expected:
            continue
        if int(config.get("epochs", -1)) != args.epochs:
            continue
        if int(config.get("init_layers", -1)) != args.init_layers:
            continue
        checkpoint = config_path.parent / "checkpoint_best.pt"
        if not checkpoint.exists():
            continue
        if key in found:
            raise RuntimeError(f"Multiple completed runs match {key}")
        found[key] = (config_path, config, checkpoint)
    missing = sorted(expected - set(found))
    if missing:
        raise RuntimeError(f"Missing {len(missing)} requested runs: {missing[:5]}")
    return found


def conv1_class_means(model, loader, device):
    channels = model.conv1.out_channels
    classes = model.spec.num_classes
    sums = torch.zeros(classes, channels, device=device, dtype=torch.float64)
    counts = torch.zeros(classes, device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")
            responses = F.relu(model.conv1(inputs)).mean(dim=(2, 3)).double()
            sums.index_add_(0, labels, responses)
            counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float64))
    return sums / counts[:, None].clamp_min(1)


def selectivity_metrics(model, class_means):
    classes, channels = class_means.shape
    channel_indices = torch.arange(channels, device=class_means.device)
    assigned = model.anchor_classes("conv1").to(class_means.device)
    if assigned.numel() != channels:
        raise ValueError("conv1 anchor assignment does not match channel count")
    total = class_means.sum(dim=0)
    assigned_mean = class_means[assigned, channel_indices]
    assigned_other = (total - assigned_mean) / max(1, classes - 1)
    assigned_score = (assigned_mean - assigned_other) / (
        assigned_mean.abs() + assigned_other.abs()
    ).clamp_min(1e-12)
    best_mean, best_class = class_means.max(dim=0)
    best_other = (total - best_mean) / max(1, classes - 1)
    best_score = (best_mean - best_other) / (
        best_mean.abs() + best_other.abs()
    ).clamp_min(1e-12)
    return {
        "mean_assigned_class_selectivity": float(assigned_score.mean()),
        "mean_best_class_selectivity": float(best_score.mean()),
        "assigned_class_alignment": float((best_class == assigned).float().mean()),
    }


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    runs = discover_runs(args)
    datasets = {}
    rows = []
    for key, (config_path, config, checkpoint) in sorted(runs.items()):
        dataset_name, model_name, method, seed = key
        if dataset_name not in datasets:
            datasets[dataset_name] = build_test_dataset(
                dataset_name,
                Path(config["data_root"]),
                Path(config["sign_root"]),
            )
        loader = DataLoader(
            datasets[dataset_name],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
        )
        model = build_model(model_name, dataset_name).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        metrics = selectivity_metrics(model, conv1_class_means(model, loader, device))
        rows.append(
            {
                "run": config_path.parent.name,
                "dataset": dataset_name,
                "model": model_name,
                "method": method,
                "model_seed": seed,
                "device": str(device),
                **metrics,
            }
        )
        print(
            f"{method:12s} seed={seed} "
            f"assigned={metrics['mean_assigned_class_selectivity']:.4f} "
            f"best={metrics['mean_best_class_selectivity']:.4f}"
        )
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["model"], row["method"])].append(row)
    summary = []
    metric_names = (
        "mean_assigned_class_selectivity",
        "mean_best_class_selectivity",
        "assigned_class_alignment",
    )
    for key, group in sorted(groups.items()):
        item = dict(zip(("dataset", "model", "method"), key))
        item["models"] = len(group)
        for metric in metric_names:
            values = [float(row[metric]) for row in group]
            item[f"mean_{metric}"] = statistics.mean(values)
            item[f"std_{metric}"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary.append(item)
    args.results_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.results_root / "selectivity_rows.csv", rows)
    write_csv(args.results_root / "selectivity_summary.csv", summary)
    config = vars(args).copy()
    config.update(
        {
            "output_root": str(args.output_root.resolve()),
            "results_root": str(args.results_root.resolve()),
            "device_resolved": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "checkpoints": len(runs),
        }
    )
    (args.results_root / "evaluation_config.json").write_text(
        json.dumps(config, indent=2, default=str), encoding="utf-8"
    )
    print(f"Results: {args.results_root.resolve()}")


if __name__ == "__main__":
    main()
