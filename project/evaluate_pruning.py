"""Evaluate post-training magnitude-pruning robustness of trained checkpoints.

This is a compressibility experiment, not a dense-inference speed benchmark:
zero-valued weights reduce the theoretical number of stored non-zero weights,
but an ordinary PyTorch dense convolution does not automatically become faster.
"""

import argparse
import copy
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils import prune
from torch.utils.data import DataLoader

from domain_mf.data import build_test_dataset
from domain_mf.models import build_model
from domain_mf.trainer import evaluate


T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate checkpoint compressibility")
    parser.add_argument("--output-roots", nargs="+", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, default=Path("pruning_results"))
    parser.add_argument("--datasets", nargs="+", default=["fashion", "cifar10", "sign"])
    parser.add_argument("--models", nargs="+", default=["standard"])
    parser.add_argument("--methods", nargs="+", default=["kaiming", "lowrank_wmf"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--init-layers", type=int, default=3)
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument(
        "--sparsities", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 0.9]
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--sign-root", type=Path, default=None)
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
    for root in args.output_roots:
        for config_path in sorted(root.glob("*/config.json")):
            config = json.loads(config_path.read_text(encoding="utf-8"))
            key = (
                config.get("dataset"), config.get("model"), config.get("init"),
                int(config.get("seed", -1)),
            )
            if key not in expected:
                continue
            if int(config.get("epochs", -1)) != args.epochs:
                continue
            if int(config.get("init_layers", -1)) != args.init_layers:
                continue
            if float(config.get("train_fraction", -1)) != args.train_fraction:
                continue
            checkpoint = config_path.parent / "checkpoint_best.pt"
            if checkpoint.exists():
                if key in found:
                    raise RuntimeError(f"Duplicate completed run for {key}")
                found[key] = (config, checkpoint)
    missing = sorted(expected - set(found))
    if missing:
        raise RuntimeError(f"Missing {len(missing)} requested runs: {missing[:5]}")
    return found


def prunable_modules(model):
    return [(module, "weight") for module in model.modules() if isinstance(module, (nn.Conv2d, nn.Linear))]


def apply_global_pruning(model, sparsity):
    parameters = prunable_modules(model)
    if sparsity > 0:
        prune.global_unstructured(
            parameters, pruning_method=prune.L1Unstructured, amount=float(sparsity)
        )
        for module, name in parameters:
            prune.remove(module, name)
    weights = [getattr(module, name) for module, name in parameters]
    total = sum(weight.numel() for weight in weights)
    nonzero = sum(int(torch.count_nonzero(weight)) for weight in weights)
    return total, nonzero


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["method"], row["sparsity"])].append(row)
    output = []
    for (dataset, method, sparsity), group in sorted(groups.items()):
        accuracies = [float(row["accuracy"]) for row in group]
        retentions = [float(row["accuracy_retention"]) for row in group]
        output.append(
            {
                "dataset": dataset,
                "method": method,
                "sparsity": sparsity,
                "runs": len(group),
                "mean_accuracy": statistics.mean(accuracies),
                "std_accuracy": statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0,
                "mean_accuracy_retention": statistics.mean(retentions),
                "std_accuracy_retention": statistics.stdev(retentions) if len(retentions) > 1 else 0.0,
                "mean_nonzero_prunable_parameters": statistics.mean(
                    int(row["nonzero_prunable_parameters"]) for row in group
                ),
            }
        )
    return output


def per_run_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["method"], row["seed"])].append(row)
    output = []
    for (dataset, method, seed), group in sorted(groups.items()):
        ordered = sorted(group, key=lambda row: float(row["sparsity"]))
        sparsities = [float(row["sparsity"]) for row in ordered]
        retentions = [float(row["accuracy_retention"]) for row in ordered]
        area = 0.0
        for left, right, y_left, y_right in zip(
            sparsities[:-1], sparsities[1:], retentions[:-1], retentions[1:]
        ):
            area += (right - left) * (y_left + y_right) / 2.0
        maximum = sparsities[-1]
        output.append(
            {
                "dataset": dataset,
                "method": method,
                "seed": seed,
                "normalised_pruning_auc": area / maximum if maximum else 1.0,
                "maximum_tested_sparsity_retaining_95pct": max(
                    (s for s, retention in zip(sparsities, retentions) if retention >= 0.95),
                    default=0.0,
                ),
            }
        )
    return output


def paired(rows, reference="kaiming", candidate="lowrank_wmf"):
    lookup = {
        (row["dataset"], row["method"], int(row["seed"]), float(row["sparsity"])): row
        for row in rows
    }
    datasets = sorted({row["dataset"] for row in rows})
    sparsities = sorted({float(row["sparsity"]) for row in rows})
    seeds = sorted({int(row["seed"]) for row in rows})
    output = []
    for dataset in datasets:
        for sparsity in sparsities:
            for metric in ("accuracy", "accuracy_retention"):
                differences = []
                for seed in seeds:
                    ref = lookup.get((dataset, reference, seed, sparsity))
                    cand = lookup.get((dataset, candidate, seed, sparsity))
                    if ref is not None and cand is not None:
                        differences.append(float(cand[metric]) - float(ref[metric]))
                if len(differences) < 2:
                    continue
                difference_mean = statistics.mean(differences)
                difference_std = statistics.stdev(differences)
                half_width = T95.get(len(differences) - 1, 1.96) * difference_std / math.sqrt(len(differences))
                output.append(
                    {
                        "dataset": dataset,
                        "sparsity": sparsity,
                        "metric": metric,
                        "reference": reference,
                        "candidate": candidate,
                        "paired_seeds": len(differences),
                        "mean_candidate_minus_reference": difference_mean,
                        "ci95_low": difference_mean - half_width,
                        "ci95_high": difference_mean + half_width,
                        "candidate_wins": sum(value > 0 for value in differences),
                    }
                )
    return output


def main():
    args = parse_args()
    if any(not 0 <= value < 1 for value in args.sparsities):
        raise ValueError("Every sparsity must be in [0, 1)")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    runs = discover_runs(args)
    dataset_cache = {}
    loader_cache = {}
    rows = []
    for (dataset_name, model_name, method, seed), (config, checkpoint) in sorted(runs.items()):
        if dataset_name not in dataset_cache:
            data_root = args.data_root or Path(config["data_root"])
            sign_root = args.sign_root or (Path(config["sign_root"]) if config.get("sign_root") else None)
            dataset_cache[dataset_name] = build_test_dataset(dataset_name, data_root, sign_root)
            loader_cache[dataset_name] = DataLoader(
                dataset_cache[dataset_name], batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=device.type == "cuda",
                persistent_workers=args.num_workers > 0,
            )
        base_model = build_model(model_name, dataset_name).to(device)
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        base_model.load_state_dict(state)
        clean = evaluate(base_model, loader_cache[dataset_name], device)["accuracy"]
        for sparsity in sorted(set(args.sparsities)):
            model = copy.deepcopy(base_model)
            total, nonzero = apply_global_pruning(model, sparsity)
            accuracy = evaluate(model, loader_cache[dataset_name], device)["accuracy"]
            rows.append(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "method": method,
                    "seed": seed,
                    "sparsity": sparsity,
                    "device_resolved": device.type,
                    "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                    "clean_accuracy": clean,
                    "accuracy": accuracy,
                    "accuracy_retention": accuracy / clean if clean else 0.0,
                    "prunable_parameters": total,
                    "nonzero_prunable_parameters": nonzero,
                    "measured_sparsity": 1.0 - nonzero / total,
                }
            )
            print(
                f"{dataset_name} {method} seed={seed} sparsity={sparsity:.2f}: "
                f"accuracy={accuracy:.2f}% retention={accuracy / clean:.3f}"
            )
    args.results_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.results_root / "pruning_rows.csv", rows)
    write_csv(args.results_root / "pruning_aggregate.csv", aggregate(rows))
    write_csv(args.results_root / "pruning_run_summary.csv", per_run_summary(rows))
    write_csv(args.results_root / "pruning_paired.csv", paired(rows))
    metadata = {
        "device_resolved": device.type,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "checkpoint_count": len(runs),
        "sparsities": sorted(set(args.sparsities)),
        "interpretation": "Unstructured sparsity measures compressibility; dense PyTorch kernels do not imply latency gains.",
    }
    (args.results_root / "config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote pruning results to {args.results_root.resolve()}")


if __name__ == "__main__":
    main()
