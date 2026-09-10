"""Evaluate real channel pruning and deployment-oriented efficiency metrics.

Channels are physically removed from both convolutions and the classifier.
This differs from ``evaluate_pruning.py``, which measures unstructured weight
sparsity without changing dense tensor shapes. GPU energy is an explicitly
labelled board-power estimate sampled through ``nvidia-smi`` and is inherently
noisier than latency, parameter and FLOP measurements.
"""

import argparse
import copy
import csv
import json
import math
import statistics
import subprocess
import threading
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from domain_mf.data import build_loaders, build_test_dataset
from domain_mf.models import SPECS, StandardCNN, build_model


T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}


class StructuredCNN(nn.Module):
    """StandardCNN forward path with arbitrary physically pruned widths."""

    def __init__(self, spec, stem_channels, body_channels):
        super().__init__()
        self.spec = spec
        self.conv1 = nn.Conv2d(
            spec.input_channels, stem_channels, kernel_size=5, padding=2
        )
        self.conv2 = nn.Conv2d(
            stem_channels, body_channels, kernel_size=3, padding=1
        )
        self.classifier = nn.Linear(body_channels, spec.num_classes)

    def forward(self, inputs):
        features = F.max_pool2d(F.relu(self.conv1(inputs)), 2)
        features = F.max_pool2d(F.relu(self.conv2(features)), 2)
        pooled = F.adaptive_avg_pool2d(features, 1).flatten(1)
        return self.classifier(pooled)


def parse_args():
    parser = argparse.ArgumentParser(description="Structured pruning efficiency")
    parser.add_argument("--output-roots", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--results-root", type=Path, default=Path("structured_efficiency_results")
    )
    parser.add_argument("--datasets", nargs="+", default=["fashion", "cifar10", "sign"])
    parser.add_argument("--methods", nargs="+", default=["kaiming", "lowrank_wmf"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--sparsities", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 64])
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--latency-iterations", type=int, default=100)
    parser.add_argument("--energy-seconds", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--finetune-epochs", type=int, default=0)
    parser.add_argument("--finetune-learning-rate", type=float, default=1e-4)
    parser.add_argument("--finetune-batch-size", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--sign-root", type=Path, default=None)
    return parser.parse_args()


def discover_runs(args):
    expected = {
        (dataset, method, seed)
        for dataset in args.datasets
        for method in args.methods
        for seed in args.seeds
    }
    found = {}
    for root in args.output_roots:
        for config_path in root.glob("*/config.json"):
            config = json.loads(config_path.read_text(encoding="utf-8"))
            key = (config.get("dataset"), config.get("init"), int(config.get("seed", -1)))
            if key not in expected or config.get("model") != "standard":
                continue
            if int(config.get("epochs", -1)) != 20:
                continue
            if float(config.get("train_fraction", -1)) != 1.0:
                continue
            checkpoint = config_path.parent / "checkpoint_best.pt"
            if checkpoint.exists():
                if key in found:
                    raise RuntimeError(f"Duplicate checkpoint for {key}")
                found[key] = (config, checkpoint)
    missing = sorted(expected - set(found))
    if missing:
        raise RuntimeError(f"Missing {len(missing)} checkpoints: {missing[:5]}")
    return found


def _balanced_indices(weight, groups, keep_per_group):
    channels_per_group = weight.shape[0] // groups
    selected = []
    importance = weight.detach().abs().flatten(1).sum(1)
    for group in range(groups):
        start = group * channels_per_group
        scores = importance[start : start + channels_per_group]
        local = scores.topk(keep_per_group, largest=True, sorted=True).indices
        selected.extend((local + start).tolist())
    return torch.tensor(selected, dtype=torch.long, device=weight.device)


def physically_prune(model: StandardCNN, sparsity: float):
    if not 0 <= sparsity < 1:
        raise ValueError("sparsity must be in [0, 1)")
    stem_per_class = model.stem_templates_per_class
    body_per_class = model.body_templates_per_class
    keep_stem = max(1, round(stem_per_class * (1.0 - sparsity)))
    keep_body = max(1, round(body_per_class * (1.0 - sparsity)))
    stem_indices = _balanced_indices(
        model.conv1.weight, model.spec.num_classes, keep_stem
    )
    body_indices = _balanced_indices(
        model.conv2.weight, model.spec.num_classes, keep_body
    )
    pruned = StructuredCNN(
        model.spec, stem_indices.numel(), body_indices.numel()
    ).to(model.conv1.weight.device)
    with torch.no_grad():
        pruned.conv1.weight.copy_(model.conv1.weight[stem_indices])
        pruned.conv1.bias.copy_(model.conv1.bias[stem_indices])
        pruned.conv2.weight.copy_(
            model.conv2.weight[body_indices][:, stem_indices]
        )
        pruned.conv2.bias.copy_(model.conv2.bias[body_indices])
        pruned.classifier.weight.copy_(model.classifier.weight[:, body_indices])
        pruned.classifier.bias.copy_(model.classifier.bias)
    return pruned, int(stem_indices.numel()), int(body_indices.numel())


def accuracy(model, loader, device):
    correct = total = 0
    model.eval()
    with torch.inference_mode():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")
            correct += int((model(inputs).argmax(1) == labels).sum())
            total += labels.numel()
    return 100.0 * correct / total


def finetune_pruned_model(
    model, train_loader, val_loader, device, epochs, learning_rate
):
    """Recover a physically pruned model using training/validation data only."""

    if epochs <= 0:
        return {"seconds": 0.0, "best_val_accuracy": float("nan"), "best_epoch": 0}
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_state = copy.deepcopy(model.state_dict())
    best_val = accuracy(model, val_loader, device)
    best_epoch = 0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
        val_accuracy = accuracy(model, val_loader, device)
        if val_accuracy > best_val:
            best_val = val_accuracy
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return {
        "seconds": time.perf_counter() - started,
        "best_val_accuracy": best_val,
        "best_epoch": best_epoch,
    }


def model_flops(model, image_size):
    stem = model.conv1.out_channels
    body = model.conv2.out_channels
    inputs = model.conv1.in_channels
    classes = model.classifier.out_features
    conv1_macs = image_size * image_size * stem * inputs * 5 * 5
    half = image_size // 2
    conv2_macs = half * half * body * stem * 3 * 3
    linear_macs = body * classes
    macs = conv1_macs + conv2_macs + linear_macs
    return int(macs), int(2 * macs)


def _power_draw():
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    output = subprocess.check_output(
        [
            "nvidia-smi", "--query-gpu=power.draw",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        creationflags=flags,
        timeout=2,
    )
    return float(output.splitlines()[0].strip())


def _sample_power(stop, samples):
    while not stop.is_set():
        try:
            samples.append(_power_draw())
        except (OSError, ValueError, subprocess.SubprocessError):
            return
        stop.wait(0.05)


def benchmark(model, dataset, batch_size, device, warmup, iterations, energy_seconds):
    spec = SPECS[dataset]
    inputs = torch.randn(
        batch_size, spec.input_channels, spec.image_size, spec.image_size,
        device=device,
    )
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            model(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            baseline_memory = torch.cuda.memory_allocated(device)
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
            for start, end in zip(starts, ends):
                start.record()
                model(inputs)
                end.record()
            torch.cuda.synchronize(device)
            times_ms = [start.elapsed_time(end) for start, end in zip(starts, ends)]
            peak_bytes = max(
                0, torch.cuda.max_memory_allocated(device) - baseline_memory
            )
        else:
            times_ms = []
            peak_bytes = 0
            for _ in range(iterations):
                start = time.perf_counter()
                model(inputs)
                times_ms.append(1000.0 * (time.perf_counter() - start))

        idle_samples = []
        if device.type == "cuda":
            for _ in range(3):
                try:
                    idle_samples.append(_power_draw())
                except (OSError, ValueError, subprocess.SubprocessError):
                    break
        power_samples = []
        energy_iterations = 0
        energy_elapsed = 0.0
        if device.type == "cuda" and idle_samples:
            stop = threading.Event()
            sampler = threading.Thread(
                target=_sample_power, args=(stop, power_samples), daemon=True
            )
            sampler.start()
            start = time.perf_counter()
            while time.perf_counter() - start < energy_seconds:
                model(inputs)
                energy_iterations += 1
            torch.cuda.synchronize(device)
            energy_elapsed = time.perf_counter() - start
            stop.set()
            sampler.join(timeout=2)

    idle_power = statistics.mean(idle_samples) if idle_samples else float("nan")
    active_power = statistics.mean(power_samples) if power_samples else float("nan")
    gross_joules_per_batch = (
        active_power * energy_elapsed / energy_iterations
        if power_samples and energy_iterations
        else float("nan")
    )
    net_joules_per_batch = (
        max(0.0, active_power - idle_power) * energy_elapsed / energy_iterations
        if power_samples and energy_iterations
        else float("nan")
    )
    return {
        "latency_mean_ms": statistics.mean(times_ms),
        "latency_median_ms": statistics.median(times_ms),
        "throughput_images_per_second": 1000.0 * batch_size / statistics.mean(times_ms),
        "peak_incremental_memory_bytes": peak_bytes,
        "idle_power_watts": idle_power,
        "active_power_watts": active_power,
        "gross_joules_per_batch": gross_joules_per_batch,
        "net_joules_per_batch": net_joules_per_batch,
        "power_samples": len(power_samples),
    }


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows, group_fields, metrics):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in group_fields)].append(row)
    output = []
    for key, group in sorted(grouped.items()):
        result = dict(zip(group_fields, key))
        result["runs"] = len(group)
        for metric in metrics:
            values = [float(row[metric]) for row in group]
            finite = [value for value in values if math.isfinite(value)]
            result[f"mean_{metric}"] = statistics.mean(finite) if finite else ""
            result[f"std_{metric}"] = statistics.stdev(finite) if len(finite) > 1 else 0.0
        output.append(result)
    return output


def paired_retention(rows):
    indexed = {
        (row["dataset"], row["method"], row["seed"], row["sparsity"]): row
        for row in rows
    }
    output = []
    datasets = sorted({row["dataset"] for row in rows})
    sparsities = sorted({row["sparsity"] for row in rows})
    seeds = sorted({row["seed"] for row in rows})
    for dataset in datasets:
        for sparsity in sparsities:
            differences = []
            for seed in seeds:
                if (
                    (dataset, "kaiming", seed, sparsity) not in indexed
                    or (dataset, "lowrank_wmf", seed, sparsity) not in indexed
                ):
                    continue
                reference = indexed[(dataset, "kaiming", seed, sparsity)]
                candidate = indexed[(dataset, "lowrank_wmf", seed, sparsity)]
                differences.append(
                    float(candidate["accuracy_retention"])
                    - float(reference["accuracy_retention"])
                )
            if not differences:
                continue
            mean = statistics.mean(differences)
            std = statistics.stdev(differences) if len(differences) > 1 else 0.0
            half = (
                T95[len(differences) - 1] * std / math.sqrt(len(differences))
                if len(differences) > 1
                else 0.0
            )
            output.append(
                {
                    "dataset": dataset,
                    "sparsity": sparsity,
                    "paired_seeds": len(differences),
                    "mean_lowrank_minus_kaiming_retention": mean,
                    "ci95_low": mean - half,
                    "ci95_high": mean + half,
                    "lowrank_wins": sum(value > 0 for value in differences),
                }
            )
    return output


def main():
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    runs = discover_runs(args)
    data_root = args.data_root
    sign_root = args.sign_root
    if data_root is None:
        data_root = Path(next(iter(runs.values()))[0]["data_root"])
    if sign_root is None:
        sign_root = Path(next(iter(runs.values()))[0]["sign_root"])
    loaders = {}
    for dataset in args.datasets:
        test_dataset = build_test_dataset(dataset, data_root, sign_root)
        loaders[dataset] = DataLoader(
            test_dataset,
            batch_size=64 if dataset == "sign" else args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

    accuracy_rows = []
    benchmark_rows = []
    finetune_loaders = {}
    for (dataset, method, seed), (config, checkpoint) in sorted(runs.items()):
        original = build_model("standard", dataset).to(device)
        original.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        clean_accuracy = accuracy(original, loaders[dataset], device)
        full_parameters = sum(parameter.numel() for parameter in original.parameters())
        full_macs, full_flops = model_flops(original, SPECS[dataset].image_size)
        for sparsity in args.sparsities:
            model, stem_channels, body_channels = physically_prune(original, sparsity)
            model.eval()
            pre_finetune_accuracy = accuracy(model, loaders[dataset], device)
            finetune_report = {
                "seconds": 0.0,
                "best_val_accuracy": float("nan"),
                "best_epoch": 0,
            }
            if args.finetune_epochs > 0 and sparsity > 0:
                loader_key = (dataset, seed)
                if loader_key not in finetune_loaders:
                    train_loader, _, val_loader, _ = build_loaders(
                        dataset,
                        data_root,
                        sign_root,
                        args.finetune_batch_size or args.batch_size,
                        seed,
                        float(config.get("train_fraction", 1.0)),
                        float(config.get("val_fraction", 0.1)),
                        args.num_workers,
                        pin_memory=device.type == "cuda",
                        augmentation=config.get("augmentation") or "none",
                    )
                    finetune_loaders[loader_key] = (train_loader, val_loader)
                train_loader, val_loader = finetune_loaders[loader_key]
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                finetune_report = finetune_pruned_model(
                    model,
                    train_loader,
                    val_loader,
                    device,
                    args.finetune_epochs,
                    args.finetune_learning_rate,
                )
            test_accuracy = accuracy(model, loaders[dataset], device)
            parameters = sum(parameter.numel() for parameter in model.parameters())
            parameter_bytes = sum(
                parameter.numel() * parameter.element_size()
                for parameter in model.parameters()
            )
            macs, flops = model_flops(model, SPECS[dataset].image_size)
            accuracy_rows.append(
                {
                    "dataset": dataset, "method": method, "seed": seed,
                    "sparsity": sparsity, "stem_channels": stem_channels,
                    "body_channels": body_channels, "clean_accuracy": clean_accuracy,
                    "pre_finetune_accuracy": pre_finetune_accuracy,
                    "pruned_accuracy": test_accuracy,
                    "accuracy_retention": test_accuracy / max(clean_accuracy, 1e-12),
                    "finetune_epochs": args.finetune_epochs if sparsity > 0 else 0,
                    "finetune_best_epoch": finetune_report["best_epoch"],
                    "finetune_best_val_accuracy": finetune_report["best_val_accuracy"],
                    "finetune_seconds": finetune_report["seconds"],
                    "parameter_count": parameters,
                    "parameter_reduction": 1.0 - parameters / full_parameters,
                    "macs": macs, "flops": flops,
                    "flop_reduction": 1.0 - flops / full_flops,
                }
            )
            for batch_size in args.batch_sizes:
                metrics = benchmark(
                    model, dataset, batch_size, device, args.warmup,
                    args.latency_iterations, args.energy_seconds,
                )
                benchmark_rows.append(
                    {
                        "dataset": dataset, "method": method, "seed": seed,
                        "sparsity": sparsity, "batch_size": batch_size,
                        "parameter_count": parameters,
                        "parameter_bytes": parameter_bytes,
                        "estimated_model_plus_peak_bytes": (
                            parameter_bytes + metrics["peak_incremental_memory_bytes"]
                        ),
                        "flops": flops,
                        **metrics,
                    }
                )
            print(f"Completed {dataset}/{method}/seed{seed}/sparsity{sparsity}")

    write_csv(args.results_root / "structured_accuracy_rows.csv", accuracy_rows)
    write_csv(args.results_root / "structured_benchmark_rows.csv", benchmark_rows)
    write_csv(
        args.results_root / "structured_accuracy_aggregate.csv",
        aggregate(
            accuracy_rows, ["dataset", "method", "sparsity"],
            ["clean_accuracy", "pruned_accuracy", "accuracy_retention",
             "pre_finetune_accuracy", "finetune_seconds",
             "parameter_count", "parameter_reduction", "flops", "flop_reduction"],
        ),
    )
    write_csv(
        args.results_root / "structured_benchmark_aggregate.csv",
        aggregate(
            benchmark_rows, ["dataset", "method", "sparsity", "batch_size"],
            ["latency_mean_ms", "latency_median_ms", "throughput_images_per_second",
             "parameter_bytes", "estimated_model_plus_peak_bytes",
             "peak_incremental_memory_bytes", "idle_power_watts", "active_power_watts",
             "gross_joules_per_batch", "net_joules_per_batch"],
        ),
    )
    write_csv(args.results_root / "structured_retention_paired.csv", paired_retention(accuracy_rows))
    config = vars(args).copy()
    config.update(
        {
            "output_roots": [str(path.resolve()) for path in args.output_roots],
            "results_root": str(args.results_root.resolve()),
            "device_resolved": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "energy_caveat": "nvidia-smi board-power estimate; affected by concurrent GPU activity",
        }
    )
    args.results_root.mkdir(parents=True, exist_ok=True)
    (args.results_root / "evaluation_config.json").write_text(
        json.dumps(config, indent=2, default=str), encoding="utf-8"
    )
    print(f"Wrote structured efficiency results to {args.results_root.resolve()}")


if __name__ == "__main__":
    main()
