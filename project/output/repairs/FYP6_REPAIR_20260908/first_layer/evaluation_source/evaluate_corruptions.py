import argparse
import csv
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from domain_mf.corruptions import (
    CORRUPTIONS,
    apply_corruption_batch,
    from_pixel_space,
    to_pixel_space,
)
from domain_mf.data import build_test_dataset
from domain_mf.models import build_model


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate clean-trained checkpoints under deterministic corruptions"
    )
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    parser.add_argument(
        "--results-root", type=Path, default=ROOT / "corruption_results"
    )
    parser.add_argument("--datasets", nargs="+", default=["fashion", "sign"])
    parser.add_argument("--models", nargs="+", default=["standard"])
    parser.add_argument("--methods", nargs="+", default=["random", "di_wmf"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--init-layers", type=int, default=3)
    parser.add_argument("--corruption", choices=CORRUPTIONS, default=None)
    parser.add_argument("--corruptions", nargs="+", choices=CORRUPTIONS, default=None)
    parser.add_argument(
        "--severities", nargs="+", type=float, default=[0.05, 0.1, 0.2, 0.3]
    )
    parser.add_argument("--noise-seeds", nargs="+", type=int, default=[0, 1, 2])
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
        if config.get("reg_schedule") != "none":
            continue
        if config.get("no_logit_calibration", True):
            continue
        checkpoint = config_path.parent / "checkpoint_best.pt"
        metrics = config_path.parent / "final_metrics.json"
        if not checkpoint.exists() or not metrics.exists():
            continue
        if key in found:
            raise RuntimeError(
                f"Multiple completed runs match {key}: {found[key][0].parent} and "
                f"{config_path.parent}"
            )
        found[key] = (config_path, config, checkpoint)
    missing = sorted(expected - set(found))
    if missing:
        raise RuntimeError(f"Missing {len(missing)} requested runs: {missing[:5]}")
    return found


def predict(
    model,
    dataset,
    batch_size,
    num_workers,
    device,
    dataset_name=None,
    corruption="gaussian",
    severity=0.0,
    noise_seed=0,
):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    predictions = []
    targets = []
    generator = None
    if severity > 0:
        generator = torch.Generator(device=device).manual_seed(int(noise_seed))
    model.eval()
    with torch.inference_mode():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=device.type == "cuda")
            if severity > 0:
                pixels = to_pixel_space(inputs, dataset_name)
                pixels = apply_corruption_batch(
                    pixels, corruption, float(severity), generator
                )
                inputs = from_pixel_space(pixels, dataset_name)
            predictions.append(model(inputs).argmax(1).cpu())
            targets.append(labels.cpu())
    predictions = torch.cat(predictions)
    targets = torch.cat(targets)
    accuracy = 100.0 * float((predictions == targets).float().mean())
    return predictions, targets, accuracy


def materialize_dataset(dataset, batch_size, num_workers, pin_memory=False):
    """Apply deterministic dataset transforms once before repeated evaluation."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    inputs = []
    labels = []
    for batch_inputs, batch_labels in loader:
        inputs.append(batch_inputs)
        labels.append(batch_labels)
    return TensorDataset(torch.cat(inputs), torch.cat(labels))


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean(rows, key):
    return statistics.mean(float(row[key]) for row in rows)


def _std(rows, key):
    values = [float(row[key]) for row in rows]
    return statistics.stdev(values) if len(values) > 1 else 0.0


def aggregate_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        key = (
            row["dataset"], row["model"], row["method"], row["corruption"],
            row["severity"],
        )
        grouped[key].append(row)
    aggregates = []
    metrics = [
        "corrupted_accuracy", "accuracy_drop", "retention", "consistency"
    ]
    for key, group in sorted(grouped.items()):
        aggregate = dict(
            zip(("dataset", "model", "method", "corruption", "severity"), key)
        )
        aggregate["evaluations"] = len(group)
        for metric in metrics:
            aggregate[f"mean_{metric}"] = _mean(group, metric)
            aggregate[f"std_{metric}"] = _std(group, metric)
        aggregates.append(aggregate)
    return aggregates


def summarise_models(rows, severities):
    per_model = defaultdict(list)
    for row in rows:
        per_model[
            (
                row["dataset"], row["model"], row["method"],
                row["model_seed"], row["corruption"],
            )
        ].append(row)
    model_rows = []
    ordered_severities = [0.0] + sorted(set(float(value) for value in severities))
    for key, group in sorted(per_model.items()):
        by_severity = defaultdict(list)
        for row in group:
            by_severity[float(row["severity"])].append(row)
        mean_accuracy = {
            severity: _mean(by_severity[severity], "corrupted_accuracy")
            for severity in ordered_severities
        }
        maximum = ordered_severities[-1]
        auc = 0.0
        for left, right in zip(ordered_severities, ordered_severities[1:]):
            auc += (right - left) * (mean_accuracy[left] + mean_accuracy[right]) / 2
        auc /= maximum
        noisy = [row for row in group if float(row["severity"]) > 0]
        model_rows.append(
            {
                "dataset": key[0],
                "model": key[1],
                "method": key[2],
                "model_seed": key[3],
                "corruption": key[4],
                "normalised_accuracy_auc": auc,
                "mean_accuracy_drop": _mean(noisy, "accuracy_drop"),
                "mean_retention": _mean(noisy, "retention"),
                "mean_consistency": _mean(noisy, "consistency"),
                "worst_accuracy": mean_accuracy[maximum],
            }
        )
    summary_groups = defaultdict(list)
    for row in model_rows:
        summary_groups[
            (row["dataset"], row["model"], row["method"], row["corruption"])
        ].append(row)
    summaries = []
    metrics = [
        "normalised_accuracy_auc", "mean_accuracy_drop", "mean_retention",
        "mean_consistency", "worst_accuracy",
    ]
    for key, group in sorted(summary_groups.items()):
        summary = dict(zip(("dataset", "model", "method", "corruption"), key))
        summary["models"] = len(group)
        for metric in metrics:
            summary[f"mean_{metric}"] = _mean(group, metric)
            summary[f"std_{metric}"] = _std(group, metric)
        summaries.append(summary)
    return model_rows, summaries


def main():
    args = parse_args()
    corruptions = args.corruptions or (
        [args.corruption] if args.corruption is not None else ["gaussian"]
    )
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    runs = discover_runs(args)
    dataset_cache = {}
    rows = []
    start = time.perf_counter()
    for key, (config_path, config, checkpoint) in sorted(runs.items()):
        dataset_name, model_name, method, model_seed = key
        if dataset_name not in dataset_cache:
            source_dataset = build_test_dataset(
                dataset_name,
                Path(config["data_root"]),
                Path(config["sign_root"]),
            )
            dataset_cache[dataset_name] = materialize_dataset(
                source_dataset,
                args.batch_size,
                args.num_workers,
                pin_memory=device.type == "cuda",
            )
        clean_dataset = dataset_cache[dataset_name]
        model = build_model(model_name, dataset_name).to(device)
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state)
        clean_predictions, targets, clean_accuracy = predict(
            model, clean_dataset, args.batch_size, args.num_workers, device
        )
        common_base = {
            "run": config_path.parent.name,
            "dataset": dataset_name,
            "model": model_name,
            "method": method,
            "model_seed": model_seed,
            "clean_accuracy": clean_accuracy,
            "device": str(device),
        }
        for corruption in corruptions:
            common = {**common_base, "corruption": corruption}
            rows.append(
                {
                    **common,
                    "noise_seed": -1,
                    "severity": 0.0,
                    "corrupted_accuracy": clean_accuracy,
                    "accuracy_drop": 0.0,
                    "retention": 1.0,
                    "consistency": 1.0,
                }
            )
            for severity in sorted(set(args.severities)):
                for noise_seed in args.noise_seeds:
                    predictions, corrupted_targets, accuracy = predict(
                        model,
                        clean_dataset,
                        args.batch_size,
                        args.num_workers,
                        device,
                        dataset_name=dataset_name,
                        corruption=corruption,
                        severity=severity,
                        noise_seed=noise_seed,
                    )
                    if not torch.equal(targets, corrupted_targets):
                        raise RuntimeError("Corruption changed test-set ordering")
                    rows.append(
                        {
                            **common,
                            "noise_seed": noise_seed,
                            "severity": severity,
                            "corrupted_accuracy": accuracy,
                            "accuracy_drop": clean_accuracy - accuracy,
                            "retention": accuracy / clean_accuracy if clean_accuracy else 0.0,
                            "consistency": float(
                                (predictions == clean_predictions).float().mean()
                            ),
                        }
                    )
        print(
            f"{dataset_name:8s} {model_name:8s} {method:8s} seed={model_seed} "
            f"clean={clean_accuracy:.2f}%"
        )
    aggregates = aggregate_rows(rows)
    model_rows, summaries = summarise_models(rows, args.severities)
    args.results_root.mkdir(parents=True, exist_ok=True)
    _write_csv(args.results_root / "corruption_rows.csv", rows)
    _write_csv(args.results_root / "corruption_aggregate.csv", aggregates)
    _write_csv(args.results_root / "corruption_model_metrics.csv", model_rows)
    _write_csv(args.results_root / "corruption_summary.csv", summaries)
    elapsed = time.perf_counter() - start
    evaluation_config = vars(args).copy()
    evaluation_config.update(
        {
            "output_root": str(args.output_root.resolve()),
            "results_root": str(args.results_root.resolve()),
            "device_resolved": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
            ),
            "torch_version": torch.__version__,
            "checkpoints": len(runs),
            "evaluation_seconds": elapsed,
        }
    )
    (args.results_root / "evaluation_config.json").write_text(
        json.dumps(evaluation_config, indent=2, default=str), encoding="utf-8"
    )
    print(f"Evaluated {len(runs)} checkpoints in {elapsed:.1f}s")
    print(f"Results: {args.results_root.resolve()}")


if __name__ == "__main__":
    main()
