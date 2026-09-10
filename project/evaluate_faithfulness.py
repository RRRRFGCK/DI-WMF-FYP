import argparse
import csv
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from domain_mf.corruptions import from_pixel_space, to_pixel_space
from domain_mf.data import build_test_dataset
from domain_mf.faithfulness import (
    contribution_cosine,
    decompose_logits,
    intervened_logits,
    mask_jaccard,
    ordinal_rank_correlation,
    random_k_mask,
    target_contributions,
    topk_mask,
)
from domain_mf.models import build_model


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate exact evidence faithfulness, causal interventions and "
            "explanation stability for models with a linear concept bottleneck"
        )
    )
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    parser.add_argument(
        "--results-root", type=Path, default=ROOT / "faithfulness_results"
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["fashion", "cifar10", "sign"]
    )
    parser.add_argument("--models", nargs="+", default=["standard"])
    parser.add_argument("--methods", nargs="+", default=["random", "di_wmf"])
    parser.add_argument("--reference-method", default="random")
    parser.add_argument("--comparison-method", default="di_wmf")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--init-layers", type=int, default=3)
    parser.add_argument(
        "--fractions", nargs="+", type=float, default=[0.1, 0.25, 0.5]
    )
    parser.add_argument(
        "--severities", nargs="+", type=float, default=[0.1, 0.2, 0.3]
    )
    parser.add_argument("--noise-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--stability-fraction", type=float, default=0.1)
    parser.add_argument("--intervention-seed", type=int, default=31415)
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
        if not checkpoint.exists():
            continue
        if key in found:
            raise RuntimeError(f"Multiple completed runs match {key}")
        found[key] = (config_path, config, checkpoint)
    missing = sorted(expected - set(found))
    if missing:
        raise RuntimeError(f"Missing {len(missing)} requested runs: {missing[:5]}")
    return found


def make_loader(dataset, args, device):
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )


def _gather_class(logits, classes):
    return logits.gather(1, classes[:, None]).squeeze(1)


def _fraction_to_k(fraction, channels):
    if not 0 < fraction <= 1:
        raise ValueError("All fractions must be in (0, 1]")
    return min(channels, max(1, int(round(float(fraction) * channels))))


def evaluate_clean(
    model, loader, fractions, stability_fraction, intervention_seed, device
):
    channels = model.classifier.in_features
    ks = {fraction: _fraction_to_k(fraction, channels) for fraction in fractions}
    stability_k = _fraction_to_k(stability_fraction, channels)
    accumulators = {fraction: defaultdict(float) for fraction in fractions}
    references = defaultdict(list)
    random_generator = torch.Generator(device=device).manual_seed(intervention_seed)
    total = 0
    correct = 0
    max_completeness_error = 0.0
    try:
        anchor_classes = model.anchor_classes("conv2").to(device)
    except KeyError:
        anchor_classes = None
    class_owned_channels = bool(
        anchor_classes is not None
        and anchor_classes.numel() == channels
        and (anchor_classes >= 0).all()
        and (anchor_classes < model.spec.num_classes).all()
    )
    model.eval()
    with torch.inference_mode():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")
            decomposition = decompose_logits(model, inputs)
            completeness_error = (
                decomposition.logits - decomposition.reconstructed_logits
            ).abs().max()
            max_completeness_error = max(
                max_completeness_error, float(completeness_error)
            )
            predictions = decomposition.logits.argmax(dim=1)
            scores = target_contributions(decomposition.contributions, predictions)
            total += labels.numel()
            correct += int((predictions == labels).sum())
            references["predictions"].append(predictions.cpu())
            references["contributions"].append(scores.cpu())
            references["top_mask"].append(topk_mask(scores, stability_k).cpu())
            references["correct"].append((predictions == labels).cpu())

            original_target_logits = _gather_class(decomposition.logits, predictions)
            positive_total = scores.clamp_min(0).sum(dim=1)
            for fraction, k in ks.items():
                selected = topk_mask(scores, k)
                random_selected = random_k_mask(
                    labels.shape[0],
                    channels,
                    k,
                    generator=random_generator,
                    device=device,
                )
                sufficient_logits = intervened_logits(
                    model, decomposition.features, selected
                )
                deleted_logits = intervened_logits(
                    model, decomposition.features, ~selected
                )
                random_deleted_logits = intervened_logits(
                    model, decomposition.features, ~random_selected
                )
                sufficient_predictions = sufficient_logits.argmax(dim=1)
                deleted_predictions = deleted_logits.argmax(dim=1)
                random_deleted_predictions = random_deleted_logits.argmax(dim=1)
                selected_target = _gather_class(deleted_logits, predictions)
                random_target = _gather_class(random_deleted_logits, predictions)
                selected_positive = (
                    scores.clamp_min(0) * selected.to(scores.dtype)
                ).sum(dim=1)
                random_positive = (
                    scores.clamp_min(0) * random_selected.to(scores.dtype)
                ).sum(dim=1)
                positive_mass = torch.where(
                    positive_total > 0,
                    selected_positive / positive_total,
                    torch.zeros_like(positive_total),
                )
                random_positive_mass = torch.where(
                    positive_total > 0,
                    random_positive / positive_total,
                    torch.zeros_like(positive_total),
                )
                values = {
                    "positive_contribution_mass": positive_mass,
                    "random_positive_contribution_mass": random_positive_mass,
                    "sufficiency_agreement": (sufficient_predictions == predictions).float(),
                    "sufficiency_accuracy": (sufficient_predictions == labels).float(),
                    "deletion_flip_rate": (deleted_predictions != predictions).float(),
                    "deletion_accuracy": (deleted_predictions == labels).float(),
                    "random_deletion_flip_rate": (
                        random_deleted_predictions != predictions
                    ).float(),
                    "topk_logit_drop": original_target_logits - selected_target,
                    "random_logit_drop": original_target_logits - random_target,
                }
                if class_owned_channels:
                    values["class_slot_alignment"] = (
                        selected & (anchor_classes[None, :] == predictions[:, None])
                    ).sum(dim=1).float() / k
                values["causal_logit_advantage"] = (
                    values["topk_logit_drop"] - values["random_logit_drop"]
                )
                values["causal_mass_advantage"] = (
                    positive_mass - random_positive_mass
                )
                values["deletion_flip_advantage"] = (
                    values["deletion_flip_rate"]
                    - values["random_deletion_flip_rate"]
                )
                for name, value in values.items():
                    accumulators[fraction][name] += float(value.sum())

    rows = []
    for fraction, k in ks.items():
        row = {"fraction": fraction, "k": k, "samples": total}
        row.update(
            {name: value / total for name, value in accumulators[fraction].items()}
        )
        rows.append(row)
    reference = {name: torch.cat(parts) for name, parts in references.items()}
    common = {
        "clean_accuracy": correct / total,
        "max_completeness_error": max_completeness_error,
        "channels": channels,
    }
    return rows, reference, common


def evaluate_noise_stability(
    model,
    loader,
    reference,
    dataset_name,
    severity,
    noise_seed,
    stability_fraction,
    device,
):
    generator = torch.Generator(device=device).manual_seed(int(noise_seed))
    channels = model.classifier.in_features
    k = _fraction_to_k(stability_fraction, channels)
    sums = defaultdict(float)
    total = 0
    stable_total = 0
    correct_total = 0
    offset = 0
    model.eval()
    with torch.inference_mode():
        for inputs, labels in loader:
            batch_size = labels.shape[0]
            inputs = inputs.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")
            pixels = to_pixel_space(inputs, dataset_name)
            noise = torch.randn(
                pixels.shape,
                generator=generator,
                dtype=pixels.dtype,
                device=device,
            )
            noisy_inputs = from_pixel_space(
                (pixels + float(severity) * noise).clamp(0.0, 1.0), dataset_name
            )
            decomposition = decompose_logits(model, noisy_inputs)
            clean_predictions = reference["predictions"][
                offset : offset + batch_size
            ].to(device)
            clean_scores = reference["contributions"][
                offset : offset + batch_size
            ].to(device)
            clean_mask = reference["top_mask"][offset : offset + batch_size].to(device)
            clean_correct = reference["correct"][offset : offset + batch_size].to(device)
            noisy_predictions = decomposition.logits.argmax(dim=1)
            noisy_scores = target_contributions(
                decomposition.contributions, clean_predictions
            )
            noisy_mask = topk_mask(noisy_scores, k)
            jaccard = mask_jaccard(clean_mask, noisy_mask)
            cosine = contribution_cosine(clean_scores, noisy_scores)
            rank_correlation = ordinal_rank_correlation(clean_scores, noisy_scores)
            stable = noisy_predictions == clean_predictions
            noisy_correct = noisy_predictions == labels
            sums["prediction_consistency"] += float(stable.sum())
            sums["noisy_accuracy"] += float(noisy_correct.sum())
            sums["topk_jaccard"] += float(jaccard.sum())
            sums["contribution_cosine"] += float(cosine.sum())
            sums["rank_correlation"] += float(rank_correlation.sum())
            if stable.any():
                sums["stable_prediction_topk_jaccard"] += float(jaccard[stable].sum())
                stable_total += int(stable.sum())
            if clean_correct.any():
                sums["clean_correct_topk_jaccard"] += float(
                    jaccard[clean_correct].sum()
                )
                sums["clean_correct_retention"] += float(
                    (noisy_correct & clean_correct).sum()
                )
                correct_total += int(clean_correct.sum())
            total += batch_size
            offset += batch_size
    return {
        "severity": severity,
        "noise_seed": noise_seed,
        "stability_fraction": stability_fraction,
        "k": k,
        "samples": total,
        "prediction_consistency": sums["prediction_consistency"] / total,
        "noisy_accuracy": sums["noisy_accuracy"] / total,
        "clean_correct_retention": (
            sums["clean_correct_retention"] / correct_total
            if correct_total else 0.0
        ),
        "topk_jaccard": sums["topk_jaccard"] / total,
        "contribution_cosine": sums["contribution_cosine"] / total,
        "rank_correlation": sums["rank_correlation"] / total,
        "stable_prediction_topk_jaccard": (
            sums["stable_prediction_topk_jaccard"] / stable_total
            if stable_total else 0.0
        ),
        "clean_correct_topk_jaccard": (
            sums["clean_correct_topk_jaccard"] / correct_total
            if correct_total else 0.0
        ),
    }


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean(rows, key):
    return statistics.mean(float(row[key]) for row in rows)


def _std(rows, key):
    values = [float(row[key]) for row in rows]
    return statistics.stdev(values) if len(values) > 1 else 0.0


def summarise(rows, group_keys, metrics):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in group_keys)].append(row)
    summaries = []
    for key, group in sorted(groups.items()):
        summary = dict(zip(group_keys, key))
        summary["models"] = len(group)
        for metric in metrics:
            summary[f"mean_{metric}"] = _mean(group, metric)
            summary[f"std_{metric}"] = _std(group, metric)
        summaries.append(summary)
    return summaries


def average_noise_seeds(rows):
    group_keys = ["dataset", "model", "method", "model_seed", "severity"]
    metrics = [
        "prediction_consistency",
        "noisy_accuracy",
        "clean_correct_retention",
        "topk_jaccard",
        "contribution_cosine",
        "rank_correlation",
        "stable_prediction_topk_jaccard",
        "clean_correct_topk_jaccard",
    ]
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in group_keys)].append(row)
    averaged = []
    for key, group in sorted(groups.items()):
        item = dict(zip(group_keys, key))
        item["noise_evaluations"] = len(group)
        item.update({metric: _mean(group, metric) for metric in metrics})
        averaged.append(item)
    return averaged


def paired_differences(
    faithfulness_rows,
    stability_model_rows,
    reference_method="random",
    comparison_method="di_wmf",
):
    output = []
    faith_metrics = [
        "positive_contribution_mass",
        "random_positive_contribution_mass",
        "causal_mass_advantage",
        "sufficiency_agreement",
        "deletion_flip_rate",
        "random_deletion_flip_rate",
        "deletion_flip_advantage",
        "topk_logit_drop",
        "random_logit_drop",
        "causal_logit_advantage",
    ]
    if all("class_slot_alignment" in row for row in faithfulness_rows):
        faith_metrics.append("class_slot_alignment")
    stability_metrics = [
        "prediction_consistency",
        "noisy_accuracy",
        "clean_correct_retention",
        "topk_jaccard",
        "contribution_cosine",
        "rank_correlation",
    ]
    analyses = [
        ("faithfulness", faithfulness_rows, "fraction", faith_metrics),
        ("stability", stability_model_rows, "severity", stability_metrics),
    ]
    for analysis, rows, condition, metrics in analyses:
        indexed = {
            (
                row["dataset"], row["model"], row["method"],
                int(row["model_seed"]), row[condition],
            ): row
            for row in rows
        }
        datasets = sorted({row["dataset"] for row in rows})
        models = sorted({row["model"] for row in rows})
        conditions = sorted({row[condition] for row in rows}, key=float)
        seeds = sorted({int(row["model_seed"]) for row in rows})
        for dataset in datasets:
            for model in models:
                for condition_value in conditions:
                    pairs = []
                    for seed in seeds:
                        reference_row = indexed.get(
                            (dataset, model, reference_method, seed, condition_value)
                        )
                        comparison_row = indexed.get(
                            (dataset, model, comparison_method, seed, condition_value)
                        )
                        if reference_row is not None and comparison_row is not None:
                            pairs.append((reference_row, comparison_row))
                    if not pairs:
                        continue
                    for metric in metrics:
                        differences = [
                            float(comparison[metric]) - float(reference[metric])
                            for reference, comparison in pairs
                        ]
                        output.append(
                            {
                                "analysis": analysis,
                                "dataset": dataset,
                                "model": model,
                                "condition": condition,
                                "condition_value": condition_value,
                                "metric": metric,
                                "reference_method": reference_method,
                                "comparison_method": comparison_method,
                                "pairs": len(differences),
                                "mean_comparison_minus_reference": statistics.mean(
                                    differences
                                ),
                                "std_comparison_minus_reference": (
                                    statistics.stdev(differences)
                                    if len(differences) > 1 else 0.0
                                ),
                                "comparison_better_pairs": sum(
                                    value > 0 for value in differences
                                ),
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
    dataset_cache = {}
    faithfulness_rows = []
    stability_rows = []
    start = time.perf_counter()
    for key, (config_path, config, checkpoint) in sorted(runs.items()):
        dataset_name, model_name, method, model_seed = key
        if dataset_name not in dataset_cache:
            dataset_cache[dataset_name] = build_test_dataset(
                dataset_name,
                Path(config["data_root"]),
                Path(config["sign_root"]) if config.get("sign_root") else None,
            )
        loader = make_loader(dataset_cache[dataset_name], args, device)
        model = build_model(model_name, dataset_name).to(device)
        model.load_state_dict(
            torch.load(checkpoint, map_location=device, weights_only=True)
        )
        clean_rows, reference, clean_common = evaluate_clean(
            model,
            loader,
            args.fractions,
            args.stability_fraction,
            args.intervention_seed,
            device,
        )
        common = {
            "run": config_path.parent.name,
            "dataset": dataset_name,
            "model": model_name,
            "method": method,
            "model_seed": model_seed,
            "device": str(device),
            **clean_common,
        }
        faithfulness_rows.extend({**common, **row} for row in clean_rows)
        for severity in sorted(set(args.severities)):
            for noise_seed in args.noise_seeds:
                stability = evaluate_noise_stability(
                    model,
                    loader,
                    reference,
                    dataset_name,
                    severity,
                    noise_seed,
                    args.stability_fraction,
                    device,
                )
                stability_rows.append({**common, **stability})
        print(
            f"{dataset_name:8s} {model_name:12s} {method:8s} seed={model_seed} "
            f"clean={100 * clean_common['clean_accuracy']:.2f}% "
            f"completeness={clean_common['max_completeness_error']:.2e}"
        )

    excluded_faith = {
        "run", "dataset", "model", "method", "model_seed", "device", "clean_accuracy",
        "max_completeness_error", "channels", "fraction", "k", "samples",
    }
    common_faith_keys = set.intersection(
        *(set(row) for row in faithfulness_rows)
    )
    faith_metrics = sorted(common_faith_keys - excluded_faith)
    faith_summary = summarise(
        faithfulness_rows, ["dataset", "model", "method", "fraction"], faith_metrics
    )
    stability_model_rows = average_noise_seeds(stability_rows)
    excluded_stability = {
        "dataset", "model", "method", "model_seed", "severity", "noise_evaluations",
    }
    stability_metrics = [
        key for key in stability_model_rows[0] if key not in excluded_stability
    ]
    stability_summary = summarise(
        stability_model_rows, ["dataset", "model", "method", "severity"], stability_metrics
    )
    paired = paired_differences(
        faithfulness_rows,
        stability_model_rows,
        args.reference_method,
        args.comparison_method,
    )
    args.results_root.mkdir(parents=True, exist_ok=True)
    _write_csv(args.results_root / "faithfulness_model_metrics.csv", faithfulness_rows)
    _write_csv(args.results_root / "faithfulness_summary.csv", faith_summary)
    _write_csv(args.results_root / "stability_rows.csv", stability_rows)
    _write_csv(args.results_root / "stability_model_metrics.csv", stability_model_rows)
    _write_csv(args.results_root / "stability_summary.csv", stability_summary)
    _write_csv(args.results_root / "paired_differences.csv", paired)
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
    print(f"Evaluated {len(runs)} checkpoints in {elapsed:.1f}s on {device}")
    print(f"Results: {args.results_root.resolve()}")


if __name__ == "__main__":
    main()
