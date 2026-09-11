"""Quantify template semantics and compare spatial explanation methods.

For each clean-trained StandardCNN checkpoint this script measures whether the
highest-activating examples of each second-layer template share a class. It
also compares exact template CAM, Grad-CAM, Integrated Gradients and a matched
random mask using pixel-deletion interventions. Representative activation
grids are saved for direct qualitative inspection.
"""

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from domain_mf.corruptions import to_pixel_space
from domain_mf.data import build_test_dataset
from domain_mf.faithfulness import decompose_logits
from domain_mf.models import build_model


T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776,
    5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate template semantics and explanation baselines"
    )
    parser.add_argument("--output-roots", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--results-root", type=Path, default=Path("template_semantics_results")
    )
    parser.add_argument("--datasets", nargs="+", default=["fashion", "cifar10", "sign"])
    parser.add_argument("--models", nargs="+", default=["standard"])
    parser.add_argument("--methods", nargs="+", default=["kaiming", "lowrank_wmf"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--init-layers", type=int, default=3)
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--max-semantic-samples", type=int, default=500)
    parser.add_argument("--explanation-samples", type=int, default=100)
    parser.add_argument("--top-examples", type=int, default=5)
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.1, 0.2, 0.3])
    parser.add_argument("--ig-steps", type=int, default=8)
    parser.add_argument("--visualisation-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
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
                    raise RuntimeError(f"Duplicate run for {key}")
                found[key] = (config, checkpoint)
    missing = sorted(expected - set(found))
    if missing:
        raise RuntimeError(f"Missing {len(missing)} requested runs: {missing[:5]}")
    return found


def dataset_targets(dataset):
    targets = getattr(dataset, "targets", None)
    if targets is None:
        targets = [label for _, label in dataset.samples]
    if isinstance(targets, torch.Tensor):
        return targets.tolist()
    return list(targets)


def stratified_subset(dataset, maximum, seed=0):
    targets = np.asarray(dataset_targets(dataset))
    generator = np.random.default_rng(seed)
    classes = np.unique(targets)
    per_class = max(1, math.ceil(maximum / len(classes)))
    indices = []
    for class_id in classes:
        class_indices = np.flatnonzero(targets == class_id)
        generator.shuffle(class_indices)
        indices.extend(class_indices[:per_class].tolist())
    generator.shuffle(indices)
    return Subset(dataset, indices[: min(maximum, len(indices))])


def make_loader(dataset, maximum, args, device):
    subset = stratified_subset(dataset, maximum)
    return DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )


def collect_features(model, loader, device):
    inputs = []
    labels = []
    features = []
    model.eval()
    with torch.inference_mode():
        for batch, target in loader:
            batch = batch.to(device, non_blocking=device.type == "cuda")
            inputs.append(batch.cpu())
            labels.append(target.cpu())
            features.append(model.second_layer_features(batch).cpu())
    return torch.cat(inputs), torch.cat(labels), torch.cat(features)


def semantic_rows(model, labels, features, top_examples):
    scores = features.amax(dim=(2, 3))
    channel_classes = model.anchor_classes("conv2").cpu()
    rows = []
    for channel in range(scores.shape[1]):
        k = min(top_examples, scores.shape[0])
        indices = scores[:, channel].topk(k).indices
        top_labels = labels[indices]
        counts = torch.bincount(top_labels, minlength=model.spec.num_classes).float()
        probabilities = counts / counts.sum().clamp_min(1.0)
        nonzero = probabilities > 0
        entropy = -(
            probabilities[nonzero] * probabilities[nonzero].log()
        ).sum() / math.log(model.spec.num_classes)
        majority_count, majority_class = counts.max(dim=0)
        anchor_class = int(channel_classes[channel])
        rows.append(
            {
                "channel": channel,
                "anchor_class": anchor_class,
                "majority_class": int(majority_class),
                "top_label_purity": float(majority_count / k),
                "semantic_purity": float(1.0 - entropy),
                "anchor_alignment": float((top_labels == anchor_class).float().mean()),
                "mean_top_activation": float(scores[indices, channel].mean()),
                "top_indices": indices.tolist(),
            }
        )
    return rows


def normalise_heatmap(heatmap):
    flat = heatmap.flatten(1)
    minimum = flat.amin(dim=1, keepdim=True)
    maximum = flat.amax(dim=1, keepdim=True)
    return ((flat - minimum) / (maximum - minimum).clamp_min(1e-12)).view_as(heatmap)


def exact_and_gradcam(model, inputs, targets):
    features = model.second_layer_features(inputs)
    pooled = F.adaptive_avg_pool2d(features, 1).flatten(1)
    logits = model.classifier(pooled)
    weights = model.classifier.weight.index_select(0, targets)
    exact = F.relu((features * weights[:, :, None, None]).sum(dim=1))

    detached = features.detach().requires_grad_(True)
    detached_logits = model.classifier(F.adaptive_avg_pool2d(detached, 1).flatten(1))
    selected = detached_logits.gather(1, targets[:, None]).sum()
    gradients = torch.autograd.grad(selected, detached)[0]
    grad_weights = gradients.mean(dim=(2, 3), keepdim=True)
    gradcam = F.relu((detached * grad_weights).sum(dim=1))
    return logits.detach(), normalise_heatmap(exact.detach()), normalise_heatmap(gradcam.detach())


def integrated_gradients(model, inputs, targets, steps):
    baseline = torch.zeros_like(inputs)
    gradient_sum = torch.zeros_like(inputs)
    for step in range(1, steps + 1):
        alpha = step / steps
        interpolated = (baseline + alpha * (inputs - baseline)).detach().requires_grad_(True)
        logits = model(interpolated)
        selected = logits.gather(1, targets[:, None]).sum()
        gradient_sum += torch.autograd.grad(selected, interpolated)[0].detach()
    attribution = (inputs - baseline) * gradient_sum / steps
    return normalise_heatmap(attribution.abs().sum(dim=1))


def delete_top_pixels(inputs, heatmap, fraction):
    batch, _, height, width = inputs.shape
    upsampled = F.interpolate(
        heatmap[:, None], size=(height, width), mode="bilinear", align_corners=False
    ).squeeze(1)
    k = max(1, int(round(fraction * height * width)))
    indices = upsampled.flatten(1).topk(k, dim=1).indices
    mask = torch.zeros((batch, height * width), dtype=torch.bool, device=inputs.device)
    mask.scatter_(1, indices, True)
    mask = mask.view(batch, 1, height, width)
    return inputs.masked_fill(mask, 0.0)


def explanation_rows(model, loader, fractions, ig_steps, random_seed, device):
    accumulators = {
        (explainer, fraction): defaultdict(float)
        for explainer in ("exact_template_cam", "gradcam", "integrated_gradients", "random")
        for fraction in fractions
    }
    total = 0
    clean_correct_total = 0
    completeness_error = 0.0
    generator = torch.Generator(device=device).manual_seed(random_seed)
    model.eval()
    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")
        with torch.no_grad():
            decomposition = decompose_logits(model, inputs)
            predictions = decomposition.logits.argmax(dim=1)
            completeness_error = max(
                completeness_error,
                float((decomposition.logits - decomposition.reconstructed_logits).abs().max()),
            )
        logits, exact, gradcam = exact_and_gradcam(model, inputs, predictions)
        ig = integrated_gradients(model, inputs, predictions, ig_steps)
        random_map = torch.rand(exact.shape, generator=generator, device=device)
        heatmaps = {
            "exact_template_cam": exact,
            "gradcam": gradcam,
            "integrated_gradients": ig,
            "random": random_map,
        }
        original_target = logits.gather(1, predictions[:, None]).squeeze(1)
        original_correct = predictions == labels
        clean_correct_total += int(original_correct.sum())
        for explainer, heatmap in heatmaps.items():
            for fraction in fractions:
                deleted = delete_top_pixels(inputs, heatmap, fraction)
                with torch.no_grad():
                    deleted_logits = model(deleted)
                deleted_prediction = deleted_logits.argmax(dim=1)
                deleted_target = deleted_logits.gather(1, predictions[:, None]).squeeze(1)
                accumulator = accumulators[(explainer, fraction)]
                accumulator["target_logit_drop"] += float((original_target - deleted_target).sum())
                accumulator["prediction_flip"] += float((deleted_prediction != predictions).sum())
                accumulator["deleted_accuracy"] += float((deleted_prediction == labels).sum())
                accumulator["correct_prediction_retention"] += float(
                    ((deleted_prediction == labels) & original_correct).sum()
                )
        total += labels.numel()
    rows = []
    for (explainer, fraction), values in accumulators.items():
        rows.append(
            {
                "explainer": explainer,
                "fraction": fraction,
                "samples": total,
                "target_logit_drop": values["target_logit_drop"] / total,
                "prediction_flip_rate": values["prediction_flip"] / total,
                "deleted_accuracy": values["deleted_accuracy"] / total,
                "correct_prediction_retention": values[
                    "correct_prediction_retention"
                ] / max(1, clean_correct_total),
                "max_logit_completeness_error": completeness_error,
            }
        )
    return rows


def save_template_grid(path, dataset_name, method, model, inputs, labels, features, rows):
    rows_by_class = defaultdict(list)
    for row in rows:
        rows_by_class[row["anchor_class"]].append(row)
    selected = [
        max(rows_by_class[class_id], key=lambda row: row["anchor_alignment"])
        for class_id in range(model.spec.num_classes)
        if rows_by_class[class_id]
    ]
    columns = 3
    figure, axes = plt.subplots(len(selected), columns, figsize=(8, 2.2 * len(selected)))
    if len(selected) == 1:
        axes = axes[None, :]
    pixels = to_pixel_space(inputs, dataset_name).clamp(0, 1)
    for row_index, row in enumerate(selected):
        channel = row["channel"]
        scores = features[:, channel].amax(dim=(1, 2))
        top_indices = scores.topk(min(columns, scores.numel())).indices
        for column, sample_index in enumerate(top_indices):
            image = pixels[sample_index]
            heatmap = features[sample_index, channel]
            heatmap = normalise_heatmap(heatmap[None])[0]
            heatmap = F.interpolate(
                heatmap[None, None], size=image.shape[-2:], mode="bilinear", align_corners=False
            )[0, 0]
            axis = axes[row_index, column]
            if image.shape[0] == 1:
                axis.imshow(image[0], cmap="gray", vmin=0, vmax=1)
            else:
                axis.imshow(image.permute(1, 2, 0))
            axis.imshow(heatmap, cmap="magma", alpha=0.42, vmin=0, vmax=1)
            axis.set_title(
                f"a{row['anchor_class']} y{int(labels[sample_index])} c{channel}", fontsize=8
            )
            axis.axis("off")
    figure.suptitle(
        f"{dataset_name} | {method} | best-aligned template per class", y=0.998
    )
    # Reserve an explicit title band so the first template row cannot overlap
    # the figure title when there are many class rows.
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarise(rows, keys, metrics):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    output = []
    for key, group in sorted(groups.items()):
        item = dict(zip(keys, key))
        item["runs"] = len(group)
        for metric in metrics:
            values = [float(row[metric]) for row in group]
            item[f"mean_{metric}"] = statistics.mean(values)
            item[f"std_{metric}"] = statistics.stdev(values) if len(values) > 1 else 0.0
        output.append(item)
    return output


def paired_interval(differences):
    if len(differences) < 2:
        raise ValueError("At least two paired seeds are required")
    centre = statistics.mean(differences)
    spread = statistics.stdev(differences)
    half_width = T95.get(len(differences) - 1, 1.96) * spread / math.sqrt(len(differences))
    return centre, centre - half_width, centre + half_width


def semantic_paired(rows, reference="kaiming", candidate="lowrank_wmf"):
    metrics = ("mean_top_label_purity", "mean_semantic_purity", "mean_anchor_alignment")
    lookup = {
        (row["dataset"], row["model"], row["method"], int(row["model_seed"])): row
        for row in rows
    }
    output = []
    datasets_models = sorted({(row["dataset"], row["model"]) for row in rows})
    seeds = sorted({int(row["model_seed"]) for row in rows})
    for dataset, model in datasets_models:
        for metric in metrics:
            differences = []
            for seed in seeds:
                ref = lookup.get((dataset, model, reference, seed))
                cand = lookup.get((dataset, model, candidate, seed))
                if ref is not None and cand is not None:
                    differences.append(float(cand[metric]) - float(ref[metric]))
            if len(differences) < 2:
                continue
            centre, low, high = paired_interval(differences)
            output.append(
                {
                    "dataset": dataset, "model": model, "metric": metric,
                    "reference": reference, "candidate": candidate,
                    "paired_seeds": len(differences),
                    "mean_candidate_minus_reference": centre,
                    "ci95_low": low, "ci95_high": high,
                    "candidate_wins": sum(value > 0 for value in differences),
                }
            )
    return output


def explanation_paired(rows):
    metrics = ("target_logit_drop", "prediction_flip_rate", "deleted_accuracy")
    lookup = {
        (
            row["dataset"], row["model"], row["method"], row["explainer"],
            float(row["fraction"]), int(row["model_seed"]),
        ): row
        for row in rows
    }
    groups = sorted(
        {(row["dataset"], row["model"], row["method"], float(row["fraction"])) for row in rows}
    )
    seeds = sorted({int(row["model_seed"]) for row in rows})
    contrasts = (
        ("random", "exact_template_cam"),
        ("random", "integrated_gradients"),
        ("exact_template_cam", "integrated_gradients"),
    )
    output = []
    for dataset, model, method, fraction in groups:
        for reference, candidate in contrasts:
            for metric in metrics:
                differences = []
                for seed in seeds:
                    ref = lookup.get((dataset, model, method, reference, fraction, seed))
                    cand = lookup.get((dataset, model, method, candidate, fraction, seed))
                    if ref is not None and cand is not None:
                        differences.append(float(cand[metric]) - float(ref[metric]))
                if len(differences) < 2:
                    continue
                centre, low, high = paired_interval(differences)
                output.append(
                    {
                        "dataset": dataset, "model": model, "method": method,
                        "fraction": fraction, "metric": metric,
                        "reference_explainer": reference,
                        "candidate_explainer": candidate,
                        "paired_seeds": len(differences),
                        "mean_candidate_minus_reference": centre,
                        "ci95_low": low, "ci95_high": high,
                        "candidate_wins": sum(value > 0 for value in differences),
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
    semantic_output = []
    model_semantics = []
    explanation_output = []
    for (dataset_name, model_name, method, seed), (config, checkpoint) in sorted(runs.items()):
        if dataset_name not in dataset_cache:
            dataset_cache[dataset_name] = build_test_dataset(
                dataset_name, Path(config["data_root"]), Path(config["sign_root"])
            )
        dataset = dataset_cache[dataset_name]
        semantic_loader = make_loader(dataset, args.max_semantic_samples, args, device)
        explanation_loader = make_loader(dataset, args.explanation_samples, args, device)
        model = build_model(model_name, dataset_name).to(device)
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state)
        inputs, labels, features = collect_features(model, semantic_loader, device)
        channel_rows = semantic_rows(model, labels, features, args.top_examples)
        for row in channel_rows:
            semantic_output.append(
                {
                    "dataset": dataset_name, "model": model_name, "method": method,
                    "model_seed": seed,
                    **{key: value for key, value in row.items() if key != "top_indices"},
                }
            )
        model_semantics.append(
            {
                "dataset": dataset_name, "model": model_name, "method": method,
                "model_seed": seed,
                "mean_top_label_purity": statistics.mean(row["top_label_purity"] for row in channel_rows),
                "mean_semantic_purity": statistics.mean(row["semantic_purity"] for row in channel_rows),
                "mean_anchor_alignment": statistics.mean(row["anchor_alignment"] for row in channel_rows),
            }
        )
        if seed == args.visualisation_seed:
            save_template_grid(
                args.results_root / "figures" / f"{dataset_name}_{model_name}_{method}_seed{seed}.png",
                dataset_name, method, model, inputs, labels, features, channel_rows,
            )
        clean_correct = 0
        clean_total = 0
        with torch.inference_mode():
            for batch, target in explanation_loader:
                batch = batch.to(device, non_blocking=device.type == "cuda")
                target = target.to(device, non_blocking=device.type == "cuda")
                clean_correct += int((model(batch).argmax(1) == target).sum())
                clean_total += target.numel()
        clean_accuracy = clean_correct / clean_total
        for row in explanation_rows(
            model, explanation_loader, args.fractions, args.ig_steps,
            10000 + seed, device,
        ):
            explanation_output.append(
                {
                    "dataset": dataset_name, "model": model_name, "method": method,
                    "model_seed": seed, "clean_accuracy": clean_accuracy, **row,
                }
            )
        print(f"Completed {dataset_name}/{model_name}/{method}/seed{seed}")

    semantic_summary = summarise(
        model_semantics, ["dataset", "model", "method"],
        ["mean_top_label_purity", "mean_semantic_purity", "mean_anchor_alignment"],
    )
    explanation_summary = summarise(
        explanation_output, ["dataset", "model", "method", "explainer", "fraction"],
        ["clean_accuracy", "target_logit_drop", "prediction_flip_rate", "deleted_accuracy", "correct_prediction_retention"],
    )
    write_csv(args.results_root / "template_channel_rows.csv", semantic_output)
    write_csv(args.results_root / "template_model_metrics.csv", model_semantics)
    write_csv(args.results_root / "template_semantic_summary.csv", semantic_summary)
    write_csv(args.results_root / "template_semantic_paired.csv", semantic_paired(model_semantics))
    write_csv(args.results_root / "explanation_model_metrics.csv", explanation_output)
    write_csv(args.results_root / "explanation_summary.csv", explanation_summary)
    write_csv(args.results_root / "explanation_paired.csv", explanation_paired(explanation_output))
    config = vars(args).copy()
    config.update(
        {
            "output_roots": [str(path.resolve()) for path in args.output_roots],
            "results_root": str(args.results_root.resolve()),
            "device_resolved": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        }
    )
    args.results_root.mkdir(parents=True, exist_ok=True)
    (args.results_root / "evaluation_config.json").write_text(
        json.dumps(config, indent=2, default=str), encoding="utf-8"
    )
    print(f"Wrote semantic and explanation results to {args.results_root.resolve()}")


if __name__ == "__main__":
    main()
