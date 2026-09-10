"""Evaluate calibration, uncertainty, OOD detection and adversarial robustness."""

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from domain_mf.data import build_loaders, build_test_dataset
from domain_mf.models import SPECS, build_model


T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)


def parse_args():
    parser = argparse.ArgumentParser(description="Reliability and shift evaluation")
    parser.add_argument("--output-roots", nargs="+", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, default=Path("reliability_results"))
    parser.add_argument("--datasets", nargs="+", default=["fashion", "cifar10", "sign"])
    parser.add_argument("--methods", nargs="+", default=["kaiming", "lowrank_wmf"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--max-eval-samples", type=int, default=2000)
    parser.add_argument("--max-adversarial-samples", type=int, default=1000)
    parser.add_argument("--pgd-steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
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
            if int(config.get("epochs", -1)) != 20 or float(config.get("train_fraction", -1)) != 1.0:
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


def limited_loader(dataset, max_samples, batch_size, workers, pin_memory):
    if max_samples and len(dataset) > max_samples:
        indices = np.linspace(0, len(dataset) - 1, max_samples, dtype=int).tolist()
        dataset = Subset(dataset, indices)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=pin_memory,
    )


def build_ood_dataset(dataset_name, data_root, sign_root):
    spec = SPECS[dataset_name]
    if dataset_name == "fashion":
        transform = transforms.Compose(
            [transforms.Resize(spec.image_size), transforms.ToTensor()]
        )
        return datasets.MNIST(data_root, train=False, download=False, transform=transform), "mnist"
    if dataset_name == "cifar10":
        transform = transforms.Compose(
            [
                transforms.Resize((spec.image_size, spec.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ]
        )
        return datasets.ImageFolder(sign_root / "test_10", transform=transform), "sign"
    if dataset_name == "sign":
        transform = transforms.Compose(
            [transforms.Resize(spec.image_size), transforms.ToTensor()]
        )
        return datasets.CIFAR10(data_root, train=False, download=False, transform=transform), "cifar10"
    raise ValueError(f"No OOD pairing defined for {dataset_name}")


def collect_logits(model, loader, device):
    logits = []
    labels = []
    model.eval()
    with torch.inference_mode():
        for inputs, target in loader:
            logits.append(model(inputs.to(device, non_blocking=device.type == "cuda")).cpu())
            labels.append(torch.as_tensor(target).cpu())
    return torch.cat(logits), torch.cat(labels).long()


def temperature_scale(validation_logits, validation_labels):
    logits = validation_logits.detach()
    labels = validation_labels.detach()
    log_temperature = torch.zeros((), device=logits.device, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=50)

    def closure():
        optimizer.zero_grad(set_to_none=True)
        temperature = log_temperature.clamp(-5.0, 5.0).exp()
        loss = F.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().clamp(-5.0, 5.0).exp())


def binary_auc(positive_scores, negative_scores):
    scores = np.concatenate([positive_scores, negative_scores]).astype(float)
    labels = np.concatenate(
        [np.ones(len(positive_scores), dtype=int), np.zeros(len(negative_scores), dtype=int)]
    )
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    index = 0
    while index < len(scores):
        end = index + 1
        while end < len(scores) and scores[order[end]] == scores[order[index]]:
            end += 1
        ranks[order[index:end]] = 0.5 * (index + 1 + end)
        index = end
    positive_count = labels.sum()
    negative_count = len(labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        return float("nan")
    rank_sum = ranks[labels == 1].sum()
    return float(
        (rank_sum - positive_count * (positive_count + 1) / 2)
        / (positive_count * negative_count)
    )


def expected_calibration_error(confidence, correct, bins=15):
    edges = torch.linspace(0, 1, bins + 1)
    value = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            value += float(mask.float().mean()) * abs(
                float(confidence[mask].mean()) - float(correct[mask].float().mean())
            )
    return value


def area_under_risk_coverage(confidence, correct):
    order = torch.argsort(confidence, descending=True)
    errors = (~correct[order]).float()
    risks = errors.cumsum(0) / torch.arange(1, len(errors) + 1)
    return float(risks.mean())


def classification_metrics(logits, labels, temperature=1.0):
    logits = logits / temperature
    probabilities = logits.softmax(1)
    confidence, predictions = probabilities.max(1)
    correct = predictions.eq(labels)
    one_hot = F.one_hot(labels, probabilities.shape[1]).float()
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(1)
    correct_confidence = confidence[correct].numpy()
    incorrect_confidence = confidence[~correct].numpy()
    return {
        "accuracy": 100.0 * float(correct.float().mean()),
        "nll": float(F.cross_entropy(logits, labels)),
        "brier": float((probabilities - one_hot).square().sum(1).mean()),
        "ece15": expected_calibration_error(confidence, correct),
        "mean_entropy": float(entropy.mean()),
        "correctness_auroc": binary_auc(correct_confidence, incorrect_confidence),
        "aurc": area_under_risk_coverage(confidence, correct),
    }


def ood_metrics(id_logits, ood_logits):
    id_probabilities = id_logits.softmax(1)
    ood_probabilities = ood_logits.softmax(1)
    id_msp = id_probabilities.max(1).values.numpy()
    ood_msp = ood_probabilities.max(1).values.numpy()
    id_energy = torch.logsumexp(id_logits, 1).numpy()
    ood_energy = torch.logsumexp(ood_logits, 1).numpy()
    id_neg_entropy = (
        id_probabilities * id_probabilities.clamp_min(1e-12).log()
    ).sum(1).numpy()
    ood_neg_entropy = (
        ood_probabilities * ood_probabilities.clamp_min(1e-12).log()
    ).sum(1).numpy()
    threshold = float(np.quantile(id_msp, 0.05))
    return {
        "msp_auroc": binary_auc(id_msp, ood_msp),
        "energy_auroc": binary_auc(id_energy, ood_energy),
        "neg_entropy_auroc": binary_auc(id_neg_entropy, ood_neg_entropy),
        "msp_fpr95": float((ood_msp >= threshold).mean()),
        "id_mean_msp": float(id_msp.mean()),
        "ood_mean_msp": float(ood_msp.mean()),
    }


def normalisation(dataset, device):
    if dataset == "cifar10":
        mean, std = CIFAR_MEAN, CIFAR_STD
    else:
        channels = SPECS[dataset].input_channels
        mean, std = (0.0,) * channels, (1.0,) * channels
    mean = torch.tensor(mean, device=device).view(1, -1, 1, 1)
    std = torch.tensor(std, device=device).view(1, -1, 1, 1)
    return mean, std


def attack_batch(model, inputs, labels, dataset, epsilon, attack, steps):
    mean, std = normalisation(dataset, inputs.device)
    clean_pixels = (inputs * std + mean).clamp(0, 1).detach()
    if attack == "fgsm":
        pixels = clean_pixels.clone().requires_grad_(True)
        loss = F.cross_entropy(model((pixels - mean) / std), labels)
        gradient = torch.autograd.grad(loss, pixels)[0]
        adversarial = (pixels + epsilon * gradient.sign()).clamp(0, 1).detach()
    elif attack == "pgd":
        adversarial = (
            clean_pixels + torch.empty_like(clean_pixels).uniform_(-epsilon, epsilon)
        ).clamp(0, 1)
        step_size = 2.5 * epsilon / max(steps, 1)
        for _ in range(steps):
            adversarial.requires_grad_(True)
            loss = F.cross_entropy(model((adversarial - mean) / std), labels)
            gradient = torch.autograd.grad(loss, adversarial)[0]
            adversarial = adversarial.detach() + step_size * gradient.sign()
            adversarial = torch.maximum(
                torch.minimum(adversarial, clean_pixels + epsilon),
                clean_pixels - epsilon,
            ).clamp(0, 1)
    else:
        raise ValueError(attack)
    return (adversarial - mean) / std


def adversarial_rows(model, loader, dataset, device, steps):
    epsilons = [0.05, 0.1, 0.2] if dataset == "fashion" else [2 / 255, 4 / 255, 8 / 255]
    clean_correct = total = 0
    batches = []
    model.eval()
    # Keep ordinary tensors because the cached batches are reused by attacks
    # that require input gradients after clean evaluation.
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")
            clean_correct += int((model(inputs).argmax(1) == labels).sum())
            total += labels.numel()
            batches.append((inputs, labels))
    clean_accuracy = 100.0 * clean_correct / total
    rows = [
        {"attack": "clean", "epsilon": 0.0, "accuracy": clean_accuracy, "retention": 1.0}
    ]
    for attack, values in (("fgsm", epsilons), ("pgd", [epsilons[-1]])):
        for epsilon in values:
            correct = 0
            for inputs, labels in batches:
                adversarial = attack_batch(
                    model, inputs, labels, dataset, epsilon, attack, steps
                )
                with torch.inference_mode():
                    correct += int((model(adversarial).argmax(1) == labels).sum())
            accuracy = 100.0 * correct / total
            rows.append(
                {
                    "attack": attack, "epsilon": epsilon, "accuracy": accuracy,
                    "retention": accuracy / max(clean_accuracy, 1e-12),
                }
            )
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows, group_fields, metric_fields):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in group_fields)].append(row)
    output = []
    for key, group in sorted(grouped.items()):
        item = dict(zip(group_fields, key))
        item["runs"] = len(group)
        for metric in metric_fields:
            values = [float(row[metric]) for row in group if math.isfinite(float(row[metric]))]
            item[f"mean_{metric}"] = statistics.mean(values) if values else ""
            item[f"std_{metric}"] = statistics.stdev(values) if len(values) > 1 else 0.0
        output.append(item)
    return output


def paired(rows, group_fields, metric_fields):
    indexed = defaultdict(dict)
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        indexed[key][(row["method"], int(row["seed"]))] = row
    output = []
    for key, group in sorted(indexed.items()):
        seeds = sorted(
            seed for method, seed in group if method == "kaiming"
            and ("lowrank_wmf", seed) in group
        )
        for metric in metric_fields:
            differences = [
                float(group[("lowrank_wmf", seed)][metric])
                - float(group[("kaiming", seed)][metric])
                for seed in seeds
            ]
            differences = [value for value in differences if math.isfinite(value)]
            if not differences:
                continue
            mean = statistics.mean(differences)
            std = statistics.stdev(differences) if len(differences) > 1 else 0.0
            half = T95[len(differences) - 1] * std / math.sqrt(len(differences)) if len(differences) > 1 else 0.0
            output.append(
                {
                    **dict(zip(group_fields, key)), "metric": metric,
                    "paired_seeds": len(differences),
                    "mean_lowrank_minus_kaiming": mean,
                    "ci95_low": mean - half, "ci95_high": mean + half,
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
    first_config = next(iter(runs.values()))[0]
    data_root = args.data_root or Path(first_config["data_root"])
    sign_root = args.sign_root or Path(first_config["sign_root"])
    datasets_cache = {}
    for dataset in args.datasets:
        id_dataset = build_test_dataset(dataset, data_root, sign_root)
        ood_dataset, ood_name = build_ood_dataset(dataset, data_root, sign_root)
        datasets_cache[dataset] = (
            limited_loader(id_dataset, args.max_eval_samples, args.batch_size, args.num_workers, device.type == "cuda"),
            limited_loader(ood_dataset, args.max_eval_samples, args.batch_size, args.num_workers, device.type == "cuda"),
            limited_loader(id_dataset, args.max_adversarial_samples, min(64, args.batch_size), args.num_workers, device.type == "cuda"),
            ood_name,
        )

    calibration_output = []
    ood_output = []
    adversarial_output = []
    for (dataset, method, seed), (config, checkpoint) in sorted(runs.items()):
        torch.manual_seed(1000 + seed)
        model = build_model("standard", dataset).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        _, _, val_loader, _ = build_loaders(
            dataset, data_root, sign_root,
            64 if dataset == "sign" else args.batch_size,
            seed, 1.0, float(config.get("val_fraction", 0.1)),
            args.num_workers, pin_memory=device.type == "cuda",
        )
        validation_logits, validation_labels = collect_logits(model, val_loader, device)
        id_loader, ood_loader, adversarial_loader, ood_name = datasets_cache[dataset]
        id_logits, id_labels = collect_logits(model, id_loader, device)
        ood_logits, _ = collect_logits(model, ood_loader, device)
        temperature = temperature_scale(
            validation_logits.to(device), validation_labels.to(device)
        )
        for calibration, value in (("raw", 1.0), ("temperature", temperature)):
            calibration_output.append(
                {
                    "dataset": dataset, "method": method, "seed": seed,
                    "calibration": calibration, "temperature": temperature,
                    **classification_metrics(id_logits, id_labels, value),
                }
            )
        ood_output.append(
            {
                "dataset": dataset, "ood_dataset": ood_name,
                "method": method, "seed": seed,
                **ood_metrics(id_logits, ood_logits),
            }
        )
        for row in adversarial_rows(model, adversarial_loader, dataset, device, args.pgd_steps):
            adversarial_output.append(
                {"dataset": dataset, "method": method, "seed": seed, **row}
            )
        print(f"Completed reliability evaluation {dataset}/{method}/seed{seed}")

    calibration_metrics = ["accuracy", "nll", "brier", "ece15", "mean_entropy", "correctness_auroc", "aurc"]
    ood_metric_names = ["msp_auroc", "energy_auroc", "neg_entropy_auroc", "msp_fpr95", "id_mean_msp", "ood_mean_msp"]
    adversarial_metrics = ["accuracy", "retention"]
    write_csv(args.results_root / "calibration_rows.csv", calibration_output)
    write_csv(args.results_root / "calibration_aggregate.csv", aggregate(calibration_output, ["dataset", "method", "calibration"], calibration_metrics + ["temperature"]))
    write_csv(args.results_root / "calibration_paired.csv", paired(calibration_output, ["dataset", "calibration"], calibration_metrics))
    write_csv(args.results_root / "ood_rows.csv", ood_output)
    write_csv(args.results_root / "ood_aggregate.csv", aggregate(ood_output, ["dataset", "ood_dataset", "method"], ood_metric_names))
    write_csv(args.results_root / "ood_paired.csv", paired(ood_output, ["dataset", "ood_dataset"], ood_metric_names))
    write_csv(args.results_root / "adversarial_rows.csv", adversarial_output)
    write_csv(args.results_root / "adversarial_aggregate.csv", aggregate(adversarial_output, ["dataset", "method", "attack", "epsilon"], adversarial_metrics))
    write_csv(args.results_root / "adversarial_paired.csv", paired(adversarial_output, ["dataset", "attack", "epsilon"], adversarial_metrics))
    config = vars(args).copy()
    config.update(
        {
            "output_roots": [str(path.resolve()) for path in args.output_roots],
            "results_root": str(args.results_root.resolve()),
            "device_resolved": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "ood_pairs": {dataset: datasets_cache[dataset][3] for dataset in args.datasets},
        }
    )
    args.results_root.mkdir(parents=True, exist_ok=True)
    (args.results_root / "evaluation_config.json").write_text(
        json.dumps(config, indent=2, default=str), encoding="utf-8"
    )
    print(f"Wrote reliability results to {args.results_root.resolve()}")


if __name__ == "__main__":
    main()
