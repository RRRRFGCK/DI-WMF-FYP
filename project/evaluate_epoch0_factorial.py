"""Four-cell Epoch-0 control for convolution and classifier initialisation.

The experiment crosses a Kaiming or DI-WMF feature extractor with a Kaiming or
training-fold-fitted classifier.  No gradient update is performed.  This
separates useful convolutional templates from the much simpler effect of
fitting a linear discriminant on frozen penultimate features.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import t as student_t
from scipy.stats import ttest_rel

from domain_mf.data import build_loaders
from domain_mf.initializers import (
    calibrate_logit_scale,
    fit_classifier_head,
    initialise_model,
)
from domain_mf.models import build_model
from domain_mf.trainer import evaluate


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
CONDITIONS = (
    "kaiming_conv__kaiming_head",
    "kaiming_conv__fitted_head",
    "di_conv__kaiming_head",
    "di_conv__fitted_head",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["fashion", "cifar10", "sign"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "majorRevision" / "data")
    parser.add_argument(
        "--sign-root", type=Path, default=PROJECT_ROOT / "data" / "sign_language_mnist"
    )
    parser.add_argument("--output-root", type=Path, default=ROOT / "epoch0_factorial_results")
    parser.add_argument("--covariance-rank", type=int, default=16)
    parser.add_argument("--shrinkage", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    half = (
        float(student_t.ppf(0.975, values.size - 1) * values.std(ddof=1) / math.sqrt(values.size))
        if values.size > 1
        else 0.0
    )
    return mean, mean - half, mean + half


def initialise_condition(model, condition, fit_loader, device, args):
    if condition.startswith("kaiming_conv"):
        report = initialise_model(model, fit_loader, "kaiming", device, layers=3)
    else:
        report = initialise_model(
            model,
            fit_loader,
            "lowrank_wmf",
            device,
            layers=2,
            shrinkage=args.shrinkage,
            covariance_rank=args.covariance_rank,
        )
    head_report = None
    if condition.endswith("fitted_head"):
        head_report = fit_classifier_head(
            model,
            fit_loader,
            "lowrank_wmf",
            device,
            shrinkage=args.shrinkage,
            covariance_rank=args.covariance_rank,
        )
    output_scale = calibrate_logit_scale(model, fit_loader, device, target_std=1.0)
    return asdict(report), head_report, output_scale


def aggregate(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    aggregate_rows = []
    contrast_rows = []
    for dataset in sorted({row["dataset"] for row in rows}):
        subset = [row for row in rows if row["dataset"] == dataset]
        by_condition = {condition: {} for condition in CONDITIONS}
        for row in subset:
            by_condition[row["condition"]][int(row["seed"])] = row
        for condition in CONDITIONS:
            for metric in ("validation_accuracy", "test_accuracy"):
                values = np.array(
                    [float(by_condition[condition][seed][metric]) for seed in sorted(by_condition[condition])]
                )
                mean, low, high = mean_ci(values)
                aggregate_rows.append(
                    {
                        "dataset": dataset,
                        "condition": condition,
                        "metric": metric,
                        "runs": values.size,
                        "mean": mean,
                        "ci95_low": low,
                        "ci95_high": high,
                    }
                )

        contrast_specs = {
            "fitted_head_effect_given_kaiming_conv": (
                "kaiming_conv__fitted_head", "kaiming_conv__kaiming_head"
            ),
            "fitted_head_effect_given_di_conv": (
                "di_conv__fitted_head", "di_conv__kaiming_head"
            ),
            "di_conv_effect_given_kaiming_head": (
                "di_conv__kaiming_head", "kaiming_conv__kaiming_head"
            ),
            "di_conv_effect_given_fitted_head": (
                "di_conv__fitted_head", "kaiming_conv__fitted_head"
            ),
        }
        common = sorted(set.intersection(*(set(by_condition[c]) for c in CONDITIONS)))
        for metric in ("validation_accuracy", "test_accuracy"):
            for contrast, (left, right) in contrast_specs.items():
                delta = np.array(
                    [
                        float(by_condition[left][seed][metric])
                        - float(by_condition[right][seed][metric])
                        for seed in common
                    ]
                )
                mean, low, high = mean_ci(delta)
                p_value = float(ttest_rel(
                    [float(by_condition[left][seed][metric]) for seed in common],
                    [float(by_condition[right][seed][metric]) for seed in common],
                ).pvalue)
                contrast_rows.append(
                    {
                        "dataset": dataset,
                        "contrast": contrast,
                        "metric": metric,
                        "paired_seeds": len(common),
                        "mean_paired_difference": mean,
                        "ci95_low": low,
                        "ci95_high": high,
                        "p_value_raw": p_value,
                    }
                )

            interaction = np.array(
                [
                    (
                        float(by_condition["di_conv__fitted_head"][seed][metric])
                        - float(by_condition["di_conv__kaiming_head"][seed][metric])
                    )
                    - (
                        float(by_condition["kaiming_conv__fitted_head"][seed][metric])
                        - float(by_condition["kaiming_conv__kaiming_head"][seed][metric])
                    )
                    for seed in common
                ]
            )
            mean, low, high = mean_ci(interaction)
            contrast_rows.append(
                {
                    "dataset": dataset,
                    "contrast": "conv_by_head_interaction",
                    "metric": metric,
                    "paired_seeds": len(common),
                    "mean_paired_difference": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "p_value_raw": "",
                }
            )
    return aggregate_rows, contrast_rows


def main():
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    args.output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for dataset in args.datasets:
        batch_size = 64 if dataset == "sign" else 512
        for seed in args.seeds:
            seed_everything(seed)
            _, fit_loader, val_loader, test_loader = build_loaders(
                dataset,
                args.data_root,
                args.sign_root,
                batch_size,
                seed,
                train_fraction=1.0,
                val_fraction=0.1,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
            )
            for condition in CONDITIONS:
                seed_everything(seed)
                model = build_model("standard", dataset).to(device)
                start = time.perf_counter()
                init_report, head_report, output_scale = initialise_condition(
                    model, condition, fit_loader, device, args
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - start
                validation = evaluate(model, val_loader, device)
                test = evaluate(model, test_loader, device)
                row = {
                    "dataset": dataset,
                    "seed": seed,
                    "condition": condition,
                    "conv_initialisation": "lowrank_wmf" if condition.startswith("di_conv") else "kaiming",
                    "head_initialisation": "lowrank_fitted" if condition.endswith("fitted_head") else "kaiming",
                    "validation_accuracy": validation["accuracy"],
                    "test_accuracy": test["accuracy"],
                    "initialisation_seconds": elapsed,
                    "output_scale": output_scale,
                    "device": str(device),
                    "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                    "initialised_modules": ";".join(init_report.get("initialised_modules", [])),
                    "fitted_head_rank": "" if head_report is None else head_report["covariance_rank"],
                }
                rows.append(row)
                print(json.dumps(row, indent=2))
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    aggregate_rows, contrast_rows = aggregate(rows)
    write_csv(args.output_root / "epoch0_factorial_rows.csv", rows)
    write_csv(args.output_root / "epoch0_factorial_aggregate.csv", aggregate_rows)
    write_csv(args.output_root / "epoch0_factorial_contrasts.csv", contrast_rows)
    configuration = {
        "datasets": args.datasets,
        "seeds": args.seeds,
        "conditions": CONDITIONS,
        "covariance_rank": args.covariance_rank,
        "shrinkage": args.shrinkage,
        "data_root": str(args.data_root.resolve()),
        "sign_root": str(args.sign_root.resolve()),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "note": "No gradient updates; every fitted head uses training-fold features only.",
    }
    with (args.output_root / "evaluation_config.json").open("w", encoding="utf-8") as handle:
        json.dump(configuration, handle, indent=2)


if __name__ == "__main__":
    main()
