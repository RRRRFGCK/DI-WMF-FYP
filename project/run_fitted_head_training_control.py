"""Train Kaiming features with the same fitted head used by full DI-WMF."""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from domain_mf.data import build_loaders
from domain_mf.initializers import calibrate_logit_scale, fit_classifier_head, initialise_model
from domain_mf.interpretability import capture_anchors
from domain_mf.models import build_model
from domain_mf.trainer import train_model


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["fashion", "cifar10", "sign"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "majorRevision" / "data")
    parser.add_argument(
        "--sign-root", type=Path, default=PROJECT_ROOT / "data" / "sign_language_mnist"
    )
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "outputs_convergence50_fitted_head_control"
    )
    parser.add_argument("--covariance-rank", type=int, default=16)
    parser.add_argument("--shrinkage", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def post_training_aulc(history_path):
    with history_path.open(newline="", encoding="utf-8") as handle:
        history = list(csv.DictReader(handle))
    return float(np.mean([float(row["val_accuracy"]) for row in history if int(row["epoch"]) >= 1]))


def main():
    args = parse_args()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for dataset in args.datasets:
        batch_size = 64 if dataset == "sign" else 512
        for seed in args.seeds:
            output_dir = args.output_root / f"{dataset}_standard_kaiming_conv_fitted_head_seed{seed}"
            summary_path = output_dir / "control_summary.json"
            if args.resume and summary_path.exists():
                summary_rows.append(json.loads(summary_path.read_text(encoding="utf-8")))
                print(f"reuse dataset={dataset} seed={seed}")
                continue
            seed_everything(seed)
            train_loader, fit_loader, val_loader, test_loader = build_loaders(
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
            model = build_model("standard", dataset).to(device)
            conv_report = initialise_model(model, fit_loader, "kaiming", device, layers=3)
            head_report = fit_classifier_head(
                model,
                fit_loader,
                "lowrank_wmf",
                device,
                shrinkage=args.shrinkage,
                covariance_rank=args.covariance_rank,
            )
            output_scale = calibrate_logit_scale(model, fit_loader, device, target_std=1.0)
            anchors = capture_anchors(model, 3)
            metrics = train_model(
                model,
                train_loader,
                val_loader,
                test_loader,
                device,
                anchors,
                output_dir,
                args.epochs,
                1e-3,
                0.1,
                "none",
                lr_scheduler="cosine",
                min_learning_rate=1e-5,
            )
            row = {
                "dataset": dataset,
                "seed": seed,
                "condition": "kaiming_conv__fitted_head",
                "initial_validation_accuracy": metrics["initial_val_accuracy"],
                "inclusive_aulc": metrics["validation_aulc_0T"],
                "post_training_aulc": post_training_aulc(output_dir / "history.csv"),
                "final_test_accuracy": metrics["test_accuracy"],
                "best_validation_accuracy": metrics["best_val_accuracy"],
                "best_epoch": metrics["best_epoch"],
                "training_seconds": metrics["training_seconds"],
                "device": str(device),
                "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            }
            configuration = {
                **row,
                "epochs": args.epochs,
                "batch_size": batch_size,
                "learning_rate": 1e-3,
                "scheduler": "cosine",
                "minimum_learning_rate": 1e-5,
                "train_fraction": 1.0,
                "validation_fraction": 0.1,
                "covariance_rank": args.covariance_rank,
                "shrinkage": args.shrinkage,
                "conv_initialisation_report": asdict(conv_report),
                "head_fit_report": head_report,
                "output_scale": output_scale,
            }
            output_dir.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
            (output_dir / "control_config.json").write_text(
                json.dumps(configuration, indent=2), encoding="utf-8"
            )
            summary_rows.append(row)
            print(json.dumps(row, indent=2))
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if summary_rows:
        with (args.output_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
            writer.writeheader()
            writer.writerows(summary_rows)


if __name__ == "__main__":
    main()
