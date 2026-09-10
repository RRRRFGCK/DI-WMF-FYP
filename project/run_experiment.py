import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from domain_mf.adaptive import AdaptiveCandidate, select_adaptive_initialisation
from domain_mf.data import build_loaders
from domain_mf.energy import NvidiaPowerSampler, measure_idle_power
from domain_mf.initializers import calibrate_logit_scale, initialise_model
from domain_mf.interpretability import capture_anchors
from domain_mf.models import SPECS, build_model
from domain_mf.trainer import train_model


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent


def parse_args():
    parser = argparse.ArgumentParser(description="Domain-informed matched-filter CNN experiment")
    parser.add_argument("--dataset", choices=sorted(SPECS), default="mnist")
    parser.add_argument(
        "--model",
        choices=[
            "1layer", "2layer", "standard", "standard_half", "standard_quarter",
            "rotation_invariant", "correncoder", "correncoder_full", "concept",
            "smallresnet", "resnet18"
        ],
        default="2layer",
    )
    parser.add_argument(
        "--init",
        choices=[
            "random", "random_stem", "kaiming", "classmean", "kmeans",
            "gabor", "pca", "di_wmf", "lowrank_wmf",
            "adaptive_wmf",
        ],
        default="di_wmf",
    )
    parser.add_argument("--init-layers", type=int, choices=[1, 2, 3], default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--lr-scheduler", choices=["none", "cosine"], default="none"
    )
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--shrinkage", type=float, default=0.1)
    parser.add_argument("--covariance-rank", type=int, default=16)
    parser.add_argument(
        "--adaptive-ranks", nargs="+", type=int, default=[4, 8, 16],
        help="Low-rank candidates considered by adaptive_wmf.",
    )
    parser.add_argument("--logit-std", type=float, default=1.0)
    parser.add_argument("--no-logit-calibration", action="store_true")
    parser.add_argument("--reg-lambda", type=float, default=0.1)
    parser.add_argument("--reg-schedule", choices=["none", "fixed", "linear"], default="none")
    parser.add_argument("--sparsity-lambda", type=float, default=0.0)
    parser.add_argument(
        "--sparsity-mode", choices=["l1", "proximal"], default="l1"
    )
    parser.add_argument("--auxiliary-lambda", type=float, default=0.0)
    parser.add_argument(
        "--augmentation", choices=["none", "cifar_standard"], default="none"
    )
    parser.add_argument(
        "--data-root", type=Path, default=PROJECT_ROOT / "majorRevision" / "data"
    )
    parser.add_argument(
        "--sign-root", type=Path, default=PROJECT_ROOT / "data" / "sign_language_mnist"
    )
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--measure-energy", action="store_true")
    parser.add_argument("--power-sample-interval", type=float, default=0.1)
    parser.add_argument("--gpu-index", type=int, default=0)
    return parser.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = parse_args()
    seed_everything(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable. Install a CUDA-enabled PyTorch "
            "build or run with --device cpu."
        )
    train_loader, fit_loader, val_loader, test_loader = build_loaders(
        args.dataset,
        args.data_root,
        args.sign_root,
        args.batch_size,
        args.seed,
        args.train_fraction,
        args.val_fraction,
        args.num_workers,
        pin_memory=device.type == "cuda",
        augmentation=args.augmentation,
    )
    model = build_model(args.model, args.dataset).to(device)
    if args.measure_energy and device.type != "cuda":
        raise ValueError("--measure-energy requires --device cuda")
    idle_watts = None
    initialisation_sampler = None
    if args.measure_energy:
        idle_watts = measure_idle_power(args.gpu_index)
        initialisation_sampler = NvidiaPowerSampler(
            idle_watts,
            gpu_index=args.gpu_index,
            interval_seconds=args.power_sample_interval,
        ).start()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    initialisation_start = time.perf_counter()
    adaptive_selection = None
    if args.init == "adaptive_wmf":
        candidates = [
            AdaptiveCandidate("kmeans", shrinkage=args.shrinkage),
            AdaptiveCandidate("di_wmf", shrinkage=args.shrinkage),
            *[
                AdaptiveCandidate(
                    "lowrank_wmf", covariance_rank=rank,
                    shrinkage=args.shrinkage,
                )
                for rank in args.adaptive_ranks
            ],
        ]
        model, report, adaptive_selection = select_adaptive_initialisation(
            lambda: build_model(args.model, args.dataset),
            fit_loader,
            val_loader,
            device,
            candidates,
            layers=args.init_layers,
            seed=args.seed,
            calibrate_logits=not args.no_logit_calibration,
            target_logit_std=args.logit_std,
        )
        # Candidate evaluation should not change the stochastic training stream.
        seed_everything(args.seed)
    else:
        report = initialise_model(
            model,
            fit_loader,
            args.init,
            device,
            layers=args.init_layers,
            shrinkage=args.shrinkage,
            covariance_rank=args.covariance_rank,
        )
        if hasattr(model, "initialise_decoder_from_encoder"):
            model.initialise_decoder_from_encoder()
        if not args.no_logit_calibration:
            report.output_scale = calibrate_logit_scale(
                model, fit_loader, device, target_std=args.logit_std
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    initialisation_seconds = time.perf_counter() - initialisation_start
    initialisation_energy = (
        initialisation_sampler.stop().to_dict()
        if initialisation_sampler is not None
        else None
    )
    anchors = capture_anchors(model, args.init_layers)
    fraction_label = str(args.train_fraction).replace(".", "p")
    lambda_label = str(args.reg_lambda).replace(".", "p")
    shrinkage_label = str(args.shrinkage).replace(".", "p")
    logit_label = "raw" if args.no_logit_calibration else str(args.logit_std).replace(".", "p")
    run_name = (
        f"{args.dataset}_{args.model}_{args.init}_layers{args.init_layers}_"
        f"seed{args.seed}_frac{fraction_label}_epochs{args.epochs}_"
        f"{args.reg_schedule}{lambda_label}_shrink{shrinkage_label}_logit{logit_label}"
    )
    if args.init == "lowrank_wmf":
        run_name += f"_rank{args.covariance_rank}"
    if args.init == "adaptive_wmf":
        run_name += f"_selected{adaptive_selection.selected_candidate}"
    if args.model == "concept":
        sparse_label = str(args.sparsity_lambda).replace(".", "p")
        run_name += f"_sparse{sparse_label}_{args.sparsity_mode}"
    if args.model in {"correncoder", "correncoder_full"}:
        auxiliary_label = str(args.auxiliary_lambda).replace(".", "p")
        run_name += f"_aux{auxiliary_label}"
    if args.augmentation != "none":
        run_name += f"_aug{args.augmentation}"
    if args.lr_scheduler != "none":
        minimum_label = str(args.min_learning_rate).replace(".", "p")
        run_name += f"_lr{args.lr_scheduler}{minimum_label}"
    output_dir = args.output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.update(
        {
            "data_root": str(args.data_root.resolve()),
            "sign_root": str(args.sign_root.resolve()),
            "output_root": str(args.output_root.resolve()),
            "device_resolved": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
            ),
            "model_spec": asdict(SPECS[args.dataset]),
            "initialisation_report": asdict(report),
            "initialisation_seconds": initialisation_seconds,
            "initialisation_energy": initialisation_energy,
            "adaptive_selection": (
                adaptive_selection.to_dict() if adaptive_selection else None
            ),
            "train_samples": len(train_loader.dataset),
            "validation_samples": len(val_loader.dataset),
            "test_samples": len(test_loader.dataset),
        }
    )
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, default=str)
    print(json.dumps(config, indent=2, default=str))
    training_sampler = None
    if args.measure_energy:
        training_sampler = NvidiaPowerSampler(
            idle_watts,
            gpu_index=args.gpu_index,
            interval_seconds=args.power_sample_interval,
        ).start()
    metrics = train_model(
        model,
        train_loader,
        val_loader,
        test_loader,
        device,
        anchors,
        output_dir,
        args.epochs,
        args.learning_rate,
        args.reg_lambda,
        args.reg_schedule,
        args.sparsity_lambda,
        args.sparsity_mode,
        args.auxiliary_lambda,
        args.lr_scheduler,
        args.min_learning_rate,
    )
    training_energy = (
        training_sampler.stop().to_dict() if training_sampler is not None else None
    )
    if args.measure_energy:
        metrics["energy_measurement"] = {
            "initialisation": initialisation_energy,
            "training_and_evaluation": training_energy,
            "total_gross_joules": (
                initialisation_energy["gross_joules"]
                + training_energy["gross_joules"]
            ),
            "total_net_joules_above_idle": (
                initialisation_energy["net_joules_above_idle"]
                + training_energy["net_joules_above_idle"]
            ),
        }
        config["energy_measurement"] = metrics["energy_measurement"]
        with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2, default=str)
        with (output_dir / "final_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"Results: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
