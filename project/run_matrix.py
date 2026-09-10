import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Run a reproducible experiment matrix")
    parser.add_argument("--datasets", nargs="+", default=["mnist", "fashion"])
    parser.add_argument(
        "--methods", nargs="+", default=["kaiming", "classmean", "di_wmf"]
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--lr-scheduler", choices=["none", "cosine"], default="none"
    )
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--shrinkage", type=float, default=0.1)
    parser.add_argument("--covariance-rank", type=int, default=16)
    parser.add_argument("--covariance-ranks", nargs="+", type=int, default=None)
    parser.add_argument("--adaptive-ranks", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--logit-std", type=float, default=1.0)
    parser.add_argument("--reg-lambda", type=float, default=0.1)
    parser.add_argument("--sparsity-lambda", type=float, default=0.0)
    parser.add_argument(
        "--sparsity-mode", choices=["l1", "proximal"], default="l1"
    )
    parser.add_argument("--auxiliary-lambda", type=float, default=0.0)
    parser.add_argument(
        "--augmentation", choices=["none", "cifar_standard"], default="none"
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--model",
        choices=[
            "1layer", "2layer", "standard", "standard_half", "standard_quarter",
            "rotation_invariant", "correncoder", "correncoder_full", "concept",
            "smallresnet", "resnet18"
        ],
        default="2layer",
    )
    parser.add_argument("--init-layers", type=int, choices=[1, 2, 3], default=2)
    parser.add_argument("--with-regularised-wmf", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip an exactly matching run when final_metrics.json already exists.",
    )
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--measure-energy", action="store_true")
    parser.add_argument("--power-sample-interval", type=float, default=0.1)
    parser.add_argument("--gpu-index", type=int, default=0)
    return parser.parse_args()


def completed_run_keys(output_root: Path) -> set[tuple]:
    """Return exact protocol keys for completed runs under ``output_root``."""

    completed = set()
    for config_path in output_root.glob("*/config.json"):
        if not (config_path.parent / "final_metrics.json").exists():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        completed.add(
            (
                config.get("dataset"), config.get("model"), config.get("init"),
                int(config.get("seed", -1)), int(config.get("epochs", -1)),
                float(config.get("train_fraction", -1)),
                float(config.get("val_fraction", -1)),
                int(config.get("batch_size", -1)),
                float(config.get("learning_rate", -1)),
                config.get("lr_scheduler") or "none",
                float(config.get("min_learning_rate") or 1e-5),
                float(config.get("shrinkage", -1)),
                int(config.get("covariance_rank", -1)),
                float(config.get("logit_std", -1)),
                int(config.get("init_layers", -1)),
                float(config.get("sparsity_lambda") or 0.0),
                config.get("sparsity_mode") or "l1",
                float(config.get("auxiliary_lambda") or 0.0),
                tuple(config.get("adaptive_ranks") or [4, 8, 16]),
                config.get("augmentation") or "none",
            )
        )
    return completed


def requested_run_key(args, dataset, method, seed, covariance_rank) -> tuple:
    return (
        dataset, args.model, method, seed, args.epochs, args.train_fraction,
        args.val_fraction, args.batch_size, args.learning_rate, args.lr_scheduler,
        args.min_learning_rate, args.shrinkage, covariance_rank, args.logit_std,
        args.init_layers, args.sparsity_lambda, args.sparsity_mode,
        args.auxiliary_lambda, tuple(args.adaptive_ranks), args.augmentation,
    )


def main():
    args = parse_args()
    commands = []
    completed = completed_run_keys(args.output_root) if args.resume else set()
    skipped = 0
    for dataset in args.datasets:
        for seed in args.seeds:
            for method in args.methods:
                ranks = (
                    args.covariance_ranks
                    if method == "lowrank_wmf" and args.covariance_ranks
                    else [args.covariance_rank]
                )
                for covariance_rank in ranks:
                    key = requested_run_key(
                        args, dataset, method, seed, covariance_rank
                    )
                    if key in completed:
                        skipped += 1
                        print(
                            f"reused dataset={dataset} model={args.model} "
                            f"method={method} seed={seed} rank={covariance_rank}"
                        )
                        continue
                    command = [
                        sys.executable,
                        str(ROOT / "run_experiment.py"),
                        "--dataset", dataset,
                        "--model", args.model,
                        "--init", method,
                        "--init-layers", str(args.init_layers),
                        "--seed", str(seed),
                        "--epochs", str(args.epochs),
                        "--train-fraction", str(args.train_fraction),
                        "--val-fraction", str(args.val_fraction),
                        "--batch-size", str(args.batch_size),
                        "--num-workers", str(args.num_workers),
                        "--learning-rate", str(args.learning_rate),
                        "--lr-scheduler", args.lr_scheduler,
                        "--min-learning-rate", str(args.min_learning_rate),
                        "--shrinkage", str(args.shrinkage),
                        "--covariance-rank", str(covariance_rank),
                        "--adaptive-ranks", *[str(rank) for rank in args.adaptive_ranks],
                        "--logit-std", str(args.logit_std),
                        "--device", args.device,
                        "--sparsity-lambda", str(args.sparsity_lambda),
                        "--sparsity-mode", args.sparsity_mode,
                        "--auxiliary-lambda", str(args.auxiliary_lambda),
                        "--augmentation", args.augmentation,
                        "--output-root", str(args.output_root),
                    ]
                    commands.append(command)
                    if args.measure_energy:
                        command.extend(
                            [
                                "--measure-energy",
                                "--power-sample-interval",
                                str(args.power_sample_interval),
                                "--gpu-index",
                                str(args.gpu_index),
                            ]
                        )
                    if method == "di_wmf" and args.with_regularised_wmf:
                        commands.append(
                            command
                            + ["--reg-schedule", "linear", "--reg-lambda", str(args.reg_lambda)]
                        )
    for command in commands:
        print(" ".join(command))
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)
    print(f"Prepared {len(commands)} runs; reused {skipped} completed runs")


if __name__ == "__main__":
    main()
