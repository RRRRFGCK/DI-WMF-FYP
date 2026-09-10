"""Signer-disjoint StandardCNN evaluation on the original static-ASL images.

Each of the seven users is held out once for testing.  The next user in the
fixed numeric order is used for validation and the remaining five users are
used for fitting and optimisation.  Bounding-box metadata from the original
CSV files is applied before the common 64 x 64 transform.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.stats import t as student_t
from scipy.stats import ttest_rel
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from domain_mf.initializers import calibrate_logit_scale, initialise_model
from domain_mf.interpretability import capture_anchors
from domain_mf.models import build_model
from domain_mf.trainer import evaluate, train_model


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
CLASSES = ("C", "E", "I", "K", "L", "O", "P", "Q", "X", "Y")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sign-root", type=Path, default=PROJECT_ROOT / "data" / "sign_language_mnist"
    )
    parser.add_argument("--output-root", type=Path, default=ROOT / "sign_user_loso_results")
    parser.add_argument("--methods", nargs="+", choices=["kaiming", "lowrank_wmf"], default=["kaiming", "lowrank_wmf"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--covariance-rank", type=int, default=16)
    parser.add_argument("--shrinkage", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CroppedSignerDataset(Dataset):
    def __init__(self, records: list[dict]):
        self.records = records
        self.transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(record["path"]) as image:
            image = image.convert("RGB")
            image = image.crop(record["box"])
            tensor = self.transform(image)
        return tensor, record["label"]


def load_records(sign_root: Path) -> tuple[list[dict], list[int]]:
    source = sign_root / "Dataset"
    user_dirs = sorted(
        source.glob("user_*"), key=lambda path: int(path.name.split("_")[-1])
    )
    if not user_dirs:
        raise FileNotFoundError(f"No user directories found in {source}")
    class_to_label = {name: index for index, name in enumerate(CLASSES)}
    records = []
    users = []
    for user_dir in user_dirs:
        user = int(user_dir.name.split("_")[-1])
        users.append(user)
        metadata = user_dir / f"{user_dir.name}_loc.csv"
        with metadata.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                filename = Path(row["image"]).name
                class_name = filename[0]
                if class_name not in class_to_label:
                    continue
                image_path = user_dir / filename
                if not image_path.exists():
                    raise FileNotFoundError(image_path)
                records.append(
                    {
                        "path": image_path,
                        "user": user,
                        "class": class_name,
                        "label": class_to_label[class_name],
                        "box": (
                            int(row["top_left_x"]),
                            int(row["top_left_y"]),
                            int(row["bottom_right_x"]),
                            int(row["bottom_right_y"]),
                        ),
                    }
                )
    expected = len(users) * len(CLASSES) * 10
    if len(records) != expected:
        raise RuntimeError(f"Expected {expected} selected images, found {len(records)}")
    return records, users


def make_loader(records, batch_size, shuffle, seed, num_workers, pin_memory):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        CroppedSignerDataset(records),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


def post_training_aulc(history_path: Path) -> float:
    with history_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    values = [float(row["val_accuracy"]) for row in rows if int(row["epoch"]) >= 1]
    return float(np.mean(values))


def mean_ci(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    half = (
        float(student_t.ppf(0.975, values.size - 1) * values.std(ddof=1) / math.sqrt(values.size))
        if values.size > 1
        else 0.0
    )
    return mean, mean - half, mean + half


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict]):
    metrics = (
        "initial_validation_accuracy",
        "initial_test_accuracy",
        "post_training_aulc",
        "final_test_accuracy",
    )
    aggregate_rows = []
    paired_rows = []
    by_method = {
        method: {int(row["test_user"]): row for row in rows if row["method"] == method}
        for method in sorted({row["method"] for row in rows})
    }
    for method, method_rows in by_method.items():
        for metric in metrics:
            values = np.array([float(method_rows[user][metric]) for user in sorted(method_rows)])
            mean, low, high = mean_ci(values)
            aggregate_rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "folds": values.size,
                    "mean": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    if {"kaiming", "lowrank_wmf"}.issubset(by_method):
        users = sorted(set(by_method["kaiming"]) & set(by_method["lowrank_wmf"]))
        for metric in metrics:
            reference = np.array([float(by_method["kaiming"][user][metric]) for user in users])
            method = np.array([float(by_method["lowrank_wmf"][user][metric]) for user in users])
            delta = method - reference
            mean, low, high = mean_ci(delta)
            paired_rows.append(
                {
                    "comparison": "lowrank_wmf_minus_kaiming",
                    "metric": metric,
                    "paired_users": len(users),
                    "kaiming_mean": float(reference.mean()),
                    "lowrank_mean": float(method.mean()),
                    "mean_paired_difference": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "p_value_raw": float(ttest_rel(method, reference).pvalue),
                    "lowrank_wins": int((delta > 0).sum()),
                }
            )
    return aggregate_rows, paired_rows


def main():
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    records, users = load_records(args.sign_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []

    for fold, test_user in enumerate(users):
        validation_user = users[(fold + 1) % len(users)]
        training_users = [user for user in users if user not in {test_user, validation_user}]
        train_records = [row for row in records if row["user"] in training_users]
        val_records = [row for row in records if row["user"] == validation_user]
        test_records = [row for row in records if row["user"] == test_user]
        fold_seed = 200 + fold

        for method in args.methods:
            output_dir = args.output_root / f"test_user_{test_user:02d}" / method
            metrics_path = output_dir / "final_metrics.json"
            summary_path = output_dir / "loso_summary.json"
            if args.resume and metrics_path.exists() and summary_path.exists():
                with summary_path.open(encoding="utf-8") as handle:
                    rows.append(json.load(handle))
                print(f"reuse test_user={test_user} method={method}")
                continue

            seed_everything(fold_seed)
            common = (args.batch_size, args.num_workers, device.type == "cuda")
            train_loader = make_loader(train_records, common[0], True, fold_seed, common[1], common[2])
            fit_loader = make_loader(train_records, common[0], False, fold_seed, common[1], common[2])
            val_loader = make_loader(val_records, common[0], False, fold_seed, common[1], common[2])
            test_loader = make_loader(test_records, common[0], False, fold_seed, common[1], common[2])

            model = build_model("standard", "sign").to(device)
            report = initialise_model(
                model,
                fit_loader,
                method,
                device,
                layers=3,
                shrinkage=args.shrinkage,
                covariance_rank=args.covariance_rank,
            )
            report.output_scale = calibrate_logit_scale(model, fit_loader, device, target_std=1.0)
            initial_test = evaluate(model, test_loader, device)["accuracy"]
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
                args.learning_rate,
                0.1,
                "none",
                lr_scheduler="cosine",
                min_learning_rate=1e-5,
            )
            row = {
                "fold": fold,
                "test_user": test_user,
                "validation_user": validation_user,
                "training_users": ";".join(map(str, training_users)),
                "method": method,
                "seed": fold_seed,
                "train_images": len(train_records),
                "validation_images": len(val_records),
                "test_images": len(test_records),
                "initial_validation_accuracy": metrics["initial_val_accuracy"],
                "initial_test_accuracy": initial_test,
                "inclusive_aulc": metrics["validation_aulc_0T"],
                "post_training_aulc": post_training_aulc(output_dir / "history.csv"),
                "best_validation_accuracy": metrics["best_val_accuracy"],
                "best_epoch": metrics["best_epoch"],
                "final_test_accuracy": metrics["test_accuracy"],
                "training_seconds": metrics["training_seconds"],
                "device": str(device),
                "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            }
            configuration = {
                **row,
                "classes": CLASSES,
                "initialisation_report": asdict(report),
                "source_root": str((args.sign_root / "Dataset").resolve()),
                "split_rule": "cyclic seven-fold: test user i, validation user i+1, five remaining train",
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "scheduler": "cosine",
                "covariance_rank": args.covariance_rank,
                "shrinkage": args.shrinkage,
            }
            with summary_path.open("w", encoding="utf-8") as handle:
                json.dump(row, handle, indent=2)
            with (output_dir / "loso_config.json").open("w", encoding="utf-8") as handle:
                json.dump(configuration, handle, indent=2)
            rows.append(row)
            print(json.dumps(row, indent=2))
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    aggregate_rows, paired_rows = aggregate(rows)
    write_csv(args.output_root / "sign_user_loso_rows.csv", rows)
    write_csv(args.output_root / "sign_user_loso_aggregate.csv", aggregate_rows)
    write_csv(args.output_root / "sign_user_loso_paired.csv", paired_rows)
    with (args.output_root / "evaluation_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "users": users,
                "classes": CLASSES,
                "folds": len(users),
                "source_repository": "https://github.com/mon95/Sign-Language-and-Static-gesture-recognition-using-sklearn",
                "source_license": "No explicit licence file was identified in the source repository.",
                "device": str(device),
                "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
