"""Train the published Correncoder on all 42 CapnoBase subjects for transfer."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from domain_mf.models import PublishedCorrEncoder1D
from run_correncoder_regression import (
    ROOT,
    _non_overlapping_segments,
    load_capnobase_subjects,
    seed_everything,
)


def parse_args():
    parser = argparse.ArgumentParser(description="CapnoBase pretraining for Correncoder")
    parser.add_argument("--capnobase-root", type=Path, default=ROOT / "data_capnobase")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs_correncoder_pretrain"
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=55)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    seed_everything(args.seed)
    subjects = load_capnobase_subjects(args.capnobase_root)
    inputs = torch.from_numpy(
        np.concatenate([_non_overlapping_segments(item["ppg"]) for item in subjects])[:, None, :]
    )
    targets = torch.from_numpy(
        np.concatenate(
            [_non_overlapping_segments(item["respiration"]) for item in subjects]
        )[:, None, :]
    )
    loader = DataLoader(
        TensorDataset(inputs, targets),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model = PublishedCorrEncoder1D(dropout=0.5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history = []
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for batch_inputs, batch_targets in loader:
            batch_inputs = batch_inputs.to(device, non_blocking=device.type == "cuda")
            batch_targets = batch_targets.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch_inputs)
            loss = F.mse_loss(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * batch_targets.numel()
            count += batch_targets.numel()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        history.append(
            {
                "epoch": epoch,
                "train_mse": total / count,
                "elapsed_seconds": time.perf_counter() - start,
            }
        )
        print(f"epoch={epoch:03d}/{args.epochs} train_mse={total / count:.6f}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output_dir / "checkpoint_capnobase_all.pt")
    with (args.output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    metadata = {
        "protocol": "published_correncoder_capnobase_all_subject_pretraining",
        "subjects": len(subjects),
        "segments": int(inputs.shape[0]),
        "segment_samples": int(inputs.shape[-1]),
        "sampling_hz": 30,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "final_train_mse": history[-1]["train_mse"],
        "training_seconds": history[-1]["elapsed_seconds"],
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
