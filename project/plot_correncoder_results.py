from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Plot full Correncoder LOSO results")
    parser.add_argument("--results-root", type=Path, default=Path("outputs_correncoder_regression_full"))
    parser.add_argument("--output-dir", type=Path, default=Path("correncoder_regression_results"))
    parser.add_argument("--sampling-hz", type=float, default=30.0)
    parser.add_argument("--waveform-seconds", type=float, default=30.6)
    return parser.parse_args()


def main():
    args = parse_args()
    with (args.results_root / "fold_metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("No Correncoder fold rows found")

    folds = np.asarray([int(row["fold"]) for row in rows])
    seeds = np.asarray([int(row["seed"]) for row in rows])
    rr_mae = np.asarray([float(row["rr_mean_absolute_error_bpm_30p6s"]) for row in rows])
    corr = np.asarray([float(row["waveform_correlation"]) for row in rows])

    median_rr = float(np.median(rr_mae))
    representative_index = int(np.argmin(np.abs(rr_mae - median_rr)))
    representative_fold = int(folds[representative_index])
    representative_seed = int(seeds[representative_index])
    waveform_path = args.results_root / f"bidmc_fold{representative_fold:02d}_seed{representative_seed}" / "waveforms.npz"
    waveform = np.load(waveform_path)
    prediction = waveform["prediction"]
    target = waveform["target"]
    samples = min(len(prediction), int(round(args.sampling_hz * args.waveform_seconds)))
    time_axis = np.arange(samples) / args.sampling_hz

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].scatter(folds + 1, rr_mae, s=26, alpha=0.8, color="#1f77b4")
    axes[0].axhline(np.mean(rr_mae), color="#d62728", linestyle="--", label=f"mean={np.mean(rr_mae):.2f}")
    axes[0].axhline(median_rr, color="#2ca02c", linestyle=":", label=f"median={median_rr:.2f}")
    axes[0].set(title="Respiratory-rate error", xlabel="Held-out BIDMC subject", ylabel="Mean absolute error (bpm)")
    axes[0].legend(frameon=False)

    colors = np.where(corr >= 0, "#1f77b4", "#d62728")
    axes[1].scatter(folds + 1, corr, s=26, alpha=0.8, c=colors)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].axhline(np.mean(corr), color="#2ca02c", linestyle="--", label=f"mean={np.mean(corr):.2f}")
    axes[1].set(title="Waveform reconstruction", xlabel="Held-out BIDMC subject", ylabel="Pearson correlation")
    axes[1].legend(frameon=False)

    axes[2].plot(time_axis, target[:samples], label="reference respiration", linewidth=1.8)
    axes[2].plot(time_axis, prediction[:samples], label="Correncoder output", linewidth=1.4, alpha=0.9)
    axes[2].set(title=f"Representative subject {representative_fold + 1}", xlabel="Time (s)", ylabel="Normalised amplitude")
    axes[2].legend(frameon=False)

    fig.suptitle("Full Correncoder: 53-fold BIDMC leave-one-subject-out evaluation", fontsize=13)
    fig.tight_layout()
    output_path = args.output_dir / "correncoder_loso_summary.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "folds": len(rows),
        "rr_mae_mean_bpm": float(np.mean(rr_mae)),
        "rr_mae_median_bpm": median_rr,
        "waveform_correlation_mean": float(np.mean(corr)),
        "waveform_correlation_median": float(np.median(corr)),
        "representative_fold_zero_based": representative_fold,
        "representative_subject_one_based": representative_fold + 1,
        "representative_seed": representative_seed,
        "representative_rr_mae_bpm": float(rr_mae[representative_index]),
        "output": str(output_path.resolve()),
    }
    with (args.output_dir / "plot_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
