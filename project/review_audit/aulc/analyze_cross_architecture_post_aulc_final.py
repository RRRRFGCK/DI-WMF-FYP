"""Build the final LOW-RANK cross-architecture AULC table from saved histories.

This corrected publication driver explicitly selects lowrank_wmf, rank 16,
for both architectures. The original script's ResNet18 diagonal selector is
preserved in the historical source snapshot; it is not used here.

Run: python analyze_cross_architecture_post_aulc_final.py --root EXPERIMENT_ROOT
     --output NEW_OUTPUT_DIRECTORY

Requires Python >=3.10, NumPy and SciPy. No GPU, training, or checkpoint loading.
The inputs are read only. Corrections are retrospective, not preregistered.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import t, ttest_rel

DATASETS = ("fashion", "cifar10", "sign")
ARCHITECTURES = ("resnet18", "correncoder")
REFERENCE = "kaiming"
CANDIDATE = "lowrank_wmf"
RANK = 16
SEEDS = tuple(range(5))
EPOCHS = 10
TRAIN_FRACTION = 0.1


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def holm(raw_pvalues):
    adjusted = [1.0] * len(raw_pvalues)
    running = 0.0
    for rank, index in enumerate(sorted(range(len(raw_pvalues)), key=raw_pvalues.__getitem__)):
        running = max(running, (len(raw_pvalues) - rank) * raw_pvalues[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    # Publication outputs must not replace the archived diagonal comparison.
    if output == root / "submission_control_results":
        raise ValueError("Choose a new output directory; retain submission_control_results as an original record.")
    output.mkdir(parents=True, exist_ok=True)
    run_rows, source_files, used = [], {}, {}

    for architecture in ARCHITECTURES:
        directory = root / f"outputs_{architecture}_cross_task"
        for config_path in sorted(directory.glob("*/config.json")):
            config = json.loads(config_path.read_text(encoding="utf-8"))
            dataset, method = config.get("dataset"), config.get("init")
            if (config.get("model") != architecture or dataset not in DATASETS
                    or method not in (REFERENCE, CANDIDATE)
                    or int(config.get("seed", -1)) not in SEEDS
                    or int(config.get("epochs", 0)) != EPOCHS
                    or float(config.get("train_fraction", -1)) != TRAIN_FRACTION):
                continue
            seed = int(config["seed"])
            if method == CANDIDATE and int(config.get("covariance_rank", -1)) != RANK:
                raise ValueError(f"Expected rank {RANK}: {config_path}")
            key = (architecture, dataset, method, seed)
            if key in used:
                raise ValueError(f"Duplicate formal run for {key}: {config_path}")
            history_path = config_path.parent / "history.csv"
            history = read_csv(history_path)
            epoch_numbers = [int(r["epoch"]) for r in history]
            if sorted(epoch_numbers) != list(range(EPOCHS + 1)):
                raise ValueError(f"Expected exactly one row per epoch 0..{EPOCHS}: {history_path}")
            by_epoch = {int(r["epoch"]): float(r["val_accuracy"]) for r in history}
            values = np.asarray([by_epoch[e] for e in range(1, EPOCHS + 1)], dtype=float)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"Non-finite validation accuracies: {history_path}")
            row = {
                "architecture": architecture, "dataset": dataset, "method": method,
                "covariance_rank": RANK if method == CANDIDATE else "not_applicable",
                "seed": seed, "train_fraction": TRAIN_FRACTION, "epochs": EPOCHS,
                "post_update_epoch_count": len(values), "metric": "AULC_1T",
                "aulc_1_to_10": float(values.mean()),
                "aulc_0_to_10_for_audit_only": float(np.mean(list(by_epoch.values()))),
                "config_path": config_path.relative_to(root).as_posix(),
                "history_path": history_path.relative_to(root).as_posix(),
            }
            run_rows.append(row)
            used[key] = row
            for path in (config_path, history_path):
                source_files[path.relative_to(root).as_posix()] = {
                    "sha256": sha(path), "bytes": path.stat().st_size,
                }

    expected_keys = {(a, d, m, s) for a in ARCHITECTURES for d in DATASETS
                     for m in (REFERENCE, CANDIDATE) for s in SEEDS}
    if set(used) != expected_keys:
        raise ValueError(f"Missing formal keys: {sorted(expected_keys - set(used))}; unexpected: {sorted(set(used) - expected_keys)}")
    paired_rows = []
    for architecture in ARCHITECTURES:
        architecture_rows = []
        for dataset in DATASETS:
            reference = np.array([used[(architecture, dataset, REFERENCE, s)]["aulc_1_to_10"] for s in SEEDS])
            candidate = np.array([used[(architecture, dataset, CANDIDATE, s)]["aulc_1_to_10"] for s in SEEDS])
            delta = candidate - reference
            mean, sd = float(delta.mean()), float(delta.std(ddof=1))
            half = float(t.ppf(0.975, len(SEEDS) - 1)) * sd / math.sqrt(len(SEEDS))
            raw_p = float(ttest_rel(candidate, reference).pvalue) if sd > 0 else (1.0 if mean == 0 else 0.0)
            architecture_rows.append({
                "architecture": architecture, "dataset": dataset,
                "method": CANDIDATE, "reference": REFERENCE,
                "covariance_rank": RANK, "paired_seeds": len(SEEDS),
                "seed_ids": json.dumps(SEEDS), "train_fraction": TRAIN_FRACTION,
                "epochs": EPOCHS, "metric": "AULC_1T", "epoch0_excluded": True,
                "kaiming_mean_aulc_1_to_10": float(reference.mean()),
                "di_mean_aulc_1_to_10": float(candidate.mean()),
                "mean_paired_difference": mean, "paired_difference_sd": sd,
                "ci95_low": mean - half, "ci95_high": mean + half,
                "p_value_raw": raw_p,
                "correction_scope": f"{architecture}|lowrank_wmf|AULC_1T|three_datasets",
                "scope_hypotheses": 3,
                "analysis_status": "retrospective_corrected_publication_analysis",
            })
        for row, adjusted in zip(architecture_rows, holm([r["p_value_raw"] for r in architecture_rows])):
            row["p_value_holm_across_tasks"] = adjusted
            row["holm_significant_0p05"] = adjusted < 0.05
        paired_rows.extend(architecture_rows)

    write_csv(output / "cross_architecture_post_aulc.csv", paired_rows)
    write_csv(output / "cross_architecture_post_aulc_run_rows.csv", run_rows)
    manifest = {
        "role": "Corrected publication analysis, not a run-time source archive",
        "candidate": CANDIDATE, "reference": REFERENCE, "rank": RANK,
        "epochs": EPOCHS, "train_fraction": TRAIN_FRACTION, "seeds": SEEDS,
        "formal_runs": len(run_rows),
        "runs_by_architecture": dict(Counter(r["architecture"] for r in run_rows)),
        "definition": "Per-run discrete mean validation accuracy over epochs 1..10, excluding epoch 0",
        "ci": "95% Student-t confidence interval of paired per-seed differences",
        "test": "Two-sided paired t test",
        "holm_scope": "Three datasets separately within each architecture for lowrank_wmf vs kaiming, AULC_1T only",
        "historical_correction": "Original ResNet18 post-AULC script selected diagonal di_wmf; publication comparison explicitly selects lowrank_wmf.",
        "script_sha256": sha(__file__), "original_inputs": source_files,
    }
    (output / "cross_architecture_post_aulc_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(paired_rows, indent=2))


if __name__ == "__main__":
    main()
