"""Read-only audit of the saved regression runs used by the thesis.

Usage: python verify_regression_records.py --root PATH_TO_domain_mf_v2 --out DIR
Optional: --replay-baseline-checkpoints (CPU inference, no training; requires BIDMC MAT).
The manifests are created at audit time. They are not run-time source snapshots.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import t

EXPERIMENTS = {
    "pretrained_mse": "outputs_correncoder_regression_full",
    "random_mse": "outputs_correncoder_initialisation/random_mse",
    "matched_1d_mse": "outputs_correncoder_initialisation/matched_1d_mse",
    "pretrained_corr": "outputs_correncoder_loss_ablation/pretrained_corr",
    "pretrained_spectral": "outputs_correncoder_loss_ablation/pretrained_spectral",
    "pretrained_corr_spectral": "outputs_correncoder_loss_ablation/pretrained_corr_spectral",
    "matched_layerwise_calibrated_mse": "outputs_correncoder_extensions/matched_layerwise_calibrated_mse",
    "pretrained_hard_lag": "outputs_correncoder_extensions/pretrained_lag_corr",
    "pretrained_soft_lag": "outputs_correncoder_extensions/pretrained_soft_lag_corr",
    "depth1": "outputs_correncoder_depth_ste/depth1",
    "depth2_diagonal": "outputs_correncoder_depth_ste/depth2_diagonal",
    "depth3_diagonal": "outputs_correncoder_depth_ste/depth3_diagonal",
    "depth3_lowrank": "outputs_correncoder_depth_ste/depth3_lowrank",
}
METRICS = ("test_mse", "test_mae", "waveform_correlation", "rr_mean_absolute_error_bpm_30p6s")


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(t.ppf(.975, len(values)-1) * values.std(ddof=1) / math.sqrt(len(values)))


def run_audit(root, out):
    out.mkdir(parents=True, exist_ok=True)
    manifest, fold_rows, summaries, aggregates, issues = [], [], [], [], []
    def record(path, kind):
        if not path.exists():
            issues.append(f"Missing {path.relative_to(root)}")
            return
        stat = path.stat()
        manifest.append({"path": path.relative_to(root).as_posix(), "kind": kind,
                         "bytes": stat.st_size, "sha256": digest(path),
                         "filesystem_mtime_utc_not_preregistration": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()})

    for relative in ("run_correncoder_regression.py", "pretrain_correncoder_capnobase.py", "domain_mf/regression.py", "domain_mf/models.py"):
        record(root / relative, "source_at_audit_time_not_run_pinned")
    for label, relative in EXPERIMENTS.items():
        path = root / relative
        config_path = path / "experiment_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        if config:
            record(config_path, "saved_experiment_configuration")
        master = read_csv(path / "fold_metrics.csv")
        record(path / "fold_metrics.csv", "saved_fold_metrics")
        master_by_fold = {int(row["fold"]): row for row in master}
        folds = sorted(path.glob("bidmc_fold*_seed*/final_metrics.json"))
        if len(folds) != 53:
            issues.append(f"{label}: {len(folds)} fold metric files, expected 53")
        local_rows = []
        for final_path in folds:
            folder = final_path.parent
            final = json.loads(final_path.read_text(encoding="utf-8"))
            history_path = folder / "history.csv"
            history = read_csv(history_path)
            epochs = [int(row["epoch"]) for row in history]
            for name, kind in (("final_metrics.json", "saved_final_metrics"), ("history.csv", "saved_training_history"),
                               ("checkpoint_final.pt", "saved_final_checkpoint"), ("waveforms.npz", "saved_test_waveforms")):
                record(folder / name, kind)
            if epochs != list(range(1, 81)):
                issues.append(f"{label}/{folder.name}: epochs not exactly 1..80")
            validation_fields = [key for key in history[0] if "val" in key.lower() or "test" in key.lower()]
            if validation_fields:
                issues.append(f"{label}/{folder.name}: unexpected validation/test history fields {validation_fields}")
            if final["train_segments"] != 2600 or final["test_segments"] != 471:
                issues.append(f"{label}/{folder.name}: unexpected segment counts")
            corr = final.get("correlation_lambda", config.get("correlation_lambda", 0.0))
            spec = final.get("spectral_lambda", config.get("spectral_lambda", 0.0))
            errs = [abs(float(h["train_objective"]) - float(h["train_mse"])
                        - corr * float(h["train_correlation_loss"]) - spec * float(h["train_spectral_loss"]))
                    for h in history if "train_objective" in h]
            objective_error = max(errs) if errs else None
            if objective_error is not None and objective_error > 1e-6:
                issues.append(f"{label}/{folder.name}: objective identity differs by {objective_error}")
            baseline_row = master_by_fold[int(final["fold"])]
            metric_delta = max(abs(float(final[metric]) - float(baseline_row[metric])) for metric in METRICS)
            if metric_delta > 1e-10:
                issues.append(f"{label}/{folder.name}: master/final metric difference {metric_delta}")
            row = {"experiment": label, "directory": folder.relative_to(root).as_posix(),
                   "fold_zero_based": final["fold"], "test_subject_one_based": final["test_subject"],
                   "seed": final["seed"], "history_epochs": len(history), "first_epoch": epochs[0],
                   "last_epoch": epochs[-1], "evaluation_epoch_from_end_of_training_records": epochs[-1],
                   "validation_split": "none", "validation_metric": "none", "validation_history_fields": ";".join(validation_fields),
                   "checkpoint_rule": "final_epoch_not_validation_selected",
                   "train_segments": final["train_segments"], "test_segments": final["test_segments"],
                   "correlation_lambda": corr, "spectral_lambda": spec,
                   "objective_identity_max_abs_error": objective_error,
                   "fold_csv_vs_final_metrics_max_abs_error": metric_delta,
                   **{metric: final[metric] for metric in METRICS}}
            fold_rows.append(row)
            local_rows.append(row)
        for metric in METRICS:
            mean, half = mean_ci([row[metric] for row in local_rows])
            aggregates.append({"experiment": label, "metric": metric, "n": len(local_rows), "mean": mean,
                               "ci95_half_width": half, "ci95_low": mean-half, "ci95_high": mean+half})
        summaries.append({"experiment": label, "path": relative, "folds": len(local_rows),
                          "configuration_file_available": bool(config),
                          "observed_last_epochs": ";".join(map(str, sorted({row["last_epoch"] for row in local_rows}))),
                          "spectral_lambdas": ";".join(map(str, sorted({row["spectral_lambda"] for row in local_rows}))),
                          "correlation_lambdas": ";".join(map(str, sorted({row["correlation_lambda"] for row in local_rows}))),
                          "validation_fields_present": any(row["validation_history_fields"] for row in local_rows),
                          "checkpoint_rule": "final_epoch_not_validation_selected",
                          "run_time_source_hash_saved": False})

    pretrain_root = root / "outputs_correncoder_pretrain"
    pretrain = json.loads((pretrain_root / "metadata.json").read_text())
    pretrain_history = read_csv(pretrain_root / "history.csv")
    for name in ("metadata.json", "history.csv", "checkpoint_capnobase_all.pt"):
        record(pretrain_root/name, "saved_capnobase_pretraining_record")
    if pretrain["epochs"] != 80 or [int(row["epoch"]) for row in pretrain_history] != list(range(1,81)):
        issues.append("CapnoBase pretraining is not exactly 80 epochs")
    pretrain.update({"checkpoint_rule": "single_final_epoch80_checkpoint_not_validation_selected",
                     "validation_split": "none", "validation_metric": "none",
                     "bidmc_subjects_used": 0,
                     "checkpoint_sha256": digest(pretrain_root / "checkpoint_capnobase_all.pt")})
    spectral = {
        "input_length": 288, "sampling_hz": 30, "requested_band_hz": [0.08, 0.8],
        "fft_bin_indices": list(range(1,8)), "fft_bin_frequencies_hz": [k*30/288 for k in range(1,8)],
        "K": 7, "window": "non-periodic Hann after subtracting each waveform mean",
        "spectrum": "rFFT absolute magnitude, not power, normalized by each waveform's in-band magnitude sum",
        "denominator_floor": 1e-8,
        "reduction": "mean over batch, single channel, and seven selected bins; F.l1_loss default reduction='mean'",
        "active_spectral_lambda": 0.1,
        "equivalent_coefficient_for_a_per_example_sum_over_7_bins": 0.1/7,
        "spectral_only_experiment": EXPERIMENTS["pretrained_spectral"],
        "correlation_and_spectral_experiment": EXPERIMENTS["pretrained_corr_spectral"],
        "provenance_limit": "The original configs and per-epoch objective components persist; no run-time source hash or full source snapshot is attached to these runs. Code reduction is verified against the available source at audit time and can be exercised by the optional numerical check, not represented as an authenticated historical source snapshot."
    }
    payload = {"created_utc": datetime.now(timezone.utc).isoformat(), "kind": "retrospective_records_audit",
               "experiments": len(summaries), "fold_records": len(fold_rows), "issues": issues,
               "pretrain": pretrain, "spectral_definition": spectral,
               "interpretation": "All archived reported regression histories end at epoch 80, contain training-only metrics, and accompany final checkpoints. The supplied trainer evaluates only after the loop. No validation selection was implemented or recorded. This conclusion does not establish that later exploratory design choices never consulted earlier test results.",
               "provenance_limit": "No original version-controlled per-run source snapshot was found. A SHA256 made now identifies this audit's inputs, not the source version at execution time. The oldest pretrained-MSE family has no experiment_config.json; its protocol, histories, per-fold records and pretraining metadata are retained."
               }
    write_csv(out/"regression_fold_checkpoint_registry.csv", fold_rows)
    write_csv(out/"regression_experiment_registry.csv", summaries)
    write_csv(out/"regression_recomputed_aggregates.csv", aggregates)
    write_csv(out/"regression_original_record_manifest.csv", manifest)
    (out/"regression_audit_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"experiments": len(summaries), "fold_records": len(fold_rows), "issues": issues, "out": str(out)}, indent=2))
    return payload


def check_spectral_implementation(root, out):
    import torch
    sys.path.insert(0, str(root))
    from domain_mf.regression import spectral_shape_loss, correncoder_regression_loss
    x = torch.arange(288, dtype=torch.float64)/30
    prediction = torch.stack([torch.sin(2*math.pi*.3*x), torch.cos(2*math.pi*.4*x)])[:,None,:]
    target = torch.stack([torch.sin(2*math.pi*.5*x), torch.cos(2*math.pi*.2*x)])[:,None,:]
    window = .5 - .5*torch.cos(2*math.pi*torch.arange(288, dtype=torch.float64)/287)
    def spectrum(values):
        magnitudes = torch.fft.rfft((values-values.mean(dim=-1,keepdim=True))*window).abs()[...,1:8]
        return magnitudes/magnitudes.sum(dim=-1,keepdim=True).clamp_min(1e-8)
    delta = (spectrum(prediction)-spectrum(target)).abs()
    mean = delta.mean()
    total = delta.sum(dim=-1).mean()
    observed = spectral_shape_loss(prediction,target)
    objective, components = correncoder_regression_loss(prediction,target,correlation_lambda=.1,spectral_lambda=.1)
    results = {"numerical_test_is_audit_time_not_original_run": True,
               "implemented": observed.item(), "independent_mean": mean.item(),
               "independent_sum_over_bins_then_batch_mean": total.item(), "sum_to_mean_ratio": (total/mean).item(),
               "mean_matches": bool(torch.allclose(observed,mean,atol=1e-12,rtol=0)),
               "objective_matches": bool(torch.allclose(objective,components['mse']+.1*components['correlation_loss']+.1*mean,atol=1e-12,rtol=0))}
    assert results["mean_matches"] and results["objective_matches"]
    (out/"spectral_numerical_check.json").write_text(json.dumps(results,indent=2),encoding="utf-8")


def replay_baseline(root, out, replay_device):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    sys.path.insert(0, str(root))
    from domain_mf.models import PublishedCorrEncoder1D
    from run_correncoder_regression import load_bidmc_subjects, build_fold_tensors, _evaluate, overlap_average
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(replay_device)
    subjects = load_bidmc_subjects(root/"data_bidmc/bidmc_data.mat")
    rows = []
    for final_path in sorted((root/EXPERIMENTS["pretrained_mse"]).glob("bidmc_fold*/final_metrics.json")):
        metrics = json.loads(final_path.read_text())
        _, _, test_x, test_y, starts = build_fold_tensors(subjects,int(metrics["fold"]))
        model = PublishedCorrEncoder1D(dropout=.5).to(device)
        model.load_state_dict(torch.load(final_path.parent/"checkpoint_final.pt", map_location="cpu", weights_only=True))
        loader = DataLoader(TensorDataset(test_x,test_y),batch_size=30,shuffle=False)
        observed = _evaluate(model,loader,device)
        waveforms = np.load(final_path.parent/"waveforms.npz")
        prediction = overlap_average(observed["prediction"],starts)
        row = {"fold": metrics["fold"], "test_subject": metrics["test_subject"], "replay_device": replay_device,
               "saved_test_mse": metrics["test_mse"], "replayed_test_mse": observed["mse"],
               "saved_test_mae": metrics["test_mae"], "replayed_test_mae": observed["mae"],
               "prediction_max_absolute_difference": float(np.max(np.abs(prediction-waveforms["prediction"]))),
               "target_max_absolute_difference": float(np.max(np.abs(overlap_average(observed["target"],starts)-waveforms["target"])))}
        row["matches_with_cpu_gpu_tolerance_1e_4"] = max(abs(row["saved_test_mse"]-row["replayed_test_mse"]), abs(row["saved_test_mae"]-row["replayed_test_mae"]),row["prediction_max_absolute_difference"],row["target_max_absolute_difference"]) < 1e-4
        rows.append(row)
        print(f"Checkpoint replay {metrics['fold']+1}/53: {row['matches_with_cpu_gpu_tolerance_1e_4']}",flush=True)
    write_csv(out/f"regression_pretrained_mse_checkpoint_replay_{replay_device}.csv",rows)
    if not all(row["matches_with_cpu_gpu_tolerance_1e_4"] for row in rows):
        raise RuntimeError("One or more checkpoint replay results differ from archived outputs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--check-spectral",action="store_true")
    parser.add_argument("--replay-baseline-checkpoints",action="store_true")
    parser.add_argument("--replay-device",choices=("cpu", "cuda"),default="cpu")
    args = parser.parse_args()
    report = run_audit(args.root.resolve(),args.out.resolve())
    if args.check_spectral:
        check_spectral_implementation(args.root.resolve(),args.out.resolve())
    if args.replay_baseline_checkpoints:
        replay_baseline(args.root.resolve(),args.out.resolve(),args.replay_device)
    if report["issues"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
