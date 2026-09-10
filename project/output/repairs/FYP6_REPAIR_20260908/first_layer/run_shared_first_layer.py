"""Opt-in replacement of the historical 140-run stem-only comparison.

All seven stems receive bitwise-identical Conv2 and raw classifier parameters
for each dataset/seed. The original method-specific output calibration remains;
this is a stem-prior-plus-calibration comparison, not equal calibrated heads.
The old trainer/data/initializer source is frozen in adjacent source/.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = HERE / "source"
sys.path.insert(0, str(SOURCE))
import numpy as np
import torch
from scipy.stats import t, ttest_1samp
from domain_mf.data import build_loaders
from domain_mf.initializers import calibrate_logit_scale, initialise_model
from domain_mf.interpretability import capture_anchors
from domain_mf.models import build_model
from domain_mf.trainer import train_model
from run_experiment import seed_everything

METHODS = ["kaiming", "random_stem", "gabor", "pca", "kmeans", "di_wmf", "lowrank_wmf"]
DATASETS = ["fashion", "cifar10", "sign"]
DEFAULT_OUTPUT = ROOT / "outputs_first_layer_shared_20260908"
LATER = ("conv2.weight", "conv2.bias", "classifier.weight", "classifier.bias")


def dump(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def digest_tensor(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def state_hashes(state):
    return {name: digest_tensor(value) for name, value in state.items()}


def source_hashes():
    files = [Path(__file__), *sorted(SOURCE.rglob("*.py"))]
    return {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def configurations():
    configs = []
    for path in sorted((ROOT / "outputs_first_layer_cross_task").glob("*/config.json")):
        cfg = json.loads(path.read_text(encoding="utf-8"))
        assert cfg["model"] == "standard" and cfg["init_layers"] == 1
        assert cfg["epochs"] == 10 and cfg["train_fraction"] == 0.1
        assert cfg["batch_size"] == 512 and cfg["num_workers"] == 0
        assert cfg["learning_rate"] == 0.001 and cfg.get("lr_scheduler", "none") == "none"
        assert cfg.get("augmentation", "none") == "none"
        cfg["original_config_path"] = str(path.resolve())
        cfg["original_config_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        configs.append(cfg)
    assert len(configs) == 140
    keys = {(c["dataset"], c["seed"], c["init"]) for c in configs}
    expected = {(ds, seed, method) for ds in DATASETS
                for seed in range(10 if ds == "sign" else 5) for method in METHODS}
    assert keys == expected
    return sorted(configs, key=lambda c: (DATASETS.index(c["dataset"]), c["seed"], METHODS.index(c["init"])))


def shared_later_state(dataset, seed, device):
    """Use the original Kaiming branch, in an isolated torch RNG context."""
    devices = [device.index if device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        donor = build_model("standard", dataset).to(device)
        initialise_model(donor, None, "kaiming", device, layers=1)
        state = {name: donor.state_dict()[name].detach().cpu().clone() for name in LATER}
    return state


class RecordingSampler:
    """Record actual training index order without changing the wrapped sampler."""
    def __init__(self, sampler, original_indices):
        self.sampler = sampler
        self.original_indices = original_indices
        self.epochs = []

    def __len__(self):
        return len(self.sampler)

    def __iter__(self):
        seen = []
        for index in self.sampler:
            seen.append(int(self.original_indices[index]))
            yield index
        values = np.asarray(seen, dtype="<i8")
        self.epochs.append({"epoch": len(self.epochs) + 1, "count": len(seen),
                            "original_index_sha256": hashlib.sha256(values.tobytes()).hexdigest()})


def prepare(cfg, device):
    seed_everything(cfg["seed"])
    loaders = build_loaders(cfg["dataset"], Path(cfg["data_root"]), Path(cfg["sign_root"]),
                            cfg["batch_size"], cfg["seed"], cfg["train_fraction"],
                            cfg["val_fraction"], cfg["num_workers"], pin_memory=True,
                            augmentation=cfg.get("augmentation", "none"))
    train_loader, fit_loader, _, _ = loaders
    model = build_model("standard", cfg["dataset"]).to(device)
    report = initialise_model(model, fit_loader, cfg["init"], device, layers=1,
                              shrinkage=cfg["shrinkage"], covariance_rank=cfg["covariance_rank"])
    old_stem = torch.load(Path(cfg["original_config_path"]).parent / "filters_initial.pt",
                          map_location="cpu", weights_only=True)["conv1"]
    stem_reproduced = torch.equal(model.conv1.weight.detach().cpu(), old_stem)
    common = shared_later_state(cfg["dataset"], cfg["seed"], device)
    with torch.no_grad():
        state = model.state_dict()
        for key in LATER:
            state[key].copy_(common[key].to(device))
    actual_common = {name: model.state_dict()[name].detach().cpu().clone() for name in LATER}
    assert all(torch.equal(common[name], actual_common[name]) for name in LATER)
    stem_before_calibration = model.conv1.weight.detach().cpu().clone()
    report.output_scale = calibrate_logit_scale(model, fit_loader, device, target_std=cfg["logit_std"])
    assert torch.equal(model.conv1.weight.detach().cpu(), stem_before_calibration)
    assert torch.equal(model.conv2.weight.detach().cpu(), common["conv2.weight"])
    for name in ("classifier.weight", "classifier.bias"):
        assert torch.allclose(model.state_dict()[name].detach().cpu(), common[name] * report.output_scale)
    sampler = RecordingSampler(train_loader.sampler, train_loader.dataset.indices)
    train_loader.batch_sampler.sampler = sampler
    return model, loaders, report, common, sampler, stem_reproduced


def ci(values):
    a = np.asarray(values, dtype=float)
    mean = float(a.mean())
    half = float(t.ppf(.975, len(a) - 1) * a.std(ddof=1) / math.sqrt(len(a)))
    return mean, mean - half, mean + half


def csv_write(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(output):
    rows, groups, pair_groups, audit_groups = [], {}, {}, {}
    for completion in sorted(output.glob("*/COMPLETE.json")):
        folder = completion.parent
        cfg = json.loads((folder / "config.json").read_text())
        metrics = json.loads((folder / "final_metrics.json").read_text())
        fairness = json.loads((folder / "fairness_audit.json").read_text())
        with (folder / "history.csv").open(newline="") as f:
            history = list(csv.DictReader(f))
        post = [float(h["val_accuracy"]) for h in history if int(h["epoch"]) > 0]
        assert len(post) == 10 and len(fairness["training_orders"]) == 10
        assert abs(np.mean(post) - metrics["validation_aulc_1T"]) < 1e-10
        row = {"dataset": cfg["dataset"], "method": cfg["init"], "model_seed": cfg["seed"],
               "initial_val_accuracy": metrics["initial_val_accuracy"],
               "validation_aulc": float(np.mean(post)),
               "validation_aulc_0T": metrics["validation_aulc_0T"],
               "test_accuracy": metrics["test_accuracy"], "best_epoch": metrics["best_epoch"],
               "conv1_drift": metrics["kernel_drift"]["conv1"]["mean"],
               "training_seconds": metrics["training_seconds"], "output_scale": cfg["initialisation_report"]["output_scale"],
               "original_stem_reproduced": fairness["original_stem_reproduced"], "run": folder.name}
        rows.append(row)
        groups.setdefault((row["dataset"], row["method"]), []).append(row)
        pair_groups.setdefault(row["dataset"], {})[(row["method"], row["model_seed"])] = row
        audit_groups.setdefault((row["dataset"], row["model_seed"]), []).append(fairness)
    for group in audit_groups.values():
        assert all(g["shared_raw_later_sha256"] == group[0]["shared_raw_later_sha256"] for g in group)
        assert all(g["training_orders"] == group[0]["training_orders"] for g in group)
        assert all(g["train_indices_sha256"] == group[0]["train_indices_sha256"] for g in group)
        assert all(g["validation_indices_sha256"] == group[0]["validation_indices_sha256"] for g in group)
        assert all(g["training_rng_sha256"] == group[0]["training_rng_sha256"] for g in group)
    aggregates, paired = [], []
    keys = ["initial_val_accuracy", "validation_aulc", "test_accuracy", "conv1_drift"]
    for (dataset, method), group in sorted(groups.items()):
        agg = {"dataset": dataset, "method": method, "runs": len(group), "device_resolved": "cuda"}
        for key in keys:
            vals = [g[key] for g in group]
            mean, low, high = ci(vals) if len(vals) > 1 else (vals[0], vals[0], vals[0])
            agg.update({f"mean_{key}": mean, f"std_{key}": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0,
                        f"ci95_low_{key}": low, f"ci95_high_{key}": high})
        aggregates.append(agg)
    for dataset, lookup in sorted(pair_groups.items()):
        for method in METHODS[1:]:
            seeds = sorted({seed for met, seed in lookup if met == method and ("kaiming", seed) in lookup})
            if len(seeds) < 2:
                continue
            for key in keys:
                differences = [lookup[(method, s)][key] - lookup[("kaiming", s)][key] for s in seeds]
                mean, low, high = ci(differences)
                p = float(ttest_1samp(differences, 0).pvalue)
                if not np.isfinite(p):
                    p = 1.0 if mean == 0 else 0.0
                paired.append({"dataset": dataset, "reference": "kaiming", "method": method, "metric": key,
                               "paired_seeds": len(seeds), "mean_paired_difference": mean,
                               "ci95_low": low, "ci95_high": high, "raw_p": p,
                               "holm_family": "repair_all_stem_comparisons_all_tasks_all_four_metrics",
                               "analysis_status": "exploratory_repair_stage"})
    ordered = sorted(range(len(paired)), key=lambda i: paired[i]["raw_p"])
    running = 0.0
    for rank, index in enumerate(ordered):
        running = max(running, min(1.0, paired[index]["raw_p"] * (len(ordered) - rank)))
        paired[index]["holm_p_all_tests"] = running
        paired[index]["significant_holm_05"] = running < .05
    csv_write(output / "rows.csv", rows)
    csv_write(output / "aggregate.csv", aggregates)
    csv_write(output / "paired_training.csv", paired)
    summary = {"complete_runs": len(rows), "expected_runs": 140, "paired_comparisons": len(paired),
               "all_common_raw_later_states_match": True, "all_actual_training_orders_match": True,
               "all_recorded_training_rng_states_match": True,
               "all_original_stems_reproduced": all(r["original_stem_reproduced"] for r in rows),
               "training_seconds_total": sum(r["training_seconds"] for r in rows),
               "holm_scope": "all 72 repaired stem-versus-Kaiming contrasts; exploratory, not preregistered",
               "scientific_scope": "shared raw downstream weights, varying stem prior plus method-specific output calibration"}
    dump(output / "summary.json", summary)
    return summary


def run_one(cfg, output, device):
    folder = output / Path(cfg["original_config_path"]).parent.name
    if (folder / "COMPLETE.json").exists():
        return "reused_completed_new_run", 0.0
    folder.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with (folder / "run.log").open("w", encoding="utf-8", buffering=1) as log:
        with contextlib.redirect_stdout(log):
            model, loaders, report, common, recorder, reproduced = prepare(cfg, device)
            train, fit, val, test = loaders
            anchors = capture_anchors(model, 1)
            torch.save(common, folder / "shared_raw_later_state.pt")
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, folder / "model_initial_calibrated.pt")
            repair_cfg = dict(cfg)
            repair_cfg.update({"output_root": str(output), "initialisation_report": asdict(report),
                               "initialisation_seconds": time.perf_counter() - started,
                               "strict_shared_later_state": True, "shared_later_donor": "original Kaiming branch for same dataset/seed",
                               "training_rng_reset": True, "persistent_workers": False, "num_workers": 0,
                               "repair_scope": "stem prior plus independent global-logit calibration",
                               "source_sha256": source_hashes(), "python": platform.python_version(),
                               "torch": torch.__version__, "cuda": torch.version.cuda,
                               "cuda_device": torch.cuda.get_device_name(device),
                               "train_samples": len(train.dataset), "validation_samples": len(val.dataset), "test_samples": len(test.dataset)})
            dump(folder / "config.json", repair_cfg)
            seed_everything(cfg["seed"])
            rng = {"cpu": digest_tensor(torch.get_rng_state()), "cuda": digest_tensor(torch.cuda.get_rng_state(device)),
                   "train_loader": digest_tensor(train.generator.get_state())}
            metrics = train_model(model, train, val, test, device, anchors, folder, cfg["epochs"],
                                  cfg["learning_rate"], cfg["reg_lambda"], cfg["reg_schedule"],
                                  cfg.get("sparsity_lambda", 0), cfg.get("sparsity_mode", "l1"),
                                  cfg.get("auxiliary_lambda", 0), cfg.get("lr_scheduler", "none"), cfg.get("min_learning_rate", 1e-5))
            fairness = {"shared_raw_later_sha256": state_hashes(common), "training_orders": recorder.epochs,
                        "train_indices_sha256": digest_tensor(torch.tensor(train.dataset.indices, dtype=torch.int64)),
                        "validation_indices_sha256": digest_tensor(torch.tensor(val.dataset.indices, dtype=torch.int64)),
                        "training_rng_sha256": rng, "original_stem_reproduced": reproduced,
                        "output_scale": report.output_scale}
            dump(folder / "fairness_audit.json", fairness)
            assert len(recorder.epochs) == cfg["epochs"]
            dump(folder / "COMPLETE.json", {"finished_utc": datetime.now(timezone.utc).isoformat(),
                                           "wall_seconds": time.perf_counter() - started,
                                           "training_seconds": metrics["training_seconds"]})
    del model, loaders, common, recorder
    gc.collect()
    return "completed", time.perf_counter() - started


def self_test(configs, device):
    tests, hashes = [], None
    for cfg in configs:
        if cfg["dataset"] != "fashion" or cfg["seed"] != 0:
            continue
        model, loaders, report, common, sampler, reproduced = prepare(cfg, device)
        current = state_hashes(common)
        if hashes is None:
            hashes = current
        assert current == hashes
        assert reproduced, f"Original stem changed: {cfg['init']}"
        tests.append({"method": cfg["init"], "same_raw_downstream": True,
                      "original_stem_reproduced": reproduced, "calibration_scale": report.output_scale})
        del model, loaders
        gc.collect()
    assert len(tests) == 7
    dump(HERE / "fairness_self_test.json", {"passed": True, "device": str(device), "tests": tests,
                                          "common_downstream_sha256": hashes, "source_sha256": source_hashes()})
    print(json.dumps({"self_test_passed": True, "methods": 7}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["self-test", "smoke", "run", "summarize"], default="run")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.mode == "summarize":
        print(json.dumps(summarize(output), indent=2))
        return
    assert torch.cuda.is_available(), "Formal repair requires the original CUDA runtime"
    device = torch.device("cuda")
    configs = configurations()
    if args.mode == "self-test":
        self_test(configs, device)
        return
    if args.mode == "smoke":
        configs = [c for c in configs if c["seed"] == 0 and c["init"] == "kaiming"]
    for i, cfg in enumerate(configs, 1):
        status, seconds = run_one(cfg, output, device)
        print(f"{i}/{len(configs)} {cfg['dataset']} {cfg['init']} seed={cfg['seed']} {status} wall={seconds:.2f}s", flush=True)
        dump(output / "queue_status.json", {"last_dataset": cfg["dataset"], "last_method": cfg["init"],
                                           "last_seed": cfg["seed"], "updated_utc": datetime.now(timezone.utc).isoformat(),
                                           "position": i, "queue_length": len(configs), "last_status": status})
        summarize(output)
    print(json.dumps(summarize(output), indent=2), flush=True)


if __name__ == "__main__":
    main()
