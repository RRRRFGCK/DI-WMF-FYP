"""Resumable forward-only evaluation of the new shared-downstream checkpoints.

Reuses frozen historical metric implementations and all original batch/noise
settings. A separate output directory prevents historical/new result mixing.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE / "source"))
import torch
from torch.utils.data import DataLoader
from domain_mf.models import build_model
from domain_mf.data import build_test_dataset
from domain_mf.corruptions import CORRUPTIONS


def module(name):
    spec = importlib.util.spec_from_file_location(name, HERE / "evaluation_source" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


corrupt = module("evaluate_corruptions")
selectivity = module("evaluate_filter_selectivity")
SEVERITIES = [.05, .1, .2, .3]
NOISE_SEEDS = [0, 1, 2]


def dump(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temp.replace(path)


def evaluate_one(folder, output, cache, device):
    destination = output / "per_checkpoint" / folder.name
    marker = destination / "COMPLETE.json"
    if marker.exists():
        return 0.0
    started = time.perf_counter()
    cfg = json.loads((folder / "config.json").read_text())
    ds = cfg["dataset"]
    if ds not in cache:
        dataset = build_test_dataset(ds, Path(cfg["data_root"]), Path(cfg["sign_root"]))
        cache[ds] = corrupt.materialize_dataset(dataset, 1024, 0, pin_memory=True)
    dataset = cache[ds]
    model = build_model("standard", ds).to(device)
    checkpoint = folder / "checkpoint_best.pt"
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    clean_predictions, targets, clean_accuracy = corrupt.predict(model, dataset, 1024, 0, device)
    original_metrics = json.loads((folder / "final_metrics.json").read_text())
    assert abs(clean_accuracy - original_metrics["test_accuracy"]) < 2e-5
    common = {"run": folder.name, "dataset": ds, "model": "standard", "method": cfg["init"],
              "model_seed": cfg["seed"], "clean_accuracy": clean_accuracy, "device": str(device)}
    rows = []
    for corruption in CORRUPTIONS:
        base = {**common, "corruption": corruption}
        rows.append({**base, "noise_seed": -1, "severity": 0.0, "corrupted_accuracy": clean_accuracy,
                     "accuracy_drop": 0.0, "retention": 1.0, "consistency": 1.0})
        for severity in SEVERITIES:
            for seed in NOISE_SEEDS:
                predictions, corrupted_targets, accuracy = corrupt.predict(
                    model, dataset, 1024, 0, device, dataset_name=ds,
                    corruption=corruption, severity=severity, noise_seed=seed)
                assert torch.equal(targets, corrupted_targets)
                rows.append({**base, "noise_seed": seed, "severity": severity, "corrupted_accuracy": accuracy,
                             "accuracy_drop": clean_accuracy - accuracy,
                             "retention": accuracy / clean_accuracy if clean_accuracy else 0.0,
                             "consistency": float((predictions == clean_predictions).float().mean())})
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0, pin_memory=True)
    sel = selectivity.selectivity_metrics(model, selectivity.conv1_class_means(model, loader, device))
    selector = {key: common[key] for key in ("run", "dataset", "model", "method", "model_seed", "device")}
    selector.update(sel)
    model_rows, _ = corrupt.summarise_models(rows, SEVERITIES)
    destination.mkdir(parents=True, exist_ok=True)
    corrupt._write_csv(destination / "corruption_rows.csv", rows)
    corrupt._write_csv(destination / "corruption_model_metrics.csv", model_rows)
    dump(destination / "selectivity.json", selector)
    source_files = [Path(__file__), *sorted((HERE / "evaluation_source").glob("*.py")),
                    *sorted((HERE / "source" / "domain_mf").glob("*.py"))]
    metadata = {"finished_utc": datetime.now(timezone.utc).isoformat(), "wall_seconds": time.perf_counter() - started,
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "checkpoint_path": str(checkpoint), "corruptions": CORRUPTIONS, "severities": SEVERITIES,
                "noise_seeds": NOISE_SEEDS, "corruption_batch_size": 1024, "selectivity_batch_size": 512,
                "num_workers": 0, "test_samples": len(dataset), "torch_version": torch.__version__,
                "device": torch.cuda.get_device_name(device),
                "source_sha256": {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}}
    dump(marker, metadata)
    return metadata["wall_seconds"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=ROOT / "outputs_first_layer_shared_20260908")
    parser.add_argument("--results-root", type=Path, default=ROOT / "first_layer_shared_results_20260908")
    parser.add_argument("--follow", action="store_true", help="Wait for current in-progress training queue, until all 140 are evaluated.")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    cache, device = {}, torch.device("cuda")
    args.results_root.mkdir(parents=True, exist_ok=True)
    completed_now = 0
    while True:
        ready = sorted(p.parent for p in args.runs_root.glob("*/COMPLETE.json"))
        pending = [p for p in ready if not (args.results_root / "per_checkpoint" / p.name / "COMPLETE.json").exists()]
        for folder in pending:
            seconds = evaluate_one(folder, args.results_root, cache, device)
            completed_now += 1
            finished = len(list((args.results_root / "per_checkpoint").glob("*/COMPLETE.json")))
            dump(args.results_root / "evaluation_status.json", {"complete_checkpoints": finished, "expected": 140,
                                                               "last_run": folder.name, "last_seconds": seconds,
                                                               "updated_utc": datetime.now(timezone.utc).isoformat()})
            print(f"{finished}/140 {folder.name} {seconds:.2f}s", flush=True)
            if args.limit is not None and completed_now >= args.limit:
                return
        finished = len(list((args.results_root / "per_checkpoint").glob("*/COMPLETE.json")))
        if finished == 140 or not args.follow:
            return
        time.sleep(5)


if __name__ == "__main__":
    main()
