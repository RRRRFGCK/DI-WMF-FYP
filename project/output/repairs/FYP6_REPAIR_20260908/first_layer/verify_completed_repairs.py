"""Independent read-only, no-GPU final checks on the actual repair records."""
import csv
import hashlib
import json
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
NEW = ROOT / "outputs_first_layer_shared_20260908"
EVAL = ROOT / "first_layer_shared_results_20260908"


def csv_rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main():
    baselines, runs, per_group = [], [], {}
    complete = sorted(NEW.glob("*/COMPLETE.json"))
    assert len(complete) == 140
    for marker in complete:
        folder = marker.parent
        cfg = json.loads((folder / "config.json").read_text())
        metrics = json.loads((folder / "final_metrics.json").read_text())
        fairness = json.loads((folder / "fairness_audit.json").read_text())
        initial = torch.load(folder / "model_initial_calibrated.pt", weights_only=True, map_location="cpu")
        shared = torch.load(folder / "shared_raw_later_state.pt", weights_only=True, map_location="cpu")
        historical = Path(cfg["original_config_path"]).parent
        old_filters = torch.load(historical / "filters_initial.pt", weights_only=True, map_location="cpu")
        assert torch.equal(initial["conv1.weight"], old_filters["conv1"])
        assert torch.equal(initial["conv2.weight"], shared["conv2.weight"])
        assert torch.equal(initial["conv2.bias"], shared["conv2.bias"])
        scale = cfg["initialisation_report"]["output_scale"]
        for name in ["classifier.weight", "classifier.bias"]:
            assert torch.equal(initial[name], shared[name] * scale)
        for name, value in shared.items():
            hashed = hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()
            assert hashed == fairness["shared_raw_later_sha256"][name]
        history = csv_rows(folder / "history.csv")
        assert [int(h["epoch"]) for h in history] == list(range(11))
        val = [float(h["val_accuracy"]) for h in history]
        assert abs(sum(val[1:]) / 10 - metrics["validation_aulc_1T"]) < 1e-10
        assert metrics["best_epoch"] == val.index(max(val))
        key = (cfg["dataset"], cfg["seed"])
        if key in per_group:
            first = per_group[key]
            assert all(torch.equal(shared[k], first["shared"][k]) for k in shared)
            assert fairness["training_orders"] == first["fairness"]["training_orders"]
            assert fairness["training_rng_sha256"] == first["fairness"]["training_rng_sha256"]
        else:
            per_group[key] = {"shared": shared, "fairness": fairness}
        if cfg["init"] == "kaiming":
            old_history = csv_rows(historical / "history.csv")
            max_val_diff = max(abs(float(a["val_accuracy"]) - float(b["val_accuracy"]))
                               for a, b in zip(history, old_history))
            original_best = torch.load(historical / "checkpoint_best.pt", weights_only=True, map_location="cpu")
            current_best = torch.load(folder / "checkpoint_best.pt", weights_only=True, map_location="cpu")
            same = all(torch.equal(current_best[k], original_best[k]) for k in original_best)
            baselines.append({"dataset": cfg["dataset"], "seed": cfg["seed"],
                              "max_validation_curve_difference": max_val_diff,
                              "historical_best_checkpoint_tensors_equal": same})
        evaluation = EVAL / "per_checkpoint" / folder.name / "COMPLETE.json"
        e = json.loads(evaluation.read_text())
        assert hashlib.sha256((folder / "checkpoint_best.pt").read_bytes()).hexdigest() == e["checkpoint_sha256"]
        observations = csv_rows(evaluation.parent / "corruption_rows.csv")
        assert len(observations) == 104
        assert abs(float(observations[0]["clean_accuracy"]) - metrics["test_accuracy"]) < 2e-5
        runs.append({"dataset": cfg["dataset"], "method": cfg["init"], "seed": cfg["seed"], "passed": True})
    summary = {"passed": True, "checked_runs": len(runs), "paired_initialization_groups": len(per_group),
               "kaiming_baseline_runs": len(baselines),
               "all_historical_kaiming_validation_curves_reproduced": all(b["max_validation_curve_difference"] == 0 for b in baselines),
               "all_historical_kaiming_best_checkpoint_tensors_reproduced": all(b["historical_best_checkpoint_tensors_equal"] for b in baselines),
               "checks": "actual saved initial tensors; common raw head and Conv2; original stems; every actual sample-order/RNG hash; AULC history; best validation epoch; evaluated-checkpoint hashes; clean accuracy"}
    (HERE / "completed_repair_verification.json").write_text(json.dumps({**summary, "baselines": baselines, "runs": runs}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
