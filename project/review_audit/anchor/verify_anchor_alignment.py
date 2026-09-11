"""Audit frozen class indices and optionally replay only the semantic evaluation.

No fitting or gradient updates are performed. The optional replay uses the saved
classification checkpoint and the original test-subset construction, not training
data to redefine Kaiming reference classes. Run with --repo-root pointing at the
project or the supplied code-and-records package.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--sign-root", type=Path)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()
    root = args.repo_root.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root))
    groups = defaultdict(list)
    saved_models = {}
    source_rows = []
    evaluation = {}
    for folder in ("expanded_500", "sign_full_10seed"):
        directory = root / "template_semantics_results" / folder
        evaluation[folder] = json.loads((directory / "evaluation_config.json").read_text())
        for name in ("template_channel_rows.csv", "template_model_metrics.csv", "evaluation_config.json"):
            path = directory / name
            source_rows.append({"path": path.relative_to(root).as_posix(), "sha256": sha256(path)})
        for row in read_csv(directory / "template_channel_rows.csv"):
            key = (row["dataset"], row["method"], int(row["model_seed"]))
            groups[key].append(row)
        for row in read_csv(directory / "template_model_metrics.csv"):
            saved_models[(row["dataset"], row["method"], int(row["model_seed"]))] = row

    rows = []
    for key, channels in sorted(groups.items()):
        dataset, method, seed = key
        channels.sort(key=lambda row: int(row["channel"]))
        assert [int(row["channel"]) for row in channels] == list(range(80)), key
        mismatches = sum(int(row["anchor_class"]) != int(row["channel"]) // 8 for row in channels)
        alignment = statistics.mean(float(row["anchor_alignment"]) for row in channels)
        error = abs(alignment - float(saved_models[key]["mean_anchor_alignment"]))
        assert mismatches == 0 and error < 1e-12, (key, mismatches, error)
        rows.append({"dataset": dataset, "method": method, "seed": seed,
                     "channels": len(channels), "class_rule": "floor(channel_index / 8)",
                     "class_assignment_mismatches": mismatches,
                     "recomputed_saved_channel_mean": alignment,
                     "absolute_error_to_saved_model_mean": error,
                     "meaning": "arbitrary_fixed_index_reference" if method == "kaiming" else "template_source_class"})

    replay_rows = []
    if args.replay:
        import torch
        from domain_mf.data import build_test_dataset
        from domain_mf.models import build_model
        from evaluate_template_semantics import make_loader, collect_features, semantic_rows

        device = torch.device(args.device)
        torch.set_num_threads(4)
        dataset_cache = {}
        for row in rows:
            key = row["dataset"], row["method"], row["seed"]
            dataset, method, seed = key
            family = "sign_full_10seed" if dataset == "sign" else "expanded_500"
            ev = evaluation[family]
            candidates = []
            for old_root in ev["output_roots"]:
                local_root = root / Path(old_root.replace("\\", "/")).name
                for config_path in local_root.glob("*/config.json"):
                    config = json.loads(config_path.read_text())
                    if (config.get("dataset"), config.get("init"), config.get("seed")) == key:
                        if config.get("model") == "standard" and config.get("epochs") == 20 and config.get("train_fraction") == 1.0:
                            checkpoint = config_path.with_name("checkpoint_best.pt")
                            if checkpoint.exists():
                                candidates.append((config, checkpoint))
            assert len(candidates) == 1, (key, len(candidates))
            config, checkpoint = candidates[0]
            if dataset not in dataset_cache:
                dataset_cache[dataset] = build_test_dataset(
                    dataset, args.data_root or Path(config["data_root"]),
                    args.sign_root or Path(config["sign_root"]))
            loader = make_loader(dataset_cache[dataset], ev["max_semantic_samples"],
                                 SimpleNamespace(batch_size=ev["batch_size"], num_workers=0), device)
            model = build_model("standard", dataset).to(device)
            model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
            _, labels, features = collect_features(model, loader, device)
            fresh = semantic_rows(model, labels, features, ev["top_examples"])
            stored = groups[key]
            max_error = max(abs(float(a["anchor_alignment"]) - float(b["anchor_alignment"])) for a,b in zip(fresh, stored))
            mean_value = statistics.mean(a["anchor_alignment"] for a in fresh)
            replay_rows.append({"dataset": dataset, "method": method, "seed": seed,
                                "checkpoint": checkpoint.relative_to(root).as_posix(),
                                "checkpoint_sha256": sha256(checkpoint), "samples": len(labels),
                                "top_examples_per_channel": ev["top_examples"],
                                "replayed_mean_anchor_alignment": mean_value,
                                "saved_mean_anchor_alignment": row["recomputed_saved_channel_mean"],
                                "max_channel_absolute_error": max_error,
                                "mean_absolute_error": abs(mean_value-row["recomputed_saved_channel_mean"]),
                                "passes_tolerance_1e-6": max_error < 1e-6})
            print(f"{dataset}/{method}/{seed}: max channel error={max_error:.3g}", flush=True)
            del model, features
        write_csv(args.output / "checkpoint_replay.csv", replay_rows)

    write_csv(args.output / "fixed_class_index_audit.csv", rows)
    write_csv(args.output / "source_hashes.csv", source_rows)
    summary = {"models": len(rows), "channels": sum(row["channels"] for row in rows),
               "class_index_mismatches": sum(row["class_assignment_mismatches"] for row in rows),
               "max_saved_aggregation_error": max(row["absolute_error_to_saved_model_mean"] for row in rows),
               "replay_performed": args.replay, "replayed_models": len(replay_rows),
               "replay_all_pass": all(row["passes_tolerance_1e-6"] for row in replay_rows) if replay_rows else None,
               "definition": "Mean over 80 channels of the fraction of the five highest-activation test examples matching floor(channel/8).",
               "kaiming_assignment": "Fixed by architecture/index before fitting; no training, validation or test data select the reference class.",
               "claim_limit": "Asymmetric provenance control: Kaiming indices are arbitrary, DI-WMF indices are source classes. This does not show superior semantic purity or stability relative to Kaiming classes learned from training responses.",
               "source_provenance_limit": "Current source and saved outputs checked on 2026-09-06; not a recovered run-date source snapshot."}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if args.replay and not summary["replay_all_pass"]:
        raise SystemExit("Replay differences require inspection.")


if __name__ == "__main__":
    main()
