"""Create a submission audit from immutable run artefacts.

This script does not train a model.  It reads saved configurations, metrics and
initial/final filters so that claims in the dissertation can be traced to the
records that actually produced them.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "submission_audit"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iso_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")


def run_records() -> list[tuple[Path, dict, dict]]:
    records = []
    for config_path in sorted(ROOT.glob("outputs*/**/config.json")):
        metrics_path = config_path.with_name("final_metrics.json")
        if not metrics_path.exists():
            continue
        try:
            records.append((config_path.parent, read_json(config_path), read_json(metrics_path)))
        except (json.JSONDecodeError, OSError):
            continue
    return records


def audit_drift(records: list[tuple[Path, dict, dict]]) -> list[dict]:
    rows = []
    for run_dir, config, metrics in records:
        initial_path = run_dir / "filters_initial.pt"
        final_path = run_dir / "filters_final.pt"
        saved = metrics.get("kernel_drift")
        if not saved or not initial_path.exists() or not final_path.exists():
            continue
        initial = torch.load(initial_path, map_location="cpu", weights_only=True)
        final = torch.load(final_path, map_location="cpu", weights_only=True)
        for layer in sorted(set(initial) & set(final) & set(saved)):
            w0 = initial[layer].double().flatten(1)
            w1 = final[layer].double().flatten(1)
            cosine = 1.0 - F.cosine_similarity(w1, w0, dim=1)
            relative_l2 = (w1 - w0).norm(dim=1) / w0.norm(dim=1).clamp_min(1e-12)
            saved_mean = float(saved[layer]["mean"])
            rows.append(
                {
                    "run": str(run_dir.relative_to(ROOT)),
                    "dataset": config.get("dataset"),
                    "model": config.get("model"),
                    "init": config.get("init"),
                    "seed": config.get("seed"),
                    "layer": layer,
                    "saved_kernel_drift_mean": saved_mean,
                    "recomputed_cosine_distance_mean": float(cosine.mean()),
                    "recomputed_relative_l2_mean": float(relative_l2.mean()),
                    "abs_error_saved_vs_cosine": abs(saved_mean - float(cosine.mean())),
                    "abs_error_saved_vs_relative_l2": abs(saved_mean - float(relative_l2.mean())),
                    "metrics_timestamp": iso_mtime(run_dir / "final_metrics.json"),
                }
            )
    return rows


def audit_lowrank(records: list[tuple[Path, dict, dict]]) -> list[dict]:
    rows = []
    for run_dir, config, _ in records:
        if config.get("init") != "lowrank_wmf":
            continue
        report = config.get("initialisation_report", {})
        rows.append(
            {
                "run": str(run_dir.relative_to(ROOT)),
                "output_root": run_dir.relative_to(ROOT).parts[0],
                "dataset": config.get("dataset"),
                "model": config.get("model"),
                "seed": config.get("seed"),
                "epochs": config.get("epochs"),
                "requested_covariance_rank": config.get("covariance_rank"),
                "recorded_effective_covariance_rank": report.get("covariance_rank"),
                "initialised_modules": ";".join(report.get("initialised_modules", [])),
                "config_timestamp": iso_mtime(run_dir / "config.json"),
            }
        )
    return rows


def audit_anchor(records: list[tuple[Path, dict, dict]]) -> list[dict]:
    rows = []
    for run_dir, config, metrics in records:
        schedule = config.get("reg_schedule", "none")
        if schedule == "none" or float(config.get("reg_lambda", 0.0)) == 0.0:
            continue
        rows.append(
            {
                "run": str(run_dir.relative_to(ROOT)),
                "output_root": run_dir.relative_to(ROOT).parts[0],
                "dataset": config.get("dataset"),
                "model": config.get("model"),
                "init": config.get("init"),
                "seed": config.get("seed"),
                "epochs": config.get("epochs"),
                "reg_schedule": schedule,
                "reg_lambda": config.get("reg_lambda"),
                "test_accuracy": metrics.get("test_accuracy"),
                "metrics_timestamp": iso_mtime(run_dir / "final_metrics.json"),
            }
        )
    return rows


def seed_registry(records: list[tuple[Path, dict, dict]]) -> list[dict]:
    groups: dict[tuple, list[int]] = defaultdict(list)
    for run_dir, config, _ in records:
        key = (
            run_dir.relative_to(ROOT).parts[0],
            config.get("dataset"),
            config.get("model"),
            config.get("init"),
            config.get("epochs"),
            config.get("train_fraction"),
        )
        seed = config.get("seed")
        if isinstance(seed, int):
            groups[key].append(seed)
    rows = []
    for key, seeds in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        unique = sorted(set(seeds))
        rows.append(
            {
                "output_root": key[0],
                "dataset": key[1],
                "model": key[2],
                "init": key[3],
                "epochs": key[4],
                "train_fraction": key[5],
                "n_unique_seeds": len(unique),
                "seeds": ";".join(map(str, unique)),
                "n_completed_runs": len(seeds),
            }
        )
    return rows


def statistics_manifest() -> dict:
    manifest_path = ROOT / "statistical_controls" / "manifest.json"
    registry_path = ROOT / "statistical_controls" / "all_core_hypotheses_holm.csv"
    family_path = ROOT / "statistical_controls" / "family_summary.csv"
    script_path = ROOT / "analyze_multiple_comparisons.py"
    manifest = read_json(manifest_path)
    with registry_path.open(newline="", encoding="utf-8-sig") as handle:
        registry_rows = list(csv.DictReader(handle))
    role_values = sorted(
        {row.get("submission_stage_role", "") for row in registry_rows}
        - {""}
    )
    source_timestamps = []
    for row in registry_rows:
        source = ROOT / row["source"]
        if source.exists():
            source_timestamps.append(source.stat().st_mtime)
    return {
        "prospective_or_preregistered_registry_found": False,
        "interpretation": (
            "The available 902-row mapping is a complete reproducible retrospective "
            "Holm audit, but no timestamped registry predating the analysed result files "
            "or version-control history was found. It must not be called preregistered "
            "or confirmatory."
        ),
        "reported_total_hypotheses": manifest.get("total_hypotheses"),
        "registry_rows": len(registry_rows),
        "submission_stage_roles": role_values,
        "role_field_status": manifest.get("role_field_status"),
        "reported_family_counts": manifest.get("families"),
        "observed_family_counts": dict(sorted(Counter(row["family"] for row in registry_rows).items())),
        "files": {
            str(path.relative_to(ROOT)): {
                "timestamp": iso_mtime(path),
                "sha256": sha256(path),
            }
            for path in (manifest_path, registry_path, family_path, script_path)
        },
        "latest_source_result_timestamp": (
            datetime.fromtimestamp(max(source_timestamps)).isoformat(timespec="seconds")
            if source_timestamps
            else None
        ),
        "version_control_metadata_found": (ROOT / ".git").exists(),
    }


def main() -> None:
    OUT.mkdir(exist_ok=True)
    records = run_records()
    drift = audit_drift(records)
    lowrank = audit_lowrank(records)
    anchor = audit_anchor(records)
    seeds = seed_registry(records)
    stats = statistics_manifest()

    write_csv(OUT / "drift_metric_registry.csv", drift)
    write_csv(OUT / "lowrank_rank_registry.csv", lowrank)
    write_csv(OUT / "anchor_run_registry.csv", anchor)
    write_csv(OUT / "seed_registry.csv", seeds)
    (OUT / "statistical_registry_manifest.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )

    max_cosine_error = max((row["abs_error_saved_vs_cosine"] for row in drift), default=float("nan"))
    max_l2_error = max((row["abs_error_saved_vs_relative_l2"] for row in drift), default=float("nan"))
    requested_ranks = Counter(str(row["requested_covariance_rank"]) for row in lowrank)
    effective_ranks = Counter(str(row["recorded_effective_covariance_rank"]) for row in lowrank)
    anchor_roots = Counter(row["output_root"] for row in anchor)
    report = f"""# Submission record audit

Generated from {len(records)} completed image-run directories. No training was performed.

## Drift formulation

- Saved layer summaries checked: {len(drift)}
- Maximum absolute error between the saved value and recomputed mean cosine distance: {max_cosine_error:.3e}
- Maximum absolute error between the saved value and recomputed mean relative L2 distance: {max_l2_error:.3e}
- Conclusion: the saved `kernel_drift` results were generated as one minus cosine similarity, not relative L2 distance.

## Anchor regularisation

- Completed runs with an active non-`none` regularisation schedule: {len(anchor)}
- Their output roots: {dict(anchor_roots)}
- No source snapshot or Git history predating these runs was found. The run artefacts prove which runs used a non-zero schedule, but do not independently prove the historical loss formula. These runs are therefore treated as a legacy exploratory ablation and are not used for central claims.

## Low-rank rank and complexity

- Completed low-rank runs: {len(lowrank)}
- Requested ranks: {dict(requested_ranks)}
- Recorded effective ranks: {dict(effective_ranks)}
- The implementation forms a dense covariance and calls `torch.linalg.eigh`; Woodbury only reduces the later inverse application. The image implementation therefore has dense-covariance/full-eigendecomposition scaling, even when the retained rank is small.

## Statistical registry

- Registry rows: {stats['registry_rows']}
- Families: {len(stats['observed_family_counts'])}
- Prospective registry found: {stats['prospective_or_preregistered_registry_found']}
- Submission-stage roles: {stats['submission_stage_roles']}
- Role status: {stats['role_field_status']}
- Conclusion: the 902-row file is a complete reproducible retrospective multiplicity audit, not evidence of prospective preregistration.

## Seeds

`seed_registry.csv` records the completed seed set for each output root, task, model, initialiser, epoch budget and training fraction. Different experimental tiers legitimately use different seed counts; every thesis figure and table must state its own tier and paired unit.
"""
    (OUT / "AUDIT_REPORT.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
