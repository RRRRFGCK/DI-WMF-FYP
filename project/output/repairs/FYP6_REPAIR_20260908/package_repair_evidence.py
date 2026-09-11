"""Build/verify the additive FYP6 repair archive; never alter historical results.

Standard-library only. Run from the project root. Archive members retain their
project-relative paths. Restore mode copies only missing, SHA-verified reused
checkpoints from the old checkpoint archive; it never overwrites a model.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REL = HERE.relative_to(ROOT).as_posix()
MANIFEST = HERE / "evidence_package_manifest.json"
DEFAULT_ZIP = HERE / "FYP6_Repair_Evidence.zip"
OLD = ROOT / "output/pdf/FYP3_RECORDS_REVISION_20260906"
ORIGINAL_REGISTRY = "statistical_controls/all_core_hypotheses_holm.csv"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def safe_path(relative):
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or "\\" in relative or ":" in relative:
        raise ValueError(f"Unsafe archive path: {relative}")
    result = (ROOT / Path(*pure.parts)).resolve()
    if not result.is_relative_to(ROOT.resolve()):
        raise ValueError(f"Path escapes the project: {relative}")
    return result


def original_relative(path):
    # The archived JSON values remain untouched. Only the package index uses
    # portable relative paths; ROOT is the original root at archive creation.
    result = Path(path).resolve().relative_to(ROOT.resolve()).as_posix()
    safe_path(result)
    return result


def add_file(files, path, category):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    relative = path.resolve().relative_to(ROOT.resolve()).as_posix()
    safe_path(relative)
    files[relative] = category


def add_tree(files, path, category, excluded_parts=(), suffixes=None):
    for item in sorted(Path(path).rglob("*")):
        if not item.is_file():
            continue
        parts = item.relative_to(path).parts
        if any(p in {"__pycache__", *excluded_parts} or "smoke" in p.lower() for p in parts):
            continue
        if item.suffix.lower() in {".pyc", ".zip"} or item.name.endswith(".partial.csv"):
            continue
        if suffixes and item.suffix.lower() not in suffixes:
            continue
        add_file(files, item, category)


def collect():
    files = {}
    for name in ("independent_qa.md", "protocol_and_statistics_cpu_results.json",
                 "cpu_occlusion_equivalence_results.json"):
        if not (HERE / "first_layer/qa" / name).is_file():
            raise FileNotFoundError(f"Independent QA must finish before package freeze: {name}")
    # Include only named repair directories, never the thesis workspace.
    for folder in ("first_layer", "regression", "physiology_evaluation",
                   "physiology_figure_sources", "summary_figure_sources", "index_sources", "figure_label_cleanup"):
        add_tree(files, HERE / folder, "repair_" + folder)
    named = (
        "package_repair_evidence.py", "RUN_AND_VERIFY_REPAIR.md",
        "current_result_index.csv",
        "current_result_index_README.md", "index_summary.json",
        "evaluate_rr_patient46.py", "plot_rr_patient46_figures.py",
        "qa_rr_patient46.py", "qa_rr_figures_patient46.py",
        "rr_cpu_legacy.py", "rr_evaluation_protocol.json",
        "RR_EVALUATION_README.md", "RR_RESULTS_PATIENT46.md",
        "summarize_prepared_rr_classical.py", "redraw_summary_and_cost.py",
    )
    for name in named:
        add_file(files, HERE / name, "repair_driver_or_index")
    for name in ("build_current_result_index.py", "run_correncoder_patient_loso.py", "run_correncoder_regression.py",
                 "domain_mf/models.py", "domain_mf/regression.py"):
        add_file(files, ROOT / name, "regression_import_path_source")
    add_file(files, ROOT / ORIGINAL_REGISTRY, "unchanged_historical_registry")
    add_file(files, OLD / "audit_work/regression/regression_experiment_registry.csv",
             "historical_configuration_registry_at_driver_path")
    for name in ("submission_control_results/final_main_ten_seed_registry.csv",
                 "total_cost_results/cost_aggregate.csv"):
        add_file(files, ROOT / name, "current_summary_figure_input")
    # The label-cleanup generator explicitly verifies the previous package's
    # manifest at this path. Supply that exact manifest and its eight named
    # inputs, not an entire duplicated historical archive.
    figure_sources = json.loads((HERE / "figure_label_cleanup/figure_source_manifest.json").read_text())
    pm = Path(figure_sources["package_manifest"])
    assert sha256(pm) == figure_sources["package_manifest_sha256"]
    add_file(files, pm, "label_figure_historical_input_at_driver_path")
    for source in figure_sources["sources"]:
        path = Path(source["original_frozen_path"])
        assert sha256(path) == source["sha256"]
        add_file(files, path, "label_figure_historical_input_at_driver_path")
    add_tree(files, ROOT / "outputs_first_layer_shared_20260908", "first_layer_run_records")
    add_tree(files, ROOT / "first_layer_shared_results_20260908", "first_layer_final_analysis")

    patient_root = ROOT / "outputs_correncoder_patient46_20260908"
    reused, new_groups = [], []
    new_checkpoint_paths = set()
    for path in sorted(patient_root.glob("*/groups/*/group_manifest.json")):
        group = json.loads(path.read_text(encoding="utf-8"))
        if group["status"] != "complete":
            raise RuntimeError(f"Incomplete patient group: {path}")
        if group["reused"]:
            copies = [v for v in group["copied_files"]
                      if Path(v["destination"]).name == "checkpoint_final.pt"]
            assert len(copies) == 1
            copied = copies[0]
            source, destination = original_relative(copied["source"]), original_relative(copied["destination"])
            assert sha256(safe_path(source)) == sha256(safe_path(destination)) == copied["sha256"]
            reused.append({"experiment": group["experiment"], "patient_id": group["patient_id"],
                           "source": source, "destination": destination, "sha256": copied["sha256"]})
        else:
            checkpoint = original_relative(group["checkpoint"])
            assert sha256(safe_path(checkpoint)) == group["checkpoint_sha256"]
            new_checkpoint_paths.add(checkpoint)
            new_groups.append({"experiment": group["experiment"], "patient_id": group["patient_id"],
                               "checkpoint": checkpoint, "sha256": group["checkpoint_sha256"]})
    for path in sorted(patient_root.rglob("*")):
        if not path.is_file() or path.name.endswith(".partial.csv") or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(ROOT).as_posix()
        if path.suffix.lower() == ".pt" and relative not in new_checkpoint_paths:
            continue
        add_file(files, path, "patient_group_records_and_new_models")

    # Audit source/ is a thesis snapshot, and is deliberately not redistributed
    # in this evidence increment. The original observations and audit code are.
    add_tree(files, ROOT / "output/audits/FYP6_DEEP_AUDIT_20260908", "deep_audit",
             excluded_parts={"source"}, suffixes={".py", ".md", ".json", ".csv"})
    assert len(reused) == 559 and len(new_groups) == 39
    assert len(list((ROOT / "outputs_first_layer_shared_20260908").glob("*/checkpoint_best.pt"))) == 140
    assert len(list((ROOT / "outputs_first_layer_shared_20260908").glob("*/history.csv"))) == 140
    assert len(list((ROOT / "first_layer_shared_results_20260908/per_checkpoint").glob("*/COMPLETE.json"))) == 140
    assert len(list(patient_root.glob("*/groups/*/group_manifest.json"))) == 598
    assert len(list(patient_root.glob("*/groups/*/history.csv"))) == 598
    assert len(list(patient_root.glob("*/experiment_config.json"))) == 13
    assert len(list(patient_root.glob("*/records/*/waveforms.npz"))) == 689
    assert len(list(patient_root.glob("*/records/*/final_metrics.json"))) == 689
    assert len(read_csv(ROOT / "first_layer_shared_results_20260908/corruption_rows.csv")) == 14560
    assert len(read_csv(HERE / "current_result_index.csv")) == 902
    assert len(read_csv(ROOT / "first_layer_shared_results_20260908/statistical_registry_replacement_378.csv")) == 378
    assert len(read_csv(HERE / "physiology_evaluation/physiology_paired_holm_patient46.csv")) == 120
    return files, reused, new_groups


def build(archive):
    archive = archive.resolve()
    if archive.exists():
        raise FileExistsError(f"Refusing to replace an existing evidence package; choose --zip PATH: {archive}")
    files, reused, new_groups = collect()
    prerequisites = []
    old_members = set()
    for name in ("FYP_Code_Records_Audit.zip", "FYP_Checkpoint_Archive.zip"):
        path = OLD / name
        with zipfile.ZipFile(path) as z:
            old_members.update(z.namelist())
        prerequisites.append({"name": name, "sha256": sha256(path), "bytes": path.stat().st_size,
                              "archive_root_prefix": "FYP_Code_Records_Audit/", "required": True})
    for item in reused:
        if "FYP_Code_Records_Audit/" + item["source"] not in old_members:
            raise RuntimeError(f"Reused model absent from old archive: {item['source']}")
    for relative in files:
        assert "/thesis/" not in relative and "smoke" not in relative.lower()
        assert not relative.startswith(("data/", "data_bidmc/", "data_capnobase/"))
    inventory = [{"path": relative, "category": files[relative],
                  "bytes": safe_path(relative).stat().st_size, "sha256": sha256(safe_path(relative))}
                 for relative in sorted(files)]
    manifest = {
        "package": "FYP6_Repair_Evidence", "kind": "incremental supplement, not a standalone complete archive",
        "created_utc": datetime.now(timezone.utc).isoformat(), "original_project_root": str(ROOT),
        "path_convention": "ZIP entries are relative to the domain_mf_v2 project root, without an enclosing directory",
        "required_previous_archives": prerequisites,
        "raw_datasets_included": False,
        "derived_reference_signal_disclosure": "The 689 waveforms.npz files contain model predictions and processed reference waveforms; the RR preparation cache also contains processed references. No original MAT/image dataset is included. Dataset terms still apply to these derived evaluation records.",
        "excluded": ["thesis sources/PDFs/Overleaf ZIP", "original raw datasets", "smoke outputs", "Python caches", "559 already-archived reused patient checkpoints", "temporary partial histories"],
        "counts": {"first_layer_trained_models": 140, "first_layer_histories": 140,
                   "first_layer_evaluated_models": 140, "first_layer_corruption_rows": 14560,
                   "patient_configurations": 13, "patient_group_manifests": 598,
                   "patient_record_outputs": 689, "new_patient_checkpoints_included": 39,
                   "reused_patient_checkpoints_in_old_archive": 559,
                   "historical_hypothesis_index_rows": 902, "replacement_first_layer_tests": 378,
                   "replacement_patient_tests": 120},
        "reused_checkpoint_restore_map": reused, "new_patient_groups": new_groups,
        "inventory": inventory,
        "manifest_self_hash_note": "This manifest is the only archive member not listed in its own inventory. The external package summary hashes the complete ZIP.",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
        for item in inventory:
            z.write(safe_path(item["path"]), arcname=item["path"])
        z.write(MANIFEST, arcname=MANIFEST.relative_to(ROOT).as_posix())
    verification = verify_zip(archive)
    summary = {"zip": str(archive), "zip_bytes": archive.stat().st_size,
               "zip_sha256": sha256(archive), "uncompressed_inventory_bytes": sum(v["bytes"] for v in inventory),
               "file_count_including_manifest": len(inventory) + 1,
               "categories": dict(Counter(v["category"] for v in inventory)),
               "verification": verification, "counts": manifest["counts"]}
    (HERE / "evidence_package_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def verify_zip(archive):
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        assert len(names) == len(set(names)), "Duplicate ZIP members"
        for name in names:
            safe_path(name)
        mname = f"{REL}/evidence_package_manifest.json"
        manifest = json.loads(z.read(mname))
        assert set(names) == {v["path"] for v in manifest["inventory"]} | {mname}
        for item in manifest["inventory"]:
            h = hashlib.sha256()
            with z.open(item["path"]) as stream:
                for data in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(data)
            assert h.hexdigest() == item["sha256"], item["path"]
            assert z.getinfo(item["path"]).file_size == item["bytes"], item["path"]
        # Reading each member to EOF verifies CRC as well as SHA-256.
        return {"passed": True, "members_checked": len(names), "sha256_and_crc_checked": True,
                "safe_relative_paths": True, "training_or_inference_performed": False}


def verify_tree(require_reused):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    issues = []
    for item in manifest["inventory"]:
        path = safe_path(item["path"])
        if not path.exists() or sha256(path) != item["sha256"]:
            issues.append(item["path"])
    absent = []
    for item in manifest["reused_checkpoint_restore_map"]:
        path = safe_path(item["destination"])
        if not path.exists():
            absent.append(item["destination"])
        elif sha256(path) != item["sha256"]:
            issues.append(item["destination"])
    result = {"passed": not issues and (not require_reused or not absent),
              "included_files_checked": len(manifest["inventory"]), "changed_or_missing_included_files": issues,
              "reused_checkpoints_absent": len(absent), "require_reused": require_reused}
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


def restore_reused():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    copied = 0
    for item in manifest["reused_checkpoint_restore_map"]:
        source, destination = safe_path(item["source"]), safe_path(item["destination"])
        if destination.exists():
            if sha256(destination) != item["sha256"]:
                raise RuntimeError(f"Existing destination differs; nothing overwritten: {destination}")
            continue
        if not source.is_file() or sha256(source) != item["sha256"]:
            raise RuntimeError(f"Missing/wrong historical source. Extract the old checkpoint archive first: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as src, destination.open("xb") as dst:
            shutil.copyfileobj(src, dst)
        assert sha256(destination) == item["sha256"]
        copied += 1
    print(json.dumps({"copied_missing_reused_checkpoints": copied, "expected_destinations": 559,
                      "historical_sources_modified": False, "existing_models_overwritten": False}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build", action="store_true")
    mode.add_argument("--verify-zip", action="store_true")
    mode.add_argument("--verify-tree", action="store_true")
    mode.add_argument("--restore-reused-checkpoints", action="store_true")
    p.add_argument("--require-reused", action="store_true")
    p.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    a = p.parse_args()
    if a.build:
        build(a.zip)
    elif a.verify_zip:
        print(json.dumps(verify_zip(a.zip), indent=2))
    elif a.verify_tree:
        verify_tree(a.require_reused)
    else:
        restore_reused()


if __name__ == "__main__":
    main()
