"""Freeze complete drivers/records into review and companion tensor archives.

Original experiment files are read only. All generated files are written next
to this builder. Large tensors are streamed into ZIP without duplicating them
on disk; selected replay checkpoints are included in the smaller review ZIP.
"""
from __future__ import annotations
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PREFIX = "FYP_Code_Records_Audit"


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def hash_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    assert (ROOT / "domain_mf").exists(), ROOT
    # Prefix local Windows paths so deeply nested recorded run names survive
    # the legacy MAX_PATH limit without renaming any evidence.
    review = HERE / PREFIX
    if os.name == "nt":
        review = Path("\\\\?\\" + str(review))
    review.mkdir(exist_ok=True)
    files = {}
    def include(source, relative=None):
        if source.is_file():
            name = relative or source.relative_to(ROOT).as_posix()
            assert not name.startswith("/") and ".." not in Path(name).parts
            files[name] = source
    for path in ROOT.glob("*.py"):
        include(path)
    for path in ROOT.glob("*.csv"):
        include(path)
    for directory in ("domain_mf", "tests"):
        for path in (ROOT / directory).rglob("*.py"):
            include(path)
    include(ROOT / "README.md", "HISTORICAL_PROJECT_README.md")
    for directory in ROOT.iterdir():
        if not directory.is_dir():
            continue
        selected = (directory.name.startswith("outputs") or directory.name.endswith("_results")
                    or directory.name in {"statistical_controls", "statistical_corrections", "submission_audit"}
                    or directory.name.startswith("total_cost_"))
        if selected:
            for path in directory.rglob("*"):
                if path.suffix.lower() in {".csv", ".json", ".npz", ".pt", ".log", ".md", ".png", ".pdf"}:
                    include(path)
    for path in (ROOT / "thesis_artifacts").rglob("*"):
        if path.is_file() and path.suffix in {".csv", ".json"}:
            include(path)
    for path in (HERE / "audit_work").rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix not in {".pyc"}:
            include(path, "review_audit/" + path.relative_to(HERE / "audit_work").as_posix())
    for path in (HERE / "thesis_overleaf" / "figures").rglob("*"):
        if path.is_file():
            include(path, "publication_figures/" + path.relative_to(HERE / "thesis_overleaf" / "figures").as_posix())
    for path in (HERE / "bundle_docs").iterdir():
        include(path, path.name)
    include(HERE / "RESPONSE_TO_REVIEWER.md", "RESPONSE_TO_REVIEWER.md")
    include(Path("C:/Users/wyq20/Downloads/FYP (3).zip"), "provenance/original_teacher_FYP3.zip")
    include(Path(__file__), "provenance/build_evidence_bundle.py")

    # Enforce the original evidence manifests rather than relying solely on
    # broad filename patterns. Do not copy the raw licensed data files.
    reg_rows = read_csv(HERE / "audit_work/regression/regression_original_record_manifest.csv")
    anchor_rows = read_csv(HERE / "audit_work/anchor/checkpoint_replay.csv")
    primary_weights = {row["path"] for row in reg_rows if row["path"].endswith(".pt")}
    primary_weights |= {row["checkpoint"] for row in anchor_rows}
    expected = {row["path"] for row in reg_rows}
    expected |= set(json.loads((HERE / "audit_work/aulc/source_file_manifest.json").read_text()))
    for name in sorted(expected):
        assert (ROOT / name).is_file(), f"Required original evidence missing: {name}"
        include(ROOT / name, name)

    versions = {}
    for name in ("torch", "torchvision", "numpy", "scipy", "matplotlib", "pillow", "psutil", "nvidia-ml-py"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    environment = {"recorded_at": datetime.now(timezone.utc).isoformat(),
                   "status": "review_time_environment_not_original_per_run_lockfile",
                   "python": sys.version, "executable": sys.executable,
                   "platform": platform.platform(), "packages": versions}
    environment_path = HERE / "ENVIRONMENT_REVIEW.json"
    environment_path.write_text(json.dumps(environment, indent=2), encoding="utf-8")
    include(environment_path, environment_path.name)

    manifest = []
    small = []
    large = []
    for name, source in sorted(files.items()):
        companion = source.suffix == ".pt" and name not in primary_weights
        archive = "FYP_Checkpoint_Archive.zip" if companion else "FYP_Code_Records_Audit.zip"
        manifest.append({"path": name, "archive": archive, "bytes": source.stat().st_size,
                         "sha256": hash_file(source),
                         "source_mtime_utc_not_preregistration": datetime.fromtimestamp(source.stat().st_mtime, timezone.utc).isoformat()})
        if companion:
            large.append((name, source))
        else:
            target = review / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            small.append((name, target))
    with (review / "PACKAGE_MANIFEST.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    small.append(("PACKAGE_MANIFEST.csv", review / "PACKAGE_MANIFEST.csv"))
    print(f"Review payload: {len(small)} files; companion tensors: {len(large)}", flush=True)
    archives = []
    for archive_name, payload, level in (("FYP_Code_Records_Audit.zip", small, 5),
                                          ("FYP_Checkpoint_Archive.zip", large, 1)):
        archive_path = HERE / archive_name
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=level, allowZip64=True) as archive:
            for index, (name, source) in enumerate(payload):
                archive.write(source, f"{PREFIX}/{name}")
                if index and index % 1000 == 0:
                    print(f"{archive_name}: {index}/{len(payload)} files", flush=True)
        with zipfile.ZipFile(archive_path) as archive:
            assert len(archive.namelist()) == len(payload)
            corrupt = archive.testzip()
            assert corrupt is None, corrupt
        archives.append({"archive": archive_name, "files": len(payload), "bytes": archive_path.stat().st_size,
                         "sha256": hash_file(archive_path), "crc_validation": "passed"})
        print(f"Finished {archive_name}: {archive_path.stat().st_size/1e6:.1f} MB", flush=True)
    summary = {"payload_files": len(manifest), "required_audit_inputs": len(expected),
               "required_audit_inputs_present": True, "selected_replay_checkpoints": len(primary_weights),
               "archives": archives, "raw_datasets_included": False,
               "source_version_status": "current_source_snapshot_not_run_time_commit"}
    (HERE / "EVIDENCE_PACKAGE_VERIFICATION.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
