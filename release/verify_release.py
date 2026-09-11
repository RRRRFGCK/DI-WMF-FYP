"""Read every release payload and verify its SHA-256, size and inventory."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import stat
import time
import zipfile
from pathlib import Path, PurePosixPath

PREFIX = "FYP_Research_Release/"
CHUNK = 1024 * 1024


def safe_path(name):
    parts = name.split("/")
    if not name or "\\" in name or ":" in name or name.startswith("/") or any(p in ("", ".", "..") for p in parts):
        raise ValueError(f"unsafe member path: {name!r}")
    if PurePosixPath(name).is_absolute():
        raise ValueError(name)
    return name


def digest(stream):
    value = hashlib.sha256()
    size = 0
    while data := stream.read(CHUNK):
        value.update(data)
        size += len(data)
    return value.hexdigest(), size


def parse_manifest(raw):
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
    names = set()
    for row in rows:
        safe_path(row["path"])
        folded = row["path"].casefold()
        if folded in names or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            raise ValueError(f"invalid or duplicate manifest row: {row['path']}")
        names.add(folded)
        int(row["bytes"])
    return rows


def verify(zip_path=None, root=None):
    started = time.time()
    zf = zipfile.ZipFile(zip_path) if zip_path else None
    try:
        if zf:
            entries = {}
            case_names = set()
            for entry in zf.infolist():
                if entry.is_dir():
                    safe_path(entry.filename.rstrip("/"))
                    continue
                safe_path(entry.filename)
                if not entry.filename.startswith(PREFIX) or stat.S_ISLNK(entry.external_attr >> 16):
                    raise ValueError(f"unexpected member: {entry.filename}")
                name = entry.filename[len(PREFIX):]
                if name.casefold() in case_names:
                    raise ValueError(f"duplicate ZIP path: {name}")
                entries[name] = entry
                case_names.add(name.casefold())
            manifest_raw = zf.read(entries["MANIFEST.csv"])
            actual = set(entries)
        else:
            root = Path(root).resolve()
            manifest_raw = (root / "MANIFEST.csv").read_bytes()
            actual = set()
            for path in root.rglob("*"):
                if path.is_symlink():
                    raise ValueError(f"symbolic link in release: {path}")
                if path.is_file():
                    actual.add(path.relative_to(root).as_posix())
        rows = parse_manifest(manifest_raw)
        expected = {r["path"] for r in rows} | {"MANIFEST.csv"}
        if actual != expected:
            raise ValueError(f"inventory mismatch: missing={sorted(expected-actual)[:10]}, extra={sorted(actual-expected)[:10]}")
        count = total = restored = aliases = 0
        last = started
        for row in rows:
            with (zf.open(entries[row["path"]]) if zf else (root / row["path"]).open("rb")) as stream:
                sha, size = digest(stream)
            if sha != row["sha256"] or size != int(row["bytes"]):
                raise ValueError(f"content mismatch: {row['path']}")
            count += 1
            total += size
            restored += row["role"] == "restored_checkpoint"
            aliases += row["role"] == "result_alias"
            if time.time() - last >= 5:
                print(f"Verified {count}/{len(rows)} payloads; {total / 1e9:.3f} GB", flush=True)
                last = time.time()
        info_raw = zf.read(entries["RELEASE_INFO.json"]) if zf else (root / "RELEASE_INFO.json").read_bytes()
        info = json.loads(info_raw)
        if restored != 559 or info["restored_patient_checkpoints"] != restored:
            raise ValueError("checkpoint destination count does not match validated mapping")
        group_models = [name for name in expected if name.startswith("project/outputs_correncoder_patient46_20260908/") and name.endswith("/checkpoint_final.pt")]
        if len(group_models) != 598:
            raise ValueError(f"expected 598 patient-group checkpoints, found {len(group_models)}")
        report = {"status": "passed", "mode": "zip" if zf else "tree", "payload_files_verified": count,
                  "total_archive_files": len(actual), "payload_bytes_read": total,
                  "manifest_bytes_read": len(manifest_raw), "all_files_read": True,
                  "zip_crc_checks_exercised": bool(zf), "restored_checkpoint_destinations": restored,
                  "patient_group_checkpoints": len(group_models), "result_aliases": aliases,
                  "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
                  "elapsed_seconds": round(time.time()-started, 3)}
        if zip_path:
            with Path(zip_path).open("rb") as stream:
                report["zip_sha256"], report["zip_bytes"] = digest(stream)
        return report
    finally:
        if zf:
            zf.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--zip", type=Path)
    group.add_argument("--root", type=Path)
    parser.add_argument("--report", type=Path, help="optional report outside the frozen release")
    args = parser.parse_args()
    report = verify(args.zip, args.root or Path(__file__).resolve().parent)
    text = json.dumps(report, indent=2)
    if args.report:
        with args.report.open("x", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print(text, flush=True)


if __name__ == "__main__":
    main()
