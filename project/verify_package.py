"""Check SHA-256 of delivered records; no dependencies and no file writes."""
import argparse
import csv
import hashlib
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if os.name == "nt" and not str(root).startswith("\\\\?\\"):
        root = Path("\\\\?\\" + str(root))
    checked = 0
    missing_companion = []
    errors = []
    with (root / "PACKAGE_MANIFEST.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            path = root / row["path"]
            if not path.exists():
                if row["archive"] == "FYP_Checkpoint_Archive.zip" and not args.require_all:
                    missing_companion.append(row["path"])
                else:
                    errors.append("missing: " + row["path"])
                continue
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024*1024), b""):
                    digest.update(chunk)
            if path.stat().st_size != int(row["bytes"]) or digest.hexdigest() != row["sha256"]:
                errors.append("hash/size mismatch: " + row["path"])
            checked += 1
    print(f"Verified payload files: {checked}")
    print(f"Companion weight files not extracted: {len(missing_companion)}")
    print(f"Errors: {len(errors)}")
    for error in errors:
        print(error)
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
