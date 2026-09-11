"""Reconstruct the research ZIP, checking each part and the complete archive."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--check-only", action="store_true",
                        help="check all part bytes without writing the reconstructed ZIP")
    args = parser.parse_args()
    root = args.directory.resolve()
    manifest = json.loads((root / "release_parts.json").read_text(encoding="utf-8"))
    output_name = manifest["archive_name"]
    if Path(output_name).name != output_name or "/" in output_name or "\\" in output_name:
        raise ValueError("Unsafe output filename")
    output = root / output_name
    partial = root / (output_name + ".partial")
    if not args.check_only and (output.exists() or partial.exists()):
        raise FileExistsError("Output or partial file already exists; nothing overwritten")
    target = None
    complete = hashlib.sha256()
    total = 0
    try:
        if not args.check_only:
            target = partial.open("xb")
        for part in manifest["parts"]:
            name = part["name"]
            if Path(name).name != name or "/" in name or "\\" in name:
                raise ValueError("Unsafe part filename")
            path = root / name
            if path.is_symlink():
                raise ValueError("Part must be a regular file")
            if path.stat().st_size != part["bytes"]:
                raise ValueError("Wrong part size: " + name)
            digest = hashlib.sha256()
            with path.open("rb") as source:
                while chunk := source.read(4 * 1024 * 1024):
                    digest.update(chunk)
                    complete.update(chunk)
                    total += len(chunk)
                    if target:
                        target.write(chunk)
            if digest.hexdigest() != part["sha256"]:
                raise ValueError("Checksum mismatch: " + name)
            print("Verified " + name, flush=True)
        if total != manifest["archive_bytes"] or complete.hexdigest() != manifest["archive_sha256"]:
            raise ValueError("Complete archive checksum mismatch")
    finally:
        if target:
            target.close()
    if args.check_only:
        print("All parts and the complete archive checksum verified.")
    else:
        # Path.rename does not overwrite an existing destination on Windows.
        if output.exists():
            raise FileExistsError("Output appeared during reconstruction")
        partial.rename(output)
        print("Created " + str(output))


if __name__ == "__main__":
    main()
