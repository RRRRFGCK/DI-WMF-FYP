"""Read-only SHA-256 verification of the public code and result manifests."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    checked = 0
    failures = []
    for name in ("PUBLIC_CODE_MANIFEST.json", "PUBLIC_RESULTS_MANIFEST.json"):
        manifest = json.loads((root / name).read_text(encoding="utf-8"))
        for item in manifest["files"]:
            path = (root / item["path"]).resolve()
            if not path.is_relative_to(root):
                raise ValueError("Manifest path leaves repository: " + item["path"])
            if not path.is_file():
                failures.append("Missing: " + item["path"])
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != item["sha256"]:
                failures.append("Hash mismatch: " + item["path"])
            checked += 1
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Verified {checked} manifest-listed files; no training or inference run.")


if __name__ == "__main__":
    main()
