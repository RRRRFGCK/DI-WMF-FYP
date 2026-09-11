"""Archive read-only review evidence; verify original code/input hashes first."""
import hashlib
import json
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / "output/pdf/FYP6_DEEP_AUDIT_EVIDENCE_20260908.zip"
manifest = json.loads((HERE/"INPUT_MANIFEST.json").read_text(encoding="utf-8"))
checked = 0
for row in manifest["files"]:
    path = (ROOT if row["kind"] == "code_review_snapshot" else HERE/"source/thesis_fyp6")/row["path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"], str(path)
    checked += 1
assert hashlib.sha256(Path(manifest["thesis_zip"]).read_bytes()).hexdigest() == manifest["zip_sha256"]

with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    z.write(HERE/"README.md", "README.md")
    z.write(ROOT/"output/pdf/FYP6_DEEP_AUDIT_20260908.pdf", "FYP6_DEEP_AUDIT_20260908.pdf")
    for path in sorted(HERE.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            z.write(path, "audit/"+path.relative_to(HERE).as_posix())
    for row in manifest["files"]:
        if row["kind"] == "code_review_snapshot":
            z.write(ROOT/row["path"], "code_review_snapshot/"+Path(row["path"]).as_posix())
with zipfile.ZipFile(OUT) as z:
    assert z.testzip() is None
    count = len(z.infolist())
print(json.dumps({"archive":str(OUT), "size_bytes":OUT.stat().st_size,
                  "archive_entries":count, "source_hashes_unchanged":checked,
                  "zip_sha256":hashlib.sha256(OUT.read_bytes()).hexdigest()}, indent=2))
