"""Create visual contact sheets and check PDF text boundaries/source immutability."""
import hashlib
import json
from pathlib import Path

import pdfplumber
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PDF = ROOT / "output/pdf/FYP6_DEEP_AUDIT_20260908.pdf"
QA = ROOT / "tmp/pdfs/audit6"
pages = sorted(QA.glob("page-*.png"))
for start in (0, 5):
    subset = pages[start:start+5]
    canvas = Image.new("RGB", (5*354, 524), "#eeeeee")
    draw = ImageDraw.Draw(canvas)
    for offset, path in enumerate(subset):
        im = Image.open(path).convert("RGB")
        im.thumbnail((344, 486))
        canvas.paste(im, (offset*354+5, 24))
        draw.text((offset*354+10, 5), f"Page {start+offset+1}", fill="black")
    canvas.save(QA/f"contact-{start//5+1}.png")

checks = []
with pdfplumber.open(PDF) as pdf:
    for index, page in enumerate(pdf.pages):
        chars = page.chars
        body = [c for c in chars if c["top"] < page.height-30]
        checks.append({"page": index+1, "characters": len(chars),
                       "body_top": min(c["top"] for c in body),
                       "body_bottom": max(c["bottom"] for c in body),
                       "left": min(c["x0"] for c in chars),
                       "right": max(c["x1"] for c in chars),
                       "first_text": page.extract_text()[:80]})
    assert len(pdf.pages) == 10
    assert all(c["left"] >= 40 and c["right"] <= 556 and c["body_bottom"] < 804 for c in checks)
manifest = json.loads((HERE/"INPUT_MANIFEST.json").read_text(encoding="utf-8"))
zip_hash = hashlib.sha256(Path(manifest["thesis_zip"]).read_bytes()).hexdigest()
assert zip_hash == manifest["zip_sha256"], "Uploaded thesis ZIP changed"
print(json.dumps({"pages": checks, "zip_unchanged": True}, ensure_ascii=False, indent=2))
(HERE/"pdf_qa.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
