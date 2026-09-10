"""Freeze exact bytes named by the first repaired group's runtime code hashes."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
config=json.loads((ROOT/"outputs_correncoder_patient46_20260908/pretrained_mse/groups/s03386/run_config.json").read_text())
rows=[]
for relative,expected in config["source_codes_sha256"].items():
    source=ROOT/relative
    actual=hashlib.sha256(source.read_bytes()).hexdigest()
    assert actual==expected, f"Source changed before freezing: {relative}"
    target=OUT/"frozen_training_source"/relative
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        assert hashlib.sha256(target.read_bytes()).hexdigest()==expected
    else:
        shutil.copy2(source,target)
    rows.append({"source":str(source),"snapshot":str(target),"sha256":expected})
(OUT/"frozen_source_manifest.json").write_text(json.dumps({
    "protocol_version":config["protocol_version"],
    "meaning":"Exact source bytes matching the repaired run_config hashes, not a retrospective claim about historical experiments.",
    "files":rows},indent=2),encoding="utf-8")
print(json.dumps(rows,indent=2))
