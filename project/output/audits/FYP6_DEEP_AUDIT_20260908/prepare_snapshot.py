"""Freeze read-only inputs for the FYP6 code/theory audit, not a run-time history."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import zipfile

audit = Path(__file__).resolve().parent
repo = audit.parents[2]
archive = Path('C:/Users/wyq20/Downloads/FYP (6).zip')
dest = audit / 'source/thesis_fyp6'
records = []
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for item in z.infolist():
        rel = PurePosixPath(item.filename)
        assert not rel.is_absolute() and '..' not in rel.parts
        if item.is_dir():
            continue
        data = z.read(item)
        path = dest.joinpath(*rel.parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            assert path.read_bytes() == data
        else:
            path.write_bytes(data)
        records.append({'kind': 'thesis', 'path': item.filename,
                        'sha256': hashlib.sha256(data).hexdigest()})
for path in sorted(list(repo.glob('*.py')) + list((repo/'domain_mf').glob('*.py'))
                   + list((repo/'tests').glob('*.py'))):
    records.append({'kind':'code_review_snapshot', 'path':str(path.relative_to(repo)),
                    'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
(audit/'INPUT_MANIFEST.json').write_text(json.dumps({
    'thesis_zip': str(archive),
    'zip_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
    'repo':str(repo), 'note':'Review-time hashes only, not original experiment-time commits.',
    'files':records},indent=2),encoding='utf-8')
print(dest)
print('source files hashed:',len(records))
