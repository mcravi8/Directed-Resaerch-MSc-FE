"""Rebuild full cell ZIPs from transferred parts and verify SHA-256."""
import csv, hashlib
from collections import defaultdict
from pathlib import Path

out=Path(__file__).resolve().parents[1]
with (out/'archive_parts_manifest.csv').open(newline='') as f:parts=list(csv.DictReader(f))
by=defaultdict(list)
for p in parts:by[p['archive']].append(p)
dest=out/'units_archives';dest.mkdir(exist_ok=True)
for name,rows in sorted(by.items()):
    rows.sort(key=lambda r:int(r['part']))
    assert len(rows)==int(rows[0]['parts'])
    assert [int(r['part']) for r in rows]==list(range(len(rows)))
    target=dest/name
    if target.exists():
        assert hashlib.sha256(target.read_bytes()).hexdigest()==rows[0]['archive_sha256']
        continue
    digest=hashlib.sha256()
    with target.open('wb') as f:
        for row in rows:
            raw=(out/row['path']).read_bytes()
            assert len(raw)==int(row['bytes']) and hashlib.sha256(raw).hexdigest()==row['sha256']
            f.write(raw);digest.update(raw)
    assert digest.hexdigest()==rows[0]['archive_sha256']
print('reassembled',len(by),'archives from',len(parts),'parts')
