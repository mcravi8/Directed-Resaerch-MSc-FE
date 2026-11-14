"""Archive immutable fit evidence into Buddy-exportable, checked chunks."""
import hashlib
import json
import tarfile
from pathlib import Path

out=Path(__file__).resolve().parents[1]
state=json.loads((out/'post_empirical_state.json').read_text())
assert state['status']=='COMPUTATION_AND_AUDIT_COMPLETE'
target=out/'evidence_archive'
target.mkdir(exist_ok=True)
archive=target/'s02a_units.tar.gz'
if not archive.exists():
    staging=target/'s02a_units.tar.gz.inprogress'
    if staging.exists(): raise FileExistsError(f'uncertain archive attempt: {staging}')
    with tarfile.open(staging,'w:gz',compresslevel=1) as tar:
        tar.add(out/'units',arcname='units',recursive=True)
        tar.add(out/'timing_remeasurement_2OU/units',arcname='timing_remeasurement_2OU/units',recursive=True)
    staging.rename(archive)
chunk_bytes=30*1024*1024
pieces=[]
whole=hashlib.sha256()
with archive.open('rb') as source:
    index=0
    while True:
        data=source.read(chunk_bytes)
        if not data: break
        whole.update(data)
        part=target/f's02a_units.tar.gz.part{index:03d}'
        if part.exists():
            assert part.read_bytes()==data
        else:
            part.write_bytes(data)
        pieces.append(dict(name=part.name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
        index+=1
record=dict(archive='s02a_units.tar.gz',bytes=archive.stat().st_size,
            sha256=whole.hexdigest(),chunk_bytes=chunk_bytes,parts=pieces,
            restore='cat s02a_units.tar.gz.part* > s02a_units.tar.gz; verify SHA-256; tar -xzf s02a_units.tar.gz')
(out/'unit_archive_manifest.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(dict(archive_bytes=record['bytes'],pieces=len(pieces),sha256=record['sha256'])))
