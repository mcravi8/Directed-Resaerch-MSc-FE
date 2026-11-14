"""Split archive bytes for Buddy's per-file transfer limit; hash every part."""
import csv, hashlib
from pathlib import Path

out=Path(__file__).resolve().parents[1]
archives=out/'units_archives'
parts=out/'units_archive_parts';parts.mkdir(exist_ok=True)
rows=[];size=20*1024*1024
for archive in sorted(archives.glob('*.zip')):
    raw=archive.read_bytes()
    for i,offset in enumerate(range(0,len(raw),size)):
        payload=raw[offset:offset+size]
        path=parts/f'{archive.name}.part{i:03d}'
        path.write_bytes(payload)
        rows.append(dict(archive=archive.name,archive_sha256=hashlib.sha256(raw).hexdigest(),
                         part=i,parts=(len(raw)+size-1)//size,path=str(path.relative_to(out)),
                         bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest()))
with (out/'archive_parts_manifest.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
print('archives',len(list(archives.glob('*.zip'))),'parts',len(rows),'bytes',sum(int(r['bytes']) for r in rows))
