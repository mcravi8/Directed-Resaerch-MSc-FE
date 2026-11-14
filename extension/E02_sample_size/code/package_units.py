"""Package and hash all immutable replication JSON files for handoff."""
import csv, hashlib, json, zipfile
from pathlib import Path

out=Path(__file__).resolve().parents[1]
unitroot=out/'units'
archives=out/'units_archives';archives.mkdir(exist_ok=True)
rows=[]
for truth in ('E1','E2','PL','SE'):
    for n in (19,40,80,160,320):
        for design in ('independent','calendar_joint'):
            paths=sorted((unitroot/truth/str(n)/design).glob('*.json'))
            assert len(paths)==200,(truth,n,design,len(paths))
            archive=archives/f'{truth}_N{n}_{design}.zip'
            with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
                for p in paths:
                    payload=p.read_bytes(); unit=json.loads(payload)
                    assert unit['status']=='complete' and unit['truth']==truth and unit['N']==n and unit['design']==design
                    z.writestr(str(p.relative_to(unitroot)),payload)
            digest=hashlib.sha256(archive.read_bytes()).hexdigest()
            rows.append(dict(truth=truth,N=n,design=design,replications=200,archive=str(archive.relative_to(out)),bytes=archive.stat().st_size,sha256=digest))
with (out/'archive_manifest.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
print('archives',len(rows),'units',sum(int(r['replications']) for r in rows),'bytes',sum(int(r['bytes']) for r in rows))
