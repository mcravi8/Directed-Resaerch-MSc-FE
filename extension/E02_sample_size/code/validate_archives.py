"""Validate transfer parts and 200-unit contents before handoff."""
import csv, hashlib, io, json, zipfile
from collections import defaultdict
from pathlib import Path

out=Path(__file__).resolve().parents[1]
with (out/'archive_manifest.csv').open(newline='') as f: archives=list(csv.DictReader(f))
with (out/'archive_parts_manifest.csv').open(newline='') as f: parts=list(csv.DictReader(f))
by=defaultdict(list)
for row in parts:by[row['archive']].append(row)
assert len(archives)==40
checked=0
for row in archives:
    name=Path(row['archive']).name;rr=sorted(by[name],key=lambda r:int(r['part']))
    assert len(rr)==int(rr[0]['parts'])
    assert [int(r['part']) for r in rr]==list(range(len(rr)))
    payload=[]
    for r in rr:
        p=out/r['path'];raw=p.read_bytes()
        assert len(raw)==int(r['bytes']) and hashlib.sha256(raw).hexdigest()==r['sha256']
        payload.append(raw)
    blob=b''.join(payload)
    assert len(blob)==int(row['bytes']) and hashlib.sha256(blob).hexdigest()==row['sha256']
    assert blob==(out/row['archive']).read_bytes()
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        assert len(z.namelist())==200 and z.testzip() is None
        reps=[]
        for member in z.namelist():
            unit=json.loads(z.read(member));assert unit['status']=='complete'
            assert unit['truth']==row['truth'] and unit['N']==int(row['N']) and unit['design']==row['design']
            reps.append(unit['rep'])
        assert sorted(reps)==list(range(200))
    checked+=200
report=dict(archives=len(archives),parts=len(parts),units_checked=checked,
            bytes=sum(int(r['bytes']) for r in archives),all_part_hashes_match=True,
            all_full_archive_hashes_match=True,all_zip_crc_pass=True)
(out/'archive_validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
