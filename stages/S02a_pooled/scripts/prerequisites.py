import csv, hashlib, json, pathlib, platform, sys
root=pathlib.Path(__file__).resolve().parents[3]
out=root/'stages/S02a_pooled'
checks=[]
for row in csv.DictReader((root/'stages/S01_protocol/input_manifest.csv').open()):
    p=root/row['path']
    actual=hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
    checks.append(dict(path=row['path'],expected=row['sha256'],actual=actual,ok=actual==row['sha256']))
(out/'prerequisite_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(dict(checks=len(checks),failed=[x for x in checks if not x['ok']],python=sys.version,platform=platform.platform()),indent=2))
sys.exit(int(not all(x['ok'] for x in checks)))
