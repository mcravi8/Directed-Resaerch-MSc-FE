"""Freeze supplemental verification/packaging code, not a scientific revision."""
import datetime as dt,hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[3];out=root/'stages/S02a_pooled'
initial=json.loads((out/'frozen_manifest.json').read_text())
assert all(hashlib.sha256((root/p).read_bytes()).hexdigest()==h for p,h in initial.items())
added={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((out/'scripts').glob('*.py')) if str(p.relative_to(root)) not in initial}
result=dict(frozen_at=dt.datetime.now(dt.timezone.utc).isoformat(),purpose='Independent verification and checkpoint packaging only; unchanged scientific generation code.',disclosure='Early source/cohort verification was exploratory verification before this supplemental freeze. Final verification must use these exact hashed bytes.',files=added)
(out/'verification_code_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
print('Original 52 files unchanged; supplemental scripts frozen:',len(added))
