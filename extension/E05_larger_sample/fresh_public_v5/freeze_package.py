"""Create a write-once SHA-256 record before any fresh public outcome execution."""
import hashlib,json,os,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy,pandas,scipy
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
FILES=[HERE/'frozen_spec_v1.md',HERE/'spec_v2.md',HERE/'e5_pipeline.py',HERE/'run_pool.py',HERE/'run_public.py',HERE/'freeze_package.py',
       ROOT/'stages/S01_protocol/attempt_02/protocol_core.py',ROOT/'omori_pipeline/data/shock_events.csv',
       *[ROOT/f'extension/E03_forecast/raw/{m}.csv' for m in ('VIX9D','VIX30D','VIX3M','VIX6M')],
       *[HERE/f'private_sources/{m}_History.csv' for m in ('VXN','RVX','OVX','GVZ')]]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
if __name__=='__main__':
    path=HERE/'public_package_freeze.json'
    if path.exists():raise RuntimeError('freeze already exists')
    d=dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),scientific_definition='unchanged E5 public v1 and equal-index correction v2',
        review_informed=True,original_freeze_history_repaired=False,licensed_extract_active=False,
        seed=5902026,python=sys.version,numpy=numpy.__version__,pandas=pandas.__version__,scipy=scipy.__version__,
        hashes={str(p.relative_to(ROOT)):sha(p) for p in FILES})
    d['freeze_sha']=hashlib.sha256(json.dumps(d,sort_keys=True).encode()).hexdigest()
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');os.replace(tmp,path)
    print(json.dumps(dict(frozen_at_utc=d['frozen_at_utc'],freeze_sha=d['freeze_sha'],files=len(FILES))))
