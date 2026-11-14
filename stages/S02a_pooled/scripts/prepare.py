import datetime as dt
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import numpy,scipy,pandas
ROOT=Path(__file__).resolve().parents[3]; out=ROOT/'stages/S02a_pooled'
repro=out/'reproduction_root'
for name in ['claude_review_2026-10-20','omori_pipeline/data','v3_run_2026-10-20/v3_out','v3_run_2026-10-20/v3_out_oos']:
    shutil.copytree(ROOT/name,repro/name,dirs_exist_ok=False)
paths=[out/'spec_v1.md',*sorted((out/'scripts').glob('*.py'))]
paths+=list((ROOT/'stages/S01_protocol').glob('*.md'))+[ROOT/'stages/S01_protocol/protocol_bundle.sha256',ROOT/'stages/S01_protocol/input_manifest.csv',ROOT/'stages/S01_protocol/attempt_02/protocol_core.py']
paths += [ROOT/'stages/S02a_pooled.md',ROOT/'decision_log.md']
paths += [x for x in repro.rglob('*') if x.is_file()]
manifest={str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in paths}
(out/'frozen_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(out/'environment.json').write_text(json.dumps(dict(python=sys.version,executable=sys.executable,platform=platform.platform(),machine=platform.machine(),numpy=numpy.__version__,scipy=scipy.__version__,pandas=pandas.__version__,concurrency=1,freeze_time=dt.datetime.now(dt.timezone.utc).isoformat()),indent=2)+'\n')
print('Frozen',len(manifest),'files; reproduction inputs copied.')
