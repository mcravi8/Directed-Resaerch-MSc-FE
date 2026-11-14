"""Compute the latter 24 frozen empirical panels alongside the forward manager.

Per-panel advisory locks in run_empirical_panel.py prevent duplicated fit work.
All units remain immutable and share the frozen seed and fit code.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

out=Path(__file__).resolve().parents[1]
root=out.parents[1]
logs=out/'reverse_logs';logs.mkdir(exist_ok=True)
panels=[(h,m,c,mode) for h in (120,61,250)
        for m in ('VIX9D','VIX30D','VIX3M','VIX6M')
        for c in ('full','common') for mode in ('event','group')]
assert len(panels)==48
for h,m,c,mode in reversed(panels[24:]):
    key=f'empirical_{h}_{m}_{c}_{mode}'
    receipt=out/'continuation_receipts'/f'{key}.json'
    if receipt.exists(): continue
    attempt=1
    while (logs/f'{key}_attempt{attempt}.json').exists(): attempt+=1
    log=logs/f'{key}_attempt{attempt}.log'
    if log.exists(): raise FileExistsError(f'uncertain run: {log}')
    command=[sys.executable,'-u',str(out/'scripts/run_empirical_panel.py'),str(h),m,c,mode]
    environment=os.environ.copy();environment['S02A_WORKERS']='5'
    began=time.perf_counter()
    with log.open('w') as file:
        proc=subprocess.run(command,cwd=root,env=environment,stdout=file,stderr=subprocess.STDOUT)
    record=dict(key=key,command=command,exit_code=proc.returncode,
                wall_seconds=time.perf_counter()-began,log=str(log.relative_to(out)))
    (logs/f'{key}_attempt{attempt}.json').write_text(json.dumps(record,indent=2)+'\n')
    if proc.returncode: raise RuntimeError(f'{key} failed; see {log}')
    assert receipt.exists(),f'{key} returned without receipt'
print('latter 24 empirical panels complete',flush=True)
