"""Resume missing empirical panels after a verified stopped worker session.

Call only after checking that no existing panel is making file progress.
Every run skips immutable completed fit units and retains failed logs.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

out=Path(__file__).resolve().parents[1]
root=out.parents[1]
script=out/'scripts/run_empirical_panel.py'
logs=out/'retry_logs'
logs.mkdir(exist_ok=True)
state=out/'retry_manager_state.json'
records=[]
for h in (120,61,250):
    for maturity in ('VIX9D','VIX30D','VIX3M','VIX6M'):
        for cohort in ('full','common'):
            for mode in ('event','group'):
                key=f'empirical_{h}_{maturity}_{cohort}_{mode}'
                receipt=out/'continuation_receipts'/f'{key}.json'
                if not receipt.exists():
                    attempts=sorted(logs.glob(f'{key}_attempt*.json'))
                    attempt=len(attempts)+1
                    log=logs/f'{key}_attempt{attempt}.log'
                    if log.exists():
                        raise FileExistsError(f'uncertain previous attempt: {log}')
                    args=[sys.executable,'-u',str(script),str(h),maturity,cohort,mode]
                    state.write_text(json.dumps(dict(status='RUNNING',key=key,attempt=attempt),indent=2)+'\n')
                    started=time.perf_counter()
                    with log.open('w') as file:
                        proc=subprocess.run(args,cwd=root,stdout=file,stderr=subprocess.STDOUT)
                    record=dict(command=args,exit_code=proc.returncode,
                                wall_seconds=time.perf_counter()-started,
                                log=str(log.relative_to(out)))
                    (logs/f'{key}_attempt{attempt}.json').write_text(json.dumps(record,indent=2)+'\n')
                    if proc.returncode:
                        state.write_text(json.dumps(dict(status='STOPPED_ON_FAILURE',key=key,
                                                         attempt=attempt),indent=2)+'\n')
                        raise RuntimeError(f'{key} failed; see {log}')
                record=json.loads(receipt.read_text())
                assert record['fit_files']==3000
                records.append(record)
assert len(records)==48
completed=out/'empirical_completed.json'
if not completed.exists():
    completed.write_text(json.dumps(records,indent=2)+'\n')
state.write_text(json.dumps(dict(status='COMPUTATION_COMPLETE',empirical_panels=48),indent=2)+'\n')
print('48 empirical panels complete')
