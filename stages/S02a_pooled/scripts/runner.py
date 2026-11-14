"""Absolute deadline worker watchdog. Stops at first failed job, preserves logs."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[3]; OUT=ROOT/'stages/S02a_pooled'
STOP=dt.datetime(2026,10,22,10,35,42,tzinfo=dt.timezone.utc).timestamp()
jobs=[('stage_tests',[sys.executable,str(OUT/'scripts/test_stage.py')])]
repro=OUT/'reproduction_root'
for number in (1,3,5,6):
    jobs.append((f'diag{number}',[sys.executable,'-u',str(repro/f'claude_review_2026-10-20/diag{number}.py'),str(repro)]))
for process in ('OU','2OU','ARFIMA'):
    jobs.append((f'timing_{process}',[sys.executable,'-u',str(OUT/'scripts/pipeline.py'),'pilot',process]))
jobs.append(('empirical',[sys.executable,'-u',str(OUT/'scripts/pipeline.py'),'empirical']))
manifest=json.loads((OUT/'frozen_manifest.json').read_text())
assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in manifest.items())
logdir=OUT/'logs';logdir.mkdir(exist_ok=True)
receipts=[]
for index,(name,command) in enumerate(jobs):
    available=STOP-time.time()
    if available<=0: break
    began=time.time()
    with (logdir/f'{name}.log').open('x') as log:
        proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=ROOT)
        timedout=False
        try: code=proc.wait(timeout=available)
        except subprocess.TimeoutExpired:
            timedout=True;os.killpg(proc.pid,signal.SIGTERM)
            try: code=proc.wait(timeout=10)
            except subprocess.TimeoutExpired: os.killpg(proc.pid,signal.SIGKILL);code=proc.wait()
    receipt=dict(job=name,command=command,exit_code=code,timed_out=timedout,started=began,finished=time.time(),wall_seconds=time.time()-began)
    receipts.append(receipt)
    (OUT/'command_receipts.json').write_text(json.dumps(receipts,indent=2)+'\n')
    state=dict(status='RUNNING' if code==0 else 'PARTIAL',completed=[x['job'] for x in receipts if x['exit_code']==0],pending=[j[0] for j in jobs[index+1:]],failed_or_interrupted=None if code==0 else name,elapsed_seconds=time.time()-dt.datetime(2026,10,22,3,5,42,tzinfo=dt.timezone.utc).timestamp())
    (OUT/'checkpoint.json').write_text(json.dumps(state,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)
    if code!=0: break
else:
    print('All computation units finished; independent verification still required.',flush=True)
