"""Run unchanged E5 public panels in four disjoint batches with periodic checkpoints."""
import json, os, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
import e5_pipeline as e5
HERE=Path(__file__).resolve().parent
RANGES=((0,258),(258,506),(506,754),(754,999))
def now(): return datetime.now(timezone.utc).isoformat()
def atomic(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');os.replace(tmp,path)
def snapshot(started,workers,t0):
    by_panel={p.name:len(list(p.glob('[0-9][0-9][0-9][0-9].json'))) for p in (HERE/'units').glob('*') if p.is_dir()}
    atomic(HERE/'public_checkpoint.json',dict(updated_at_utc=now(),started_at_utc=started,
        elapsed_seconds=time.monotonic()-t0,expected_panels=12,expected_draw_units=11988,
        completed_draw_units=sum(by_panel.values()),by_panel=by_panel,
        worker_pids=[p.pid for p in workers],worker_exit_codes=[p.poll() for p in workers],
        point_units=len(list((HERE/'units').glob('*/point.json')))))
if __name__=='__main__':
    e5.check_freeze();started=now();t0=time.monotonic()
    atomic(HERE/'public_run_start.json',dict(started_at_utc=started,freeze_sha=e5.check_freeze()['freeze_sha'],ranges=RANGES))
    print('point',e5.run(0,0,True),flush=True)
    workers=[]
    for a,b in RANGES:
        log=(HERE/f'worker_{a}_{b}.log').open('w')
        workers.append(subprocess.Popen([sys.executable,str(HERE/'e5_pipeline.py'),'bootstrap','--start',str(a),'--end',str(b)],stdout=log,stderr=subprocess.STDOUT))
    while any(p.poll() is None for p in workers):
        snapshot(started,workers,t0)
        if time.monotonic()-t0>7*3600+45*60:
            for p in workers:
                if p.poll() is None:p.terminate()
            raise TimeoutError('7h45 watchdog')
        time.sleep(15)
    snapshot(started,workers,t0)
    codes=[p.returncode for p in workers]
    if any(c!=0 for c in codes):raise RuntimeError(f'worker exit codes {codes}')
    print('summary',e5.summarize(),flush=True)
    snapshot(started,workers,t0)
    atomic(HERE/'public_run_end.json',dict(ended_at_utc=now(),elapsed_seconds=time.monotonic()-t0,worker_exit_codes=codes,
         summary=json.loads((HERE/'summary_status.json').read_text())))
