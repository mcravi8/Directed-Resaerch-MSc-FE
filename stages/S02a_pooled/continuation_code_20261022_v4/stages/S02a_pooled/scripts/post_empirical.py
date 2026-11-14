"""Finish already-authorized S02a computation after all empirical receipts exist."""
import json
import subprocess
import sys
import time
from pathlib import Path

out = Path(__file__).resolve().parents[1]
root = out.parents[1]
scripts = out/'scripts'
logs = out/'continuation_logs'
state_path = out/'post_empirical_state.json'


def state(**fields):
    old = json.loads(state_path.read_text()) if state_path.exists() else {}
    old.update(fields)
    state_path.write_text(json.dumps(old,indent=2)+'\n')


def run(name, script):
    receipt = logs/f'{name}.json'
    if receipt.exists():
        prior = json.loads(receipt.read_text())
        if prior['exit_code'] == 0:
            return
        raise RuntimeError(f'prior {name} failed; inspect {receipt}')
    log = logs/f'{name}.log'
    if log.exists():
        raise FileExistsError(f'uncertain prior {name} run: {log}')
    state(status='RUNNING',current=name)
    started = time.perf_counter()
    with log.open('w') as file:
        proc = subprocess.run([sys.executable,'-u',str(scripts/script)],
                              cwd=root,stdout=file,stderr=subprocess.STDOUT)
    record = dict(command=[sys.executable,'-u',str(scripts/script)],
                  exit_code=proc.returncode,wall_seconds=time.perf_counter()-started,
                  log=str(log.relative_to(out)))
    receipt.write_text(json.dumps(record,indent=2)+'\n')
    if proc.returncode:
        state(status='STOPPED_ON_FAILURE',current=name)
        raise RuntimeError(f'{name} exited {proc.returncode}; see {log}')
    state(status='RUNNING',last_completed=name,current=None)


state(status='WAITING_FOR_EMPIRICAL',current=None)
complete = out/'empirical_completed.json'
while not complete.exists():
    manager_path = out/'continuation_manager_state.json'
    if manager_path.exists():
        manager = json.loads(manager_path.read_text())
        if manager.get('status','').startswith('STOPPED'):
            retry_path = out/'retry_manager_state.json'
            retry = json.loads(retry_path.read_text()) if retry_path.exists() else {}
            if retry.get('status') not in ('RUNNING','COMPUTATION_COMPLETE'):
                state(status='STOPPED_ON_EMPIRICAL_FAILURE',manager=manager)
                raise RuntimeError('empirical manager stopped; inspect its log and receipts')
    time.sleep(60)
records=json.loads(complete.read_text())
assert len(records)==48 and all(r['fit_files']==3000 for r in records)
assert len(list((out/'continuation_receipts').glob('empirical_*.json')))==48
state(status='EMPIRICAL_COMPLETE',empirical_panels=len(records))
run('clean_full_timing_2OU','remeasure_2ou.py')
run('verify_clean_timing_2OU','verify_remeasurement_2ou.py')
clean_verification=json.loads((out/'timing_remeasurement_2OU/fit_verification.json').read_text())
assert not clean_verification['errors'] and len(clean_verification['units'])==4
assert all(u['complete'] and u['completed_files']==1200 for u in clean_verification['units'])
run('reconcile_three_timings','reconcile_timing.py')
run('verify_all_fits','verify_fits.py')
verified=json.loads((out/'fit_verification.json').read_text())
assert not verified['errors'] and all(unit['complete'] for unit in verified['units'])
assert len([u for u in verified['units'] if u['unit'].startswith('empirical_')])==48
run('summarize_all_fits','summarize_fits.py')
run('create_execution_manifests','create_execution_manifests.py')
state(status='COMPUTATION_AND_AUDIT_COMPLETE',current=None,
      fit_records=verified['fit_records'],starts_checked=verified['optimizer_starts_checked'])
