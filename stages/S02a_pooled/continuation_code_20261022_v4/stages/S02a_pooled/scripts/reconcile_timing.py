"""Read-only timing reconciliation; never substitutes resumed time for full time."""
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

out = Path(__file__).resolve().parents[1]
read = lambda name: json.loads((out / name).read_text())
ou = read('pilots/OU_timing.json')
two = read('pilots/2OU_timing.json')
two_reconciliation = read('pilots/2OU_timing_reconciliation.json')
lm = read('pilots/ARFIMA_timing.json')
remeasure_path = out/'timing_remeasurement_2OU/pilots/2OU_timing.json'
remeasured = json.loads(remeasure_path.read_text()) if remeasure_path.exists() else None
assert (ou['detected_events'], two['detected_events'], lm['detected_events']) == (22, 160, 375)
assert all(r['completed'] for r in (ou, two, lm))
expected = {f'r{rep}_b{b}_s{s}.json' for rep in range(-1, 199)
            for b in (0, 1) for s in (1, 3, 5)} | {'panel.json'}
panels = {}
for process in ('OU', '2OU'):
    for h in (120, 61):
        for anchor in ('argmax', 'onset5'):
            key = f'timing_{process}_{h}_{anchor}'
            actual = {p.name for p in (out / 'units' / key).glob('*.json')}
            assert actual == expected, (key, len(actual), len(expected))
            panels[key] = len(actual)-1
assert not list((out/'units').glob('timing_ARFIMA_*/*.json'))
if remeasured:
    assert remeasured['completed'] and remeasured['seed'] == two['seed']
    assert remeasured['detected_events'] == two['detected_events']
    clean_verification_path=out/'timing_remeasurement_2OU/fit_verification.json'
    if not clean_verification_path.exists():
        subprocess.run([sys.executable,str(out/'scripts/verify_remeasurement_2ou.py')],check=True)
    clean_verification=json.loads(clean_verification_path.read_text())
    assert not clean_verification['errors'] and len(clean_verification['units'])==4
    assert all(u['complete'] and u['completed_files']==1200 for u in clean_verification['units'])
    for h in (120, 61):
        for anchor in ('argmax', 'onset5'):
            key = f'timing_2OU_{h}_{anchor}'
            actual = {p.name for p in (out/'timing_remeasurement_2OU/units'/key).glob('*.json')}
            assert actual == expected, (key, len(actual), len(expected))
rows = [
    dict(process='OU', seed=ou['seed'], detected_events=22, eligible_fit_files=4800,
         full_grid_completed=True, full_wall_seconds=ou['total_wall_seconds'],
         continuation_wall_seconds='', wall_lower_bound_seconds=ou['total_wall_seconds'],
         timing_scope='original_full_measured'),
    dict(process='2OU', seed=two['seed'], detected_events=160, eligible_fit_files=4800,
         full_grid_completed=True, full_wall_seconds=remeasured['total_wall_seconds'] if remeasured else '',
         continuation_wall_seconds=two['total_wall_seconds'],
         wall_lower_bound_seconds=two_reconciliation['elapsed_from_first_existing_fit_seconds'],
         timing_scope='clean_full_measured; resumed grid retained separately' if remeasured else 'resumed_full_grid; full process wall unavailable'),
    dict(process='ARFIMA', seed=lm['seed'], detected_events=375, eligible_fit_files=0,
         full_grid_completed=True, full_wall_seconds=lm['total_wall_seconds'],
         continuation_wall_seconds='', wall_lower_bound_seconds=lm['total_wall_seconds'],
         timing_scope='full_measured; no panel reached ten common eligible events'),
]
with (out/'timing_benchmark.csv').open('w', newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
allocation = read('benchmark_allocation.json')
lower = 6*ou['total_wall_seconds']
assert lower > 21600 and allocation['N'] == 0
allocation.update(status='FROZEN_N0_BY_OU_LOWER_BOUND',
                  measured_full_process_seconds={'OU':ou['total_wall_seconds'],
                                                 'ARFIMA':lm['total_wall_seconds'],
                                                 **({'2OU':remeasured['total_wall_seconds']} if remeasured else {})},
                  two_ou_resume_seconds=two['total_wall_seconds'],
                  two_ou_full_wall_seconds=remeasured['total_wall_seconds'] if remeasured else None,
                  all_three_fit_grids_completed=True,
                  full_process_timings_measured=3 if remeasured else 2,
                  verified_pilot_count=3 if remeasured else 2,
                  timing_benchmark='timing_benchmark.csv')
if remeasured:
    denominator=lower+8*remeasured['total_wall_seconds']+4*lm['total_wall_seconds']
    assert min(100,math.floor(21600/denominator)) == 0
    allocation.update(status='FROZEN_N0_THREE_FULL_TIMINGS',denominator_seconds=denominator,
                      reason='Three full process timings measured; frozen allocation formula yields N=0.')
(out/'benchmark_allocation.json').write_text(json.dumps(allocation,indent=2)+'\n')
verification=dict(status='three_full_timings_verified' if remeasured else 'two_full_timings_one_resumed_grid',
                  pilot_counts={'OU':22,'2OU':160,'ARFIMA':375},
                  eligible_panel_fit_files=panels,
                  recovered_ou_audit=read('recovered_ou_audit.json'),
                  two_ou_timing_reconciliation=two_reconciliation,
                  arithmetic=dict(OU_weighted_doubled_seconds=lower,
                                  full_denominator_seconds=allocation.get('denominator_seconds'),
                                  planning_seconds=21600,N=0),
                  source_sha256={name:hashlib.sha256((out/name).read_bytes()).hexdigest()
                                 for name in ('pilots/OU_timing.json','pilots/2OU_timing.json',
                                              'pilots/ARFIMA_timing.json')})
(out/'timing_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
print(json.dumps(dict(pilot_counts=verification['pilot_counts'],
                      eligible_panels=len(panels),N=0,
                      full_process_timings_measured=3 if remeasured else 2)))
