"""Describe measured pilot scope and immutable checkpoint provenance."""
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import numpy
import scipy

out=Path(__file__).resolve().parents[1]
root=out.parents[1]
load=lambda path: json.loads((out/path).read_text())
digest=lambda path: hashlib.sha256((out/path).read_bytes()).hexdigest()
frozen=load('frozen_manifest.json')
for path in ('stages/S02a_pooled/spec_v1.md',
             'stages/S01_protocol/attempt_02/protocol_core.py'):
    assert hashlib.sha256((root/path).read_bytes()).hexdigest()==frozen[path],path
assert digest('scripts/pipeline_frozen.py')==frozen['stages/S02a_pooled/scripts/pipeline.py']
timing=load('timing_verification.json')
assert timing['status']=='three_full_timings_verified'
records={p:load(f'pilots/{p}_timing.json') for p in ('OU','2OU','ARFIMA')}
records['2OU_clean']=load('timing_remeasurement_2OU/pilots/2OU_timing.json')
env=dict(python=sys.version.split()[0],numpy=numpy.__version__,
         scipy=scipy.__version__,platform=platform.platform(),
         machine=platform.machine(),logical_cpu_count=os.cpu_count(),
         blas_threads_per_worker=1)
manifest=dict(
    status='three_full_process_timings_verified',
    processes={
        'OU':dict(seed=records['OU']['seed'],detected_events=22,
                  full_wall_seconds=records['OU']['total_wall_seconds'],
                  provenance='recovered original full run; original hardware and concurrency unverified',
                  unit_files=4800,checkpoint_sha256=digest('pilots/OU_checkpoint.json')),
        '2OU':dict(seed=records['2OU_clean']['seed'],detected_events=160,
                   full_wall_seconds=records['2OU_clean']['total_wall_seconds'],
                   provenance='clean full process remeasurement, distinct from resumed fit grid',
                   worker_processes=8,unit_files=4800,
                   checkpoint_sha256=digest('timing_remeasurement_2OU/pilots/2OU_checkpoint.json')),
        'ARFIMA':dict(seed=records['ARFIMA']['seed'],detected_events=375,
                      full_wall_seconds=records['ARFIMA']['total_wall_seconds'],
                      provenance='full generation/detection/eligibility run; zero fit-eligible panels',
                      unit_files=0,checkpoint_sha256=digest('pilots/ARFIMA_checkpoint.json'))},
    execution_environment=env,
    frozen_scientific_spec_sha256=digest('spec_v1.md'),
    frozen_core_sha256=frozen['stages/S01_protocol/attempt_02/protocol_core.py'],
    frozen_pipeline_sha256=frozen['stages/S02a_pooled/scripts/pipeline.py'],
    continuation_pipeline_sha256=digest('scripts/pipeline.py'),
    allocation=load('benchmark_allocation.json'),
    verification='timing_verification.json')
(out/'timing_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
checkpoints={path:str(digest(path)) for path in (
    'pilots/OU_checkpoint.json','pilots/OU_timing.json',
    'pilots/2OU_checkpoint.json','pilots/2OU_timing.json',
    'pilots/ARFIMA_checkpoint.json','pilots/ARFIMA_timing.json',
    'timing_remeasurement_2OU/pilots/2OU_checkpoint.json',
    'timing_remeasurement_2OU/pilots/2OU_timing.json',
    'empirical_completed.json','fit_verification.json')}
(out/'checkpoint_manifest.json').write_text(json.dumps(dict(
    status='complete',sha256=checkpoints,empirical_panel_receipts=48,
    continuation_logs='continuation_logs/',reverse_logs='reverse_logs/',
    resume_note='Missing immutable fit units were resumed with their frozen seeds; full 2OU timing was measured separately.'
),indent=2)+'\n')
print(json.dumps(dict(manifests=2,processes=3,empirical_panel_receipts=48)))
