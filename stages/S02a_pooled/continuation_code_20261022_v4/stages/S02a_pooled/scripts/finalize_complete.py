"""Package completed S02a evidence; N=0 remains PARTIAL by frozen protocol."""
import collections
import csv
import hashlib
import json
import shutil
from pathlib import Path

out=Path(__file__).resolve().parents[1]
read=lambda name: json.loads((out/name).read_text())
hashof=lambda name: hashlib.sha256((out/name).read_bytes()).hexdigest()
receipts=read('empirical_completed.json')
assert len(receipts)==48 and all(r['fit_files']==3000 for r in receipts)
verified=read('fit_verification.json')
assert not verified['errors'] and len(verified['units'])==56
assert all(r['complete'] for r in verified['units'])
summary=read('pooled_summary.json')
allocation=read('benchmark_allocation.json')
timing=read('timing_verification.json')
cohorts=read('cohort_audit.json')
assert not cohorts['failures'] and cohorts['comparisons']==12
assert allocation['N']==0 and allocation['full_process_timings_measured']==3
assert timing['status']=='three_full_timings_verified'
with (out/'diagnostic_reproduction.csv').open() as file:
    diagnostic=list(csv.DictReader(file))
assert len(diagnostic)==39 and not any(r['status']=='pending' for r in diagnostic)
with (out/'timing_benchmark.csv').open() as file:
    benchmark=list(csv.DictReader(file))
assert len(benchmark)==3 and all(r['full_wall_seconds'] for r in benchmark)

primary={}
for row in summary['summaries']:
    unit=row['unit']
    if not (unit.startswith('empirical_120_') and unit.endswith('_full_event')):
        continue
    if row['zero_baseline'] or row['start']!=1:
        continue
    maturity=unit.split('_')[2]
    models=row['models']
    winner=min(models,key=lambda name:models[name]['score']) if models else None
    primary[maturity]={
        'event_count':row['events'], 'group_count':row['groups'],
        'point_score_winner':winner,
        'models':{name:{'score':fit['score'],'sse':fit['sse'],
                        'parameters':fit['parameters'],
                        'successful_bootstrap_refits':fit['complete_refits'],
                        'event_bootstrap_win_fraction':fit['win_fraction'],
                        'parameter_percentiles_2_5_50_97_5':fit['parameter_percentiles_2_5_50_97_5']}
                  for name,fit in models.items()},
    }
assert set(primary)=={'VIX9D','VIX30D','VIX3M','VIX6M'}

result={
    'stage':'S02a', 'status':'PARTIAL', 'analysis_complete':True,
    'benchmark_complete':True,'acceptance_pending':True,
    'reason':'Frozen S02b allocation yields N=0; S02a cannot meet its PASS allocation gate.',
    'diagnostic_claims':dict(total=len(diagnostic),statuses=dict(collections.Counter(r['status'] for r in diagnostic))),
    'cohort_audit':dict(comparisons=cohorts['comparisons'],failures=cohorts['failures'],
                        primary_120_full_counts={r['maturity']:r['event_count'] for r in cohorts['cohorts'] if r['horizon']==120}),
    'pooled_analysis':dict(empirical_panels=len(receipts),models=['E1','E2','PL','SE'],
                           fit_records=verified['fit_records'],
                           optimizer_starts_checked=verified['optimizer_starts_checked'],
                           verification_errors=len(verified['errors']),
                           summary='pooled_summary.json',fits='fits_complete.csv.gz',
                           fits_uncompressed_sha256=hashof('fits.csv'),
                           number_ledger='number_ledger.csv',figures='figures/',
                           primary_120_full_event=primary,
                           paired_common_contrasts=len(summary['paired_maturity_contrasts'])),
    'recovered_ou':read('recovered_ou_audit.json'),
    'timing_benchmark':benchmark,
    'timing_verification':'timing_verification.json',
    'allocation':allocation,
    'limitations':[
        'N=0 under the frozen allocation formula; no S02b recovery grid was run.',
        'Synthetic timing outcomes are not empirical VIX family evidence.',
        'ARFIMA timing has zero eligible 120-session panels; no fit was invented.',
        'Original PID 79806 lifecycle was not verifiable from this isolated task.',
        'Vera independent acceptance and Chief final acceptance are pending.',
    ],
    'provenance_sha256':{name:hashof(name) for name in (
        'diagnostic_reproduction.csv','timing_benchmark.csv','benchmark_allocation.json',
        'timing_verification.json','fit_verification.json','pooled_summary.json',
        'fits.csv','fits_complete.csv.gz','number_ledger.csv','empirical_completed.json')},
    'decisions':['D-49','D-50'],
    'next':'No later stage started; Chief determines how to handle the frozen N=0 gate.',
}
(out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')

prior=out/'verification_checkpoint_prior.md'
if not prior.exists():
    shutil.copy2(out/'verification.md',prior)
lines=[
    '# S02a verification', '',
    'Status: PARTIAL under the frozen N=0 allocation gate. Empirical computation and all three full process timings are complete. Vera and Chief acceptance remain pending.', '',
    '## Reproduction and pooled analysis',
    f'- Historical diagnostic claims: {len(diagnostic)} reconciled; status counts {dict(collections.Counter(r["status"] for r in diagnostic))}. Unsupported full-sample ACF claim remains explicitly unsupported.',
    '- Twelve independent maturity/horizon cohort comparisons passed with zero failures. Primary 120-session full cohorts: 9/19/12/12 events for VIX9D/VIX30D/VIX3M/VIX6M.',
    f'- Forty-eight empirical event/group, full/common panels completed with 499 resamples each, all four frozen models, starts 1/3/5 and both baseline rules. {verified["fit_records"]} fit records and {verified["optimizer_starts_checked"]} optimizer starts independently checked; zero numerical errors.',
    f'- Pooled summary has {len(summary["summaries"])} model comparisons and {len(summary["paired_maturity_contrasts"])} paired common-cohort contrasts. See pooled_summary.json, fits_complete.csv.gz, number_ledger.csv and figures/. The gzip archive expands to fits.csv with the SHA-256 in results.json.', '',
    '## Recovered OU and synthetic timing',
    '- Original OU recovery: 22 events, seed 9503688850173507821, 15958.618400167004 measured seconds (4.43295 hours). Four eligible panels contain 4800 fit files; 19200 model fits and 614400 optimizer starts independently checked with zero errors. No OU pilot rerun.',
    '- The earlier 2OU unit grid completed across commands. Its incremental 4608.919532958011-second clock is not used as a full process timing. A separate clean replication with the identical seed and frozen numerical rules supplies T_2OU; the clean 4800 files were independently verified.',
    '- ARFIMA detected 375 events; no horizon reached ten common eligible events. Its full timing covers generation, detection, eligibility and checkpoint writing; no ineligible fit was invented.',
    f'- Three full timings and allocation arithmetic independently checked. Denominator {allocation["denominator_seconds"]:.6f} seconds; N=0 under the frozen rule. The OU contribution alone is 95751.71040100203 seconds, above 21600.', '',
    '## Provenance and limits',
    '- Frozen scientific models, starts, draws, inputs and seeds were preserved. The continuation runtime disabled only the historical S02a clock guard pursuant to D-49; pipeline_frozen.py retains the original frozen hash. The Python spawn main guard was an execution repair recorded in D-50.',
    '- Original PID 79806 containment cannot be inferred from this isolated workspace. No unsupported PID claim was used for scientific conclusions.',
    '- Commands and exit codes: continuation_logs/*.json and continuation_logs/*.log; seed and pipeline details: timing_manifest.json, frozen_manifest.json, source manifest and unit panel records. File hashes are in results.json.',
    '- Connected-folder save and read-back receipts are recorded after Buddy export; this local file alone is not proof of export.',
    '- S02b and later stages were not run. The N=0 rule makes S02a PARTIAL; Chief must resolve the scientific plan gate before any later-stage execution.',
]
(out/'verification.md').write_text('\n'.join(lines)+'\n')
(out/'acceptance.md').write_text(
    'Stage | Status | Outputs | Key numbers | Acceptance | Decisions | Issues | Next\n'
    '---|---|---|---|---|---|---|---\n'
    f'S02a | PARTIAL (N=0) | Five required files plus pooled_summary.json, fits_complete.csv.gz, figures/ | '
    f'{len(receipts)} empirical panels; {verified["fit_records"]} fits; 3 full timings; N=0 | '
    'Computational evidence complete; frozen N=0 gate prevents PASS; Vera/Chief pending | '
    'D-49 and D-50 | Original PID lifecycle unverified | No later stage started\n')
print(json.dumps(dict(status='PARTIAL',empirical_panels=len(receipts),
                      full_timings=3,N=0,fit_records=verified['fit_records'])))
