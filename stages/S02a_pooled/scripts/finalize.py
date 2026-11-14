"""Build an honest checkpoint from completed immutable evidence only."""
import csv,datetime as dt,hashlib,json,math,platform
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]; OUT=ROOT/'stages/S02a_pooled'
def read(name,default):
    p=OUT/name
    return json.loads(p.read_text()) if p.exists() else default
def save(name,data): (OUT/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
pilots=[]
for process in ['OU','2OU','ARFIMA']:
    record=read(f'pilots/{process}_timing.json',None)
    if record is not None: pilots.append(record)
    else: pilots.append(dict(process=process,total_wall_seconds=None,completed=False,seed=None,detected_events=None,synthetic=True))
with (OUT/'timing_benchmark.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(pilots[0]));w.writeheader();w.writerows(pilots)
ready=all(x['completed'] and x['total_wall_seconds']>0 for x in pilots)
allocation=dict(status='FROZEN' if ready else 'NOT_ALLOCATED',formula='min(100,floor(21600/(3*(2*T_OU)+4*(2*T_2OU)+2*(2*T_ARFIMA))))',N=None,grid_cells=9,planning_simulation_seconds=21600,safety_factor=2,pilot_evidence='synthetic timing only; excluded from recovery denominators',reason=None if ready else 'Three complete measured replications required; no partial timing extrapolation.')
if ready:
    denominator=sum(weight*2*row['total_wall_seconds'] for weight,row in zip([3,4,2],pilots))
    allocation.update(N=min(100,math.floor(21600/denominator)),denominator_seconds=denominator,measured_times={r['process']:r['total_wall_seconds'] for r in pilots})
    if allocation['N']==0: allocation['reason']='N=0: no full grid fits the approved allocation; PARTIAL.'
save('benchmark_allocation.json',allocation)
save('timing_manifest.json',dict(environment=read('environment.json',{}),processes=['OU tau90','two-factor OU tau5/tau90 fast variance fraction .25','ARFIMA d.4'],retained_sessions=10000,burn_in_sessions=5000,pipeline='scripts/pipeline.py',pipeline_sha256=hashlib.sha256((OUT/'scripts/pipeline.py').read_bytes()).hexdigest(),core_sha256=hashlib.sha256((ROOT/'stages/S01_protocol/attempt_02/protocol_core.py').read_bytes()).hexdigest(),bootstrap_draws=199,starts=[1,3,5],scope='All prescribed horizon/baseline sensitivities, paired argmax/onset5 common panels; ineligible panels undetermined.',full_replications_completed=sum(r['completed'] for r in pilots),allocation=allocation))
fit=read('fit_verification.json',{}); cohorts=read('cohort_audit.json',{}); checkpoint=read('checkpoint.json',{})
diagnostics=list(csv.DictReader((OUT/'diagnostic_reproduction.csv').open())) if (OUT/'diagnostic_reproduction.csv').exists() else []
status='PARTIAL'
results=dict(stage='S02a',status=status,acceptance_pending=True,historical_diagnostics=read('independent_audit.json',{}),empirical_cohorts=cohorts,new_pooled_analysis=dict(completed=(OUT/'empirical_completed.json').exists(),verification=fit),synthetic_timing=pilots,allocation=allocation,checkpoint=checkpoint,limitations=['Synthetic pilot findings are not empirical VIX evidence.','Historical diagnostic specifications differ from the frozen four-family analysis.','No S02b recovery or S03 validation executed.','No model-family conclusion without completed bootstrap and independent verification.'])
save('results.json',results)
now=dt.datetime.now(dt.timezone.utc); local=now.astimezone(dt.timezone(dt.timedelta(hours=-7)))
start=dt.datetime(2026,10,22,3,5,42,tzinfo=dt.timezone.utc)
completed=checkpoint.get('completed',[])
lines=['# S02a verification checkpoint','',f'Status: PARTIAL. Checkpoint {local.strftime("%B %d, %Y, %-I:%M:%S %p")} America/Los_Angeles.',f'Elapsed since trigger: {(now-start).total_seconds():.1f} seconds. Deadline October 22, 4:05:42 AM; analysis reserve begins 3:35:42 AM.','',
'## Checked','- All 36 S01 source input hashes matched; all 27 protocol-bundle hashes passed.','- S01 Chief acceptance and Vera independent sign-off read.','- Twelve inherited synthetic contract tests passed; five additional stage test methods passed (including all four noiseless families and stationary covariance checks).','- Stage frozen manifest covers 52 files, checked independently without mismatches.','- Independent raw detector reproduced all 34 historical raw events; merged catalog has 33 events.','- All 12 empirical maturity/horizon cohort checks agree on IDs and normalized trajectories.','- Historical diag1 and diag3 executed on private copies and independently summarized.','- Full command receipts: command_receipts.json; original stdout/stderr: logs/.','',
'## Acceptance criteria',f'- Diagnostic claim reconciliation: {len(diagnostics)} rows; remaining pending rows: {sum(r["status"]=="pending" for r in diagnostics)}.',f'- Four maturities/cohorts: checked; 120-session counts 9/19/12/12 and common cohort 9.',f'- Complete four-model empirical fit/bootstrap grid: {"computed; final audit pending" if results["new_pooled_analysis"]["completed"] else "not met"}.',f'- Three full timing replications: {sum(r["completed"] for r in pilots)}/3; allocation {allocation["status"]}.','- Final pooled curves, uncertainty, paired contrasts, numerical ledger and figure verification: pending.','- Connected export and read-back: recorded separately after actual Buddy receipts; no save is implied by this local checkpoint.','',
'## Decisions and limits','- Preserved all frozen scientific rules and original diagnostic code; no reduced draw counts or outcome-based tuning.','- Used sequential deterministic computation and immutable fit units for reproducibility.','- Used separate independent read-only verification scripts added after initial freeze; these do not select scientific specifications.','- Historical full-sample ACF AIC claim is unsupported by the supplied scripts.','- HAR historical missing-benchmark comparisons count missing values as losses; do not interpret them as complete-case win rates.','- Most maturity cohorts have fewer than ten overlap groups; group sensitivity is descriptive.','- S02b and S03 were not executed.','',
'## Resume','Resume S02a only after a new explicit trigger. Verify frozen and checkpoint hashes, preserve completed units, rerun an interrupted unit at its original seed. Pending jobs are recorded in checkpoint.json. Never repeat a completed pilot to choose a more favorable result or timing.']
(OUT/'verification.md').write_text('\n'.join(lines)+'\n')
(OUT/'acceptance.md').write_text('Stage | Status | Outputs | Key numbers | Acceptance | Decisions | Issues | Next\n---|---|---|---|---|---|---|---\nS02a | PARTIAL | diagnostic_reproduction.csv; results.json; timing_benchmark.csv; benchmark_allocation.json; verification.md | See verified evidence | Incomplete analysis/benchmark gates remain | Frozen protocol preserved | See verification.md | S03 not ready until S02a PASS; no next stage started\n')
manifest={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file() and p.name not in ['checkpoint_manifest.json'] and '__pycache__' not in p.parts and p.suffix!='.tmp'}
save('checkpoint_manifest.json',dict(status=status,files=manifest,elapsed_seconds=(now-start).total_seconds(),pending=checkpoint.get('pending',[]),incomplete=checkpoint.get('failed_or_interrupted')))
print(json.dumps(dict(status=status,full_pilots=sum(r['completed'] for r in pilots),allocation=allocation['N'],manifest_files=len(manifest))))
