"""Finalize control-blocked checkpoint; never mutate or restart the worker."""
import datetime as dt,hashlib,json,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'stages/S02a_pooled'
snapshot=OUT/'checkpoints/control_interruption_20261021'
check=json.loads((snapshot/'fit_verification.json').read_text())
assert not check['errors']
manifest=json.loads((snapshot/'snapshot_manifest.json').read_text())
assert all(hashlib.sha256((snapshot/p).read_bytes()).hexdigest()==h for p,h in manifest['files'].items())
state=dict(status='PARTIAL',reason='Supervisor interrupted during runtime transition; native worker-stop request rejected; worker exit unconfirmed.',completed=['stage_tests','diag1','diag3','diag5','diag6'],pending=['timing_OU_complete_original_seed_after_process_reconciliation','timing_2OU','timing_ARFIMA','empirical_four_model_full_grid','final_summary_figures_and_acceptance'],failed_or_interrupted='timing_OU',worker_exit_confirmed=False,supervisor_exit_code=130,replacement_started=False,completed_synthetic_fit_files=check['units'][0]['completed_files'],snapshot='checkpoints/control_interruption_20261021',elapsed_seconds=(dt.datetime.now(dt.timezone.utc)-dt.datetime(2026,10,22,3,5,42,tzinfo=dt.timezone.utc)).total_seconds())
(OUT/'checkpoint.json').write_text(json.dumps(state,indent=2)+'\n')
shutil.copy2(snapshot/'fit_verification.json',OUT/'fit_verification.json')
shutil.copy2(snapshot/'fits.csv',OUT/'fits.csv')
subprocess.run([sys.executable,str(OUT/'scripts/finalize.py')],check=True)
results=json.loads((OUT/'results.json').read_text())
results['execution_control']=state
results['verified_fit_snapshot']=dict(path=state['snapshot'],synthetic_only=True,fit_records=check['fit_records'],optimizer_starts_checked=check['optimizer_starts_checked'],errors=check['errors'])
results['new_pooled_analysis']['verification']={'status':'not_started','note':'Only empirical cohorts checked; all completed four-family fits are synthetic OU pilot work.'}
results['limitations'].extend(['The serial supervisor exited 130; child worker exit is not confirmed.','Native stop permission was rejected; no replacement worker launched.','The saved snapshot has 96 of 1200 fit files for only the first pilot panel; it is not a full process replication.','Additional private worker files after snapshot are not verified or credited.'])
(OUT/'results.json').write_text(json.dumps(results,indent=2)+'\n')
verification=(OUT/'verification.md').read_text()
verification += '\n## Material execution-control limitation\n'+(OUT/'execution_control_issue.md').read_text()+'\n## Immutable fit snapshot\n96 completed fit files, 384 four-family fit records and 12,288 optimizer starts were independently checked on the saved snapshot copy: zero loss/score/start-selection errors. These are partial synthetic OU work only, not empirical fits or a complete timing replication. The snapshot source hash manifest passed before packaging.\n'
(OUT/'verification.md').write_text(verification)
(OUT/'acceptance.md').write_text('''# S02a acceptance report

Stage: S02a pooled diagnostics and benchmark allocation.
Status: PARTIAL; not accepted. No later stage started.

Outputs: diagnostic_reproduction.csv, results.json, verification.md,
timing_benchmark.csv and benchmark_allocation.json are checkpoint artifacts.
Zero complete timing replications; allocation N is null, not zero or frozen.

Acceptance criteria:
- S01 acceptance, independent sign-off, prerequisites and source hashes: MET.
- Historical diagnostic claims reproduced/refuted/qualified/unsupported with evidence: MET; 39 reconciliation rows, none pending.
- Four-maturity coverage and independent cohort checks: MET; primary event counts 9/19/12/12, common count 9.
- Frozen four-model empirical analysis, all starts, baselines and bootstrap sensitivities: NOT MET; not executed.
- Three full simulated timing replications and independently frozen allocation: NOT MET; first pilot incomplete.
- Full numerical/figure/uncertainty verification: NOT MET. Partial synthetic snapshot only: 384 model fits, 12,288 starts checked, zero errors.
- Export/read-back: see actual Buddy save and read-back receipts; no acceptance inferred from successful saves.

Decisions: preserved scientific rules and seeds; stopped the replacement launch
when old-worker exit could not be established; saved only independently verified
snapshot evidence as credited results. Proposed parallel code is unexecuted.

Issues: supervisor interrupted (exit 130); child exit unconfirmed after native
stop permission was rejected. Child retains its absolute analysis guard, but
verified shutdown is not claimed. Historical NaN HAR comparisons and unsupported
full-sample ACF claim remain disclosed. See execution_control_issue.md.

Next stage ready: NO. S03 requires S02a PASS. S02a needs a new explicit trigger
for resumed work after worker-state reconciliation; S02b and S03 remain unstarted.
''')
paths=[p for p in OUT.iterdir() if p.is_file() and p.name!='checkpoint_manifest.json']
paths+=list(snapshot.rglob('*'))
frozen={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file() and '__pycache__' not in p.parts}
(OUT/'checkpoint_manifest.json').write_text(json.dumps(dict(status='PARTIAL',files=frozen,worker_exit_confirmed=False,scope='Immutable credited snapshot and top-level checkpoint files; live private units excluded.'),indent=2)+'\n')
print(json.dumps(dict(status='PARTIAL',snapshot_fits=check['fit_records'],snapshot_starts=check['optimizer_starts_checked'],files=len(frozen),worker_exit_confirmed=False)))
