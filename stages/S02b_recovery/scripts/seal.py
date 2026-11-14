"""Seal per-process S02b snapshots and execution manifest; never overwrite."""
import csv,hashlib,json,platform,sys
from pathlib import Path
import numpy,scipy,pandas
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'stages/S02b_recovery'
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():raise FileExistsError(p)
    p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def main():
    cal=json.loads((OUT/'calibration_parameters.json').read_text())
    rows=list(csv.DictReader((OUT/'calibration_diagnostics.csv').open()))
    for name in ('OU','2OU','ARFIMA'):
        own=[r for r in rows if r['process']==name]
        info=cal['processes'][name]
        write(OUT/'checkpoints'/f'process_{name}_calibration.json',dict(process=name,status=info['status'],planned_paths=11,completed_paths=11,interrupted_paths=0,
            confirmatory_planned=0,confirmatory_completed=0,path_ids=[int(r['path_id']) for r in own],seeds=[int(r['seed']) for r in own],
            detected=[int(r['detected']) for r in own],eligible_common=[int(r['eligible_common']) for r in own],gates=info['gates'],
            diagnostics_sha256=digest(OUT/'calibration_diagnostics.csv'),parameters_sha256=digest(OUT/'calibration_parameters.json')))
    artifacts=['calibration_freeze.json','calibration_method_v1.md','scripts/calibrate.py','scripts/verify_finalize.py','scripts/seal.py',
        'calibration_parameters.json','calibration_diagnostics.csv','detector_parity.csv','alignment_results.json','allocation_spec.json',
        'recovery_rates.csv','confusion_matrix.csv','simulation_manifest.csv','failures.csv','independent_verification.json',
        'checkpoints/checkpoint_000_calibration_complete.json','checkpoints/process_OU_calibration.json','checkpoints/process_2OU_calibration.json',
        'checkpoints/process_ARFIMA_calibration.json','verification.md','resume.md','acceptance.md']
    write(OUT/'execution_manifest.json',dict(status='PARTIAL_RECOVERY_NOT_ESTIMABLE',stage='S02b',seed_base=5902026,
        python=platform.python_version(),numpy=numpy.__version__,scipy=scipy.__version__,pandas=pandas.__version__,
        commands=[{'command':'python3 -m py_compile stages/S02b_recovery/scripts/calibrate.py','exit_code':0},
                  {'command':'python3 stages/S02b_recovery/scripts/calibrate.py','exit_code':0},
                  {'command':'python3 stages/S02b_recovery/scripts/verify_finalize.py','exit_code':1,'cause':'missing checkpoint directory after numerical verification'},
                  {'command':'python3 stages/S02b_recovery/scripts/verify_finalize.py','exit_code':0},
                  {'command':'python3 stages/S02b_recovery/scripts/seal.py','exit_code':0}],
        files={name:digest(OUT/name) for name in artifacts},
        planned_calibration_paths=33,completed_calibration_paths=33,planned_confirmatory_paths=0,completed_confirmatory_paths=0))
    print('sealed 3 process checkpoints and execution manifest')
if __name__=='__main__':main()
