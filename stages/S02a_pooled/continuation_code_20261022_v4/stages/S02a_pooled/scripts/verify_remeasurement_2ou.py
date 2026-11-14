"""Independent algebraic checks of all completed fit files; no optimizer call."""
import csv,json,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]; OUT=ROOT/'stages/S02a_pooled/timing_remeasurement_2OU'
def predict(f,t,q):
    b,a=q[:2]
    if f=='E1': return b+a*np.exp(-t/q[2])
    if f=='E2': return b+a*(q[2]*np.exp(-t/q[3])+(1-q[2])*np.exp(-t/q[4]))
    if f=='PL': return b+a*np.power(1+t/q[2],-q[3])
    if f=='SE': return b+a*np.exp(-np.power(t/q[2],q[3]))
    raise ValueError(f)
rows=[]; errors=[]; units=[]; starts_count=0
for folder in sorted((OUT/'units').glob('*')):
    panel_path=folder/'panel.json'
    if not panel_path.exists(): continue
    panel=json.loads(panel_path.read_text()); z=np.asarray(panel['trajectories']); checked=0
    for file in sorted(folder.glob('r*_b*_s*.json')):
        r=json.loads(file.read_text()); t=np.arange(r['start'],z.shape[1]); y=z[r['indices']].mean(axis=0)[r['start']:]
        for family,fit in r['models'].items():
            if len(fit['starts'])!=32: errors.append(f'{file}: start count')
            finite_success=[]
            for s in fit['starts']:
                starts_count+=1; q=s['parameters']; loss=float(np.sum((predict(family,t,q)-y)**2))
                if not np.isclose(loss,s['sse'],rtol=1e-8,atol=1e-10): errors.append(f'{file}/{family}/{s["start"]}: loss')
                if family=='E2' and q[3]>q[4]: errors.append(f'{file}/{family}: order')
                if r['zero_baseline'] and q[0]!=0: errors.append(f'{file}/{family}: zero baseline')
                if s['success']: finite_success.append(s['sse'])
            best=fit['best']
            if best:
                if best['sse']!=min(finite_success): errors.append(f'{file}/{family}: best selection')
                k={'E1':3,'E2':5,'PL':4,'SE':4}[family]-int(r['zero_baseline']);n=len(t)
                score=n*math.log(best['sse']/n)+2*k
                if not np.isclose(score,fit['score'],rtol=1e-10,atol=1e-9): errors.append(f'{file}/{family}: score')
                rows.append(dict(unit=folder.name,replicate=r['replicate'],zero_baseline=r['zero_baseline'],start=r['start'],family=family,sse=best['sse'],score=score,parameters=json.dumps(best['parameters']),failed_starts=sum(not s['success'] for s in fit['starts']),boundary_hits=sum(any(s['active_mask']) for s in fit['starts'])))
            elif finite_success: errors.append(f'{file}/{family}: discarded successful fit')
        checked+=1
    units.append(dict(unit=folder.name,completed_files=checked,required_files=(panel['draws']+1)*6,event_count=panel['event_count'],group_count=panel['group_count'],complete=checked==(panel['draws']+1)*6))
if rows:
    with (OUT/'fits.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
result=dict(fit_records=len(rows),optimizer_starts_checked=starts_count,errors=errors,units=units)
(OUT/'fit_verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(fit_records=len(rows),optimizer_starts_checked=starts_count,error_count=len(errors),units=units),indent=2))
raise SystemExit(bool(errors))
