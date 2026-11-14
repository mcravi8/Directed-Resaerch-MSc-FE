"""Frozen, review-informed E5 equal-index pooled correction."""
import argparse, csv, hashlib, json, math, os, signal, sys, time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import scipy
from scipy.optimize import least_squares

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'stages/S01_protocol/attempt_02'))
import protocol_core as core
SOURCE=HERE
INDICES=('VIX9D','VIX30D','VIX3M','VIX6M','VXN','RVX','OVX','GVZ')
FAMILIES=('E1','E2','PL','SE')
STARTS=(1,5)
RUN_STARTED=time.monotonic()

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat()
def atomic(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
    os.replace(temp,path)
def frozen_paths():
    return [HERE/'spec_v2.md',HERE/'run_pool.py',SOURCE/'prepared.json',SOURCE/'execution_freeze.json',ROOT/'stages/S01_protocol/attempt_02/protocol_core.py']
def freeze():
    if (HERE/'freeze_v2.json').exists(): raise RuntimeError('freeze already exists')
    d={'frozen_at_utc':now(),'status':'review-informed new correction freeze, before pooled outcomes',
       'input_handoff':'public_run_complete_v1','seed':core.SEED,
       'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,
       'hashes':{str(p.relative_to(ROOT)):sha(p) for p in frozen_paths()}}
    d['freeze_sha']=hashlib.sha256(json.dumps(d,sort_keys=True).encode()).hexdigest()
    atomic(HERE/'freeze_v2.json',d)
    return d
def check():
    d=json.loads((HERE/'freeze_v2.json').read_text())
    for name,h in d['hashes'].items():
        if sha(ROOT/name)!=h: raise RuntimeError('changed frozen input: '+name)
    return d
def prep():
    d=json.loads((SOURCE/'prepared.json').read_text())
    tr=d['trajectories']; ass=d['assignment']
    assert set(ass)==set(tr) and len(tr)==84 and len(set(ass.values()))==7
    assert set(k.split('|')[0] for k in tr)==set(INDICES)
    return tr,ass,sorted(set(ass.values()))
def fit(family,t,y,start,draw):
    lo,hi=(np.asarray(z,float) for z in core.BOUNDS[family])
    rng=np.random.default_rng(core.seed('Extension','E5','equal_index_pool',start,family,draw))
    starts=[]; best=None
    for i in range(32):
        x=lo+(hi-lo)*rng.random(len(lo))
        for j in {'E1':[2],'E2':[3,4],'PL':[2,3],'SE':[2]}[family]:
            x[j]=math.exp(rng.uniform(math.log(lo[j]),math.log(hi[j])))
        try:
            sol=least_squares(lambda p:core.curve(family,t,core.ordered(family,p))-y,x,
                bounds=(lo,hi),method='trf',ftol=1e-8,xtol=1e-8,gtol=1e-8,max_nfev=5000)
            sse=float(np.sum(sol.fun**2)); pars=core.ordered(family,sol.x).tolist()
            ok=bool(sol.success and np.isfinite(sse) and np.all(np.isfinite(pars)))
            rec=dict(start_id=i,success=ok,status=int(sol.status),nfev=int(sol.nfev),
                sse=sse,parameters=pars,active_mask=sol.active_mask.tolist())
        except Exception as e: rec=dict(start_id=i,success=False,error=repr(e))
        starts.append(rec)
        if rec['success'] and (best is None or rec['sse']<best['sse']): best=rec
    return dict(family=family,start=start,best=best,starts=starts,
        score=core.score(best['sse'],len(t),len(lo)) if best else None,
        tau=best['parameters'][2] if best and family=='E1' else None)
def unit(draw,tr,ass,clusters,freeze_sha):
    if draw is None: selected=clusters.copy()
    else:
        rng=np.random.default_rng(core.seed('Extension','E5','cluster_bootstrap','equal_index_pool',draw))
        selected=rng.choice(clusters,size=len(clusters),replace=True).tolist()
    mult=dict(Counter(selected)); curves={}; counts={}
    for index in INDICES:
        keys=[k for k in tr if k.startswith(index+'|')]
        expanded=[tr[k] for k in keys for _ in range(mult.get(ass[k],0))]
        counts[index]=len(expanded)
        if expanded: curves[index]=np.mean(np.asarray(expanded),axis=0)
    obj=dict(draw_id=draw,freeze_sha=freeze_sha,seed=core.SEED,selected_clusters=selected,
        multiplicity=mult,index_event_counts=counts,base_clusters=len(clusters),
        included_indices=list(INDICES),created_at_utc=now())
    missing=[i for i in INDICES if i not in curves]
    if missing: obj.update(failure='missing_index',missing_indices=missing,fits=[]);return obj
    y=np.mean(np.asarray([curves[i] for i in INDICES]),axis=0)
    obj.update(failure=None,pooled_response=y.tolist(),fits=[])
    for start in STARTS:
        for family in FAMILIES:
            obj['fits'].append(fit(family,np.arange(start,121),y[start:],start,draw))
    return obj
def checkpoint(started):
    paths=sorted((HERE/'units').glob('*.json'))
    done=sum(p.name!='point.json' for p in paths)
    atomic(HERE/'checkpoint.json',dict(updated_at_utc=now(),freeze_sha=check()['freeze_sha'],
        started_at_utc=started,elapsed_seconds=time.monotonic()-RUN_STARTED,
        completed_draw_units=done,remaining_draw_units=999-done,point_done=(HERE/'units/point.json').exists(),
        unit_ids=[p.stem for p in paths]))
def run():
    f=check();tr,ass,clusters=prep();started=now()
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('7h45 watchdog')))
    signal.alarm(7*3600+45*60)
    for draw in [None,*range(999)]:
        if time.monotonic()-RUN_STARTED>=7*3600: break
        path=HERE/'units'/('point.json' if draw is None else f'{draw:04d}.json')
        if path.exists():
            old=json.loads(path.read_text())
            if old['freeze_sha']!=f['freeze_sha'] or old['draw_id']!=draw: raise RuntimeError('stale unit '+str(path))
            continue
        atomic(path,unit(draw,tr,ass,clusters,f['freeze_sha']))
        checkpoint(started)
    signal.alarm(0)
    return summarize()
def summarize():
    f=check();tr,ass,clusters=prep();paths=[HERE/'units'/f'{i:04d}.json' for i in range(999)]
    if not (HERE/'units/point.json').exists() or not all(p.exists() for p in paths):
        return {'status':'incomplete','saved_draws':sum(p.exists() for p in paths)}
    point=json.loads((HERE/'units/point.json').read_text())
    draws=[json.loads(p.read_text()) for p in paths]
    for i,d in enumerate(draws):
        if d['draw_id']!=i or d['freeze_sha']!=f['freeze_sha']: raise RuntimeError('unit ID/freeze mismatch')
        rng=np.random.default_rng(core.seed('Extension','E5','cluster_bootstrap','equal_index_pool',i))
        if d['selected_clusters']!=rng.choice(clusters,size=len(clusters),replace=True).tolist(): raise RuntimeError('draw mismatch')
    rows=[]
    for fitrow in point['fits']:
        start=fitrow['start']; family=fitrow['family']; best=fitrow['best']
        vals=[next((r for r in d['fits'] if r['start']==start and r['family']==family),None) for d in draws]
        valid=sum(r is not None and r['best'] is not None for r in vals)
        rows.append(dict(start=start,family=family,point_score=fitrow['score'],
            point_sse=best['sse'] if best else None,parameters=json.dumps(best['parameters']) if best else '',
            E1_tau=fitrow['tau'],point_bound_active=bool(best and any(best['active_mask'])),
            successful_point_starts=sum(r['success'] for r in fitrow['starts']),valid_draw_fits=valid,
            failed_draw_fits=999-valid,base_clusters=7,interval_estimable=False,interval_reason='base_clusters_below_10'))
    with (HERE/'pooled_family_comparison.csv').open('w',newline='') as h:
        w=csv.DictWriter(h,rows[0]);w.writeheader();w.writerows(rows)
    valid_draws=sum(d['failure'] is None for d in draws)
    failures=Counter(d['failure'] for d in draws)
    summary=dict(status='complete',freeze_sha=f['freeze_sha'],indices=list(INDICES),
        index_event_counts=point['index_event_counts'],base_clusters=7,point_fit_count=len(point['fits']),
        expected_draws=999,saved_draws=len(draws),valid_draws=valid_draws,
        failed_draws=999-valid_draws,draw_failure_reasons=dict(failures),
        start5_winner=min((r for r in rows if r['start']==5),key=lambda r:r['point_score'])['family'],
        inferential_status='unestimable: 7 observed clusters < 10 required')
    atomic(HERE/'pooled_summary.json',summary)
    return summary
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['freeze','run','summarize']);a=ap.parse_args()
    print(json.dumps(freeze() if a.mode=='freeze' else run() if a.mode=='run' else summarize(),indent=2))
