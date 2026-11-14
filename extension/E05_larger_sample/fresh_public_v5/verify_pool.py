"""Post-run arithmetic audit; this is not Vera's independent verification."""
import csv, hashlib, json, math, sys
from collections import Counter
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'stages/S01_protocol/attempt_02'))
import protocol_core as core

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
f=json.loads((HERE/'freeze_v2.json').read_text())
for name,digest in f['hashes'].items(): assert sha(ROOT/name)==digest,name
p=json.loads((HERE/'prepared.json').read_text())
tr=p['trajectories']; ass=p['assignment']; clusters=sorted(set(ass.values()))
indices=('VIX9D','VIX30D','VIX3M','VIX6M','VXN','RVX','OVX','GVZ')
assert len(tr)==84 and len(clusters)==7
checked=0; valid=0; failed=0; fits=0
for draw in [None,*range(999)]:
    path=HERE/'units'/('point.json' if draw is None else f'{draw:04d}.json')
    u=json.loads(path.read_text()); assert u['freeze_sha']==f['freeze_sha'] and u['draw_id']==draw
    if draw is None: selected=clusters
    else:
        rng=np.random.default_rng(core.seed('Extension','E5','cluster_bootstrap','equal_index_pool',draw))
        selected=rng.choice(clusters,size=len(clusters),replace=True).tolist()
    assert u['selected_clusters']==selected
    mult=Counter(selected); curves={};counts={}
    for index in indices:
        expanded=[tr[k] for k in tr if k.startswith(index+'|') for _ in range(mult[ass[k]])]
        counts[index]=len(expanded)
        if expanded: curves[index]=np.mean(expanded,axis=0)
    assert u['index_event_counts']==counts
    if len(curves)<8:
        assert u['failure']=='missing_index' and u['fits']==[]
        failed+=draw is not None;checked+=1;continue
    assert u['failure'] is None
    y=np.mean([curves[i] for i in indices],axis=0)
    np.testing.assert_allclose(y,u['pooled_response'],rtol=0,atol=2e-14)
    assert len(u['fits'])==8
    for r in u['fits']:
        t=np.arange(r['start'],121);z=y[r['start']:]
        assert len(r['starts'])==32 and r['best'] is not None
        good=[]
        for s in r['starts']:
            if s['success']:
                e=core.curve(r['family'],t,np.asarray(s['parameters']))-z
                np.testing.assert_allclose(np.sum(e*e),s['sse'],rtol=2e-8,atol=1e-10)
                good.append(s['sse'])
        assert good and math.isclose(min(good),r['best']['sse'],rel_tol=1e-9,abs_tol=1e-10)
        assert math.isclose(core.score(r['best']['sse'],len(t),len(r['best']['parameters'])),r['score'],rel_tol=1e-9,abs_tol=1e-8)
        fits+=1
    valid+=draw is not None;checked+=1
summary=json.loads((HERE/'pooled_summary.json').read_text())
assert checked==1000 and summary['valid_draws']==valid and summary['failed_draws']==failed
rows=list(csv.DictReader((HERE/'pooled_family_comparison.csv').open()))
assert len(rows)==8 and all(r['interval_estimable']=='False' for r in rows)
result=dict(status='local_arithmetic_check_passed',units_checked=checked,draws_valid=valid,
    draws_failed=failed,fits_checked=fits,base_clusters=len(clusters),
    verifier_sha256=sha(HERE/'verify_pool.py'),freeze_sha=f['freeze_sha'])
(HERE/'local_verification_v2.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
