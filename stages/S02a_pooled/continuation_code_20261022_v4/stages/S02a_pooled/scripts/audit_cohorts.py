"""Independently enumerate cohorts using explicit dates, compare frozen builder."""
import json
import numpy as np
import pandas as pd
import pipeline as p
ts,ev=p.inputs(); results=[]; allrows=[]; failed=[]
for h in p.HORIZONS:
    bymat={}
    for mat in p.MATS:
        expected={}
        for e in ev.itertuples():
            s=ts[mat]; pre=s.loc[s.index<e.onset_date].dropna().tail(250)
            future=ts.index[ts.index>=e.peak_date][:h+1]
            later=ev.loc[ev.onset_date>e.onset_date,'onset_date']
            if len(future)!=h+1 or len(pre)!=250 or (len(later) and future[-1]>=later.min()): continue
            y=s.loc[future].to_numpy()
            if not np.isfinite(y).all() or np.any(y<=0) or np.any(pre.to_numpy()<=0): continue
            baseline=sum(np.log(float(v)) for v in pre)/250
            denominator=np.log(y[0])-baseline
            if denominator<=1e-8: continue
            expected[e.event_id]=(np.log(y)-baseline)/denominator
        actual,interval,rows=p.panels(ts,ev,mat,h); allrows.extend(rows)
        if set(expected)!=set(actual): failed.append(f'{mat}/{h}: IDs disagree')
        for eid in expected.keys()&actual.keys():
            if not np.allclose(expected[eid],actual[eid],rtol=1e-11,atol=1e-11): failed.append(f'{mat}/{h}/{eid}: normalization')
        groups=p.core.overlap_groups([interval[eid] for eid in sorted(actual)]) if actual else []
        bymat[mat]=set(expected)
        results.append(dict(maturity=mat,horizon=h,event_count=len(actual),group_count=len(set(groups)),event_ids=sorted(actual)))
    common=set.intersection(*bymat.values())
    for row in results:
        if row['horizon']==h: row['common_event_count']=len(common)
pd.DataFrame(allrows).to_csv(p.OUT/'cohort_manifest.csv',index=False)
(p.OUT/'cohort_audit.json').write_text(json.dumps(dict(comparisons=12,failures=failed,cohorts=results),indent=2)+'\n')
print(json.dumps(dict(failures=failed,counts=[{k:v for k,v in row.items() if k!='event_ids'} for row in results]),indent=2))
raise SystemExit(bool(failed))
