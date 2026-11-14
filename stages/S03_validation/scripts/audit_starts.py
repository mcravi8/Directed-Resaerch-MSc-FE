"""Deterministically replay frozen S03 fits to retain every optimizer start and compare best fits."""
import csv, json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd
import run

ROOT=run.ROOT; OUT=run.OUT
FIELDS=['event_id','maturity','design','model','start','success','status','nfev','sse','parameters','active_mask','boundary_hit']
def main():
    ts,ev=run.inputs(); original=pd.read_csv(OUT/'curve_fits.csv')
    csvpath=OUT/'curve_starts.csv'
    completed=set()
    if csvpath.exists():
        old=pd.read_csv(csvpath,usecols=['event_id','maturity','design','model','start'])
        counts=old.groupby(['event_id','maturity','design','model']).start.nunique()
        completed={k for k,n in counts.items() if n==32}
        if any(n!=32 for n in counts):raise ValueError('incomplete panel in prior start audit; preserve file and investigate')
    with csvpath.open('a',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS)
        if not completed:writer.writeheader()
        checked=0
        for design in run.DESIGNS:
            for _,e in ev.iterrows():
                for mat in run.MATS:
                    b,p,reason=run.event_panel(ts,ev,e,mat,design)
                    if reason:continue
                    s,dates,on,a,pre,vals,z=p
                    _,tr0,tr1,te0,te1=run.DESIGNS[design]
                    for fam in run.FAMILIES:
                        key=(e.event_id,mat,design,fam)
                        if key in completed:checked+=1;continue
                        fit=run.core.fit(fam,np.arange(tr0,tr1+1),z[tr0:tr1+1],['S03',design,e.event_id,mat,tr0,tr1])
                        saved=original[(original.event_id==e.event_id)&(original.maturity==mat)&(original.design==design)&(original.model==fam)]
                        assert len(saved)==1 and fit['best'] is not None
                        assert np.allclose(fit['best']['parameters'],json.loads(saved.iloc[0].parameters),rtol=1e-12,atol=1e-12)
                        assert math.isclose(fit['best']['sse'],saved.iloc[0].sse,rel_tol=1e-12,abs_tol=1e-12)
                        for rec in fit['starts']:
                            writer.writerow(dict(event_id=e.event_id,maturity=mat,design=design,model=fam,start=rec['start'],success=rec['success'],status=rec['status'],nfev=rec['nfev'],sse=rec['sse'],parameters=json.dumps(rec['parameters']),active_mask=json.dumps(rec['active_mask']),boundary_hit=any(rec['active_mask'])))
                        f.flush();checked+=1
        assert checked==len(original)
    starts=pd.read_csv(csvpath)
    assert len(starts)==len(original)*32
    groups=starts.groupby(['event_id','maturity','design','model'])
    for key,g in groups:
        r=original[(original.event_id==key[0])&(original.maturity==key[1])&(original.design==key[2])&(original.model==key[3])].iloc[0]
        assert g.success.sum()==r.successful_starts and g.boundary_hit.sum()==r.boundary_starts
    result={'status':'PASS','fits':len(original),'starts':len(starts),'failures':int((~starts.success).sum()),'boundary_contacts':int(starts.boundary_hit.sum()),'best_fit_match':'all'}
    run.write_json(OUT/'start_audit.json',result);print(json.dumps(result))
if __name__=='__main__':main()
