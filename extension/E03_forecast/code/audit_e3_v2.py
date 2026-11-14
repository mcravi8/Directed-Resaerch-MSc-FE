"""Independent prefix and changed-suffix prediction audit for frozen E3 outputs."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from importlib.machinery import SourceFileLoader
e=SourceFileLoader('frozen_runner','extension/E03_forecast/code/run_e3.py').load_module()
root=Path('extension/E03_forecast')
manifest=json.loads((root/'input_manifest.json').read_text())
assert e.sha(e.SPEC)==manifest['spec_sha256']
assert e.sha(e.__file__)==manifest['code_sha256']
for n,s in manifest['sources'].items():assert e.sha(root/'raw'/(n+'.csv'))==s['sha256']
pred=pd.read_csv(root/'predictions.csv');ledger=pd.read_csv(root/'origin_dependency_ledger.csv').fillna('')
panel={n:e.parse(n)[:2] for n in e.NAMES}
vd,vv=panel['VIX30D'];rng=np.random.Generator(np.random.PCG64(e.seed('Extension','E3','truncation_audit')))
def rebuild_features(dates,values):
    """Independent state traversal, without calling the production feature routine."""
    states=[];last=None;maximum=None;baseline=None;onsets=[]
    for i,(dt,val) in enumerate(zip(dates,values)):
        fire=(i>0 and val-values[i-1]>=7) or (i>=5 and val-values[i-5]>=10)
        if fire and (last is None or (dt-dates[last]).days>30):
            last=i;maximum=val;baseline=float(np.mean(values[i-20:i])) if i>=20 else None
            onsets.append(i)
        if last is not None:maximum=max(maximum,val)
        active=last is not None and i-last<120
        if active and baseline is not None and i>=5:
            state=(1.,float(np.log(maximum/baseline)),float(i-last),float(np.log(val/values[i-5])))
        elif active:state=(1.,np.nan,np.nan,np.nan)
        else:state=(0.,0.,0.,0.)
        states.append(state)
    return np.asarray(states),onsets
def rebuilt_model(rows,h,cutoff):
    """Independent annual-fold OLS using numpy lstsq and direct target indexing."""
    date=np.asarray([r['date'] for r in rows]); y=np.asarray([r['target'][h] if 'target' in r else np.nan for r in rows]);
    eligible=np.asarray(['har' in r for r in rows]);end=np.asarray([date[i+h] if i+h<len(rows) else pd.NaT for i in range(len(rows))]);
    mature=eligible & np.isfinite(y) & (date<=cutoff) & (end<=cutoff)
    complete_years=[cutoff.year-3,cutoff.year-2,cutoff.year-1]
    if cutoff.month==12 and cutoff.day==31:complete_years=[cutoff.year-2,cutoff.year-1,cutoff.year]
    def x_for(indices,tau):
        z=np.asarray([rows[i]['har'] for i in indices]);c=np.asarray([rows[i]['core'] for i in indices])
        return np.column_stack((z,c[:,0],c[:,1],c[:,2],c[:,0]*np.exp(-c[:,2]/tau),c[:,3]))
    def fit(x,z):
        coef,_,rank,_=np.linalg.lstsq(x,z,rcond=max(x.shape)*np.finfo(float).eps)
        return coef if rank==x.shape[1] and len(z)>=x.shape[1]+1 else None
    scores={tau:[] for tau in e.TAUS}
    for yr in complete_years:
        jan=pd.Timestamp(year=yr,month=1,day=1);dec=pd.Timestamp(year=yr,month=12,day=31)
        tr=np.where(mature & (end<jan))[0];va=np.where(mature & (date>=jan)&(date<=dec)&(end<=dec))[0]
        if not len(tr) or not len(va):return None
        for tau in e.TAUS:
            beta=fit(x_for(tr,tau),y[tr]);
            if beta is None:return None
            scores[tau].extend((y[va]-x_for(va,tau)@beta)**2)
    best=min(e.TAUS,key=lambda tau:(float(np.mean(scores[tau])),tau))
    tr=np.where(mature)[0];bh=fit(np.asarray([rows[i]['har'] for i in tr]),y[tr]);bx=fit(x_for(tr,best),y[tr])
    return (best,bh,bx) if bh is not None and bx is not None else None
def rebuilt_rows(mdates,mvalues,vdates,vvalues):
    states,_=rebuild_features(vdates,vvalues);lookup={dt:i for i,dt in enumerate(vdates)}
    x=np.log(mvalues);rows=[];mv=mdates.values;vvdates=vdates.values
    for i,dt in enumerate(mdates):
        j=lookup.get(dt,-1);r=dict(date=dt)
        if i>=21 and j>=21 and np.all(np.isfinite(states[j])) and np.array_equal(mv[i-21:i+1],vvdates[j-21:j+1]):
            r['har']=np.array([1.,x[i],x[i-4:i+1].mean(),x[i-21:i+1].mean()]);r['core']=states[j]
            r['target']={h:float(x[i+1:i+h+1].mean()) if i+h<len(mdates) and j+h<len(vdates) and np.array_equal(mv[i+1:i+h+1],vvdates[j+1:j+h+1]) else np.nan for h in e.HS}
        rows.append(r)
    return rows
dates=sorted(pred.date.unique());chosen=set(rng.choice(dates,size=min(200,len(dates)),replace=False))
for _,g in pred.groupby(pred.date.str[:4]):chosen.update((g.date.min(),g.date.max()))
records=[]
for ds in sorted(chosen):
    ds=str(ds);t=pd.Timestamp(ds);j=int(np.searchsorted(vd,t));assert vd[j]==t
    prefix,prefix_onsets=rebuild_features(vd[:j+1],vv[:j+1]);changed=vv.copy()
    changed[j+1:]=np.maximum(.01,changed[j+1:]*rng.uniform(.5,1.5,size=len(changed)-j-1))
    changedf,changed_onsets=rebuild_features(vd,changed)
    row_cache={}
    for p in pred[pred.date==ds].itertuples():
        md=json.loads((root/'monthly_models'/(p.model_id+'.json')).read_text());model=md['model']
        d,v=panel[p.maturity];idx=int(np.searchsorted(d,t));assert d[idx]==t
        if p.maturity not in row_cache:
            original_rows=rebuilt_rows(d[:idx+1],v[:idx+1],vd[:j+1],vv[:j+1]);
            alterv=v.copy();alterv[idx+1:]=np.maximum(.01,alterv[idx+1:]*rng.uniform(.5,1.5,size=len(alterv)-idx-1))
            changed_rows=rebuilt_rows(d[:idx+1],alterv[:idx+1],vd[:j+1],changed[:j+1]);
            row_cache[p.maturity]=(original_rows,changed_rows)
        original_rows,changed_rows=row_cache[p.maturity];row=original_rows[idx];alterrow=changed_rows[idx]
        cutoff=pd.Timestamp(p.cutoff)
        fitted=rebuilt_model(original_rows,p.h,cutoff)
        fitted_changed=rebuilt_model(changed_rows,p.h,cutoff)
        model_match=bool(fitted is not None and fitted_changed is not None and fitted[0]==model['tau'] and fitted_changed[0]==model['tau'] and np.allclose(fitted[1],model['har'],rtol=0,atol=1e-12) and np.allclose(fitted[2],model['harx'],rtol=0,atol=1e-12) and np.allclose(fitted_changed[1],model['har'],rtol=0,atol=1e-12) and np.allclose(fitted_changed[2],model['harx'],rtol=0,atol=1e-12))
        def reforecast(r,fit):
            c=r['core'];z=np.r_[r['har'],c[0],c[1],c[2],c[0]*np.exp(-c[2]/fit[0]),c[3]]
            return float(r['har']@fit[1]),float(z@fit[2])
        a,b=reforecast(row,fitted);c,d2=reforecast(alterrow,fitted_changed)
        rec=ledger[(ledger.maturity==p.maturity)&(ledger.h==p.h)&(ledger.date==ds)].iloc[0]
        boundary=bool(pd.Timestamp(rec.refit_cutoff)<t and pd.Timestamp(rec.latest_train_label)<=pd.Timestamp(rec.refit_cutoff) and pd.Timestamp(rec.feature_max_date)<=t and (not rec.running_max_source_date or pd.Timestamp(rec.running_max_source_date)<=t) and (not rec.baseline_end or pd.Timestamp(rec.baseline_end)<t))
        exact=bool(np.isclose(a,p.har,rtol=0,atol=1e-12) and np.isclose(b,p.harx,rtol=0,atol=1e-12) and np.isclose(c,p.har,rtol=0,atol=1e-12) and np.isclose(d2,p.harx,rtol=0,atol=1e-12) and np.array_equal(row['core'],alterrow['core']) and np.array_equal(row['core'],prefix[-1]) and np.array_equal(row['core'],changedf[j]))
        records.append(dict(date=ds,maturity=p.maturity,h=p.h,origin_index=j,model_id=p.model_id,model_sha256=e.sha(root/'monthly_models'/(p.model_id+'.json')),prefix_har=a,prefix_harx=b,perturbed_har=c,perturbed_harx=d2,saved_har=p.har,saved_harx=p.harx,exact=exact,model_match=model_match,boundary=boundary,seed=e.seed('Extension','E3','truncation_audit')))
e.savecsv(root/'leakage_audit_v2.csv',records)
summary=dict(selected_dates=len(chosen),comparison_rows=len(records),exact=sum(r['exact'] for r in records),models_matched=sum(r['model_match'] for r in records),boundaries=sum(r['boundary'] for r in records),status='passed' if all(r['exact'] and r['model_match'] and r['boundary'] for r in records) else 'failed')
e.atom(root/'leakage_audit_v2_summary.json',summary)
print(json.dumps(summary))
if summary['status']!='passed':sys.exit(1)
