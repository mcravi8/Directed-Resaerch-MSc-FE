"""S02a implementation; immutable outputs, frozen before outcome access."""
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'stages/S02a_pooled'
sys.path.insert(0,str(ROOT/'stages/S01_protocol/attempt_02'))
import protocol_core as core
MATS=['VIX9D','VIX30D','VIX3M','VIX6M']
HORIZONS=[120,61,250]
STOP=float('inf')  # S02a continuation authorized without the original time cutoff.

def guard():
    if time.time()>=STOP: raise TimeoutError('analysis reserve reached')

def write(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): raise FileExistsError(path)
    raw=json.dumps(obj,indent=2,allow_nan=False)+'\n'
    tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(raw); tmp.rename(path)

def inputs():
    ts=pd.read_csv(ROOT/'omori_pipeline/data/vix_term_structure.csv',parse_dates=['Date']).set_index('Date').sort_index()
    assert ts.index.is_unique and ts.index.is_monotonic_increasing
    ev=pd.read_csv(ROOT/'omori_pipeline/data/shock_events.csv',parse_dates=['onset_date','peak_date']).fillna({'label':''})
    a=ev.loc[ev.label=='Lehman'].iloc[0]; b=ev.loc[ev.label=='GFC Peak'].iloc[0]
    ev=pd.concat([ev.loc[~ev.label.isin(['Lehman','GFC Peak'])],pd.DataFrame([dict(onset_date=a.onset_date,peak_date=b.peak_date,label='GFC (merged Lehman + GFC Peak)')])],ignore_index=True).sort_values('onset_date').reset_index(drop=True)
    assert len(ev)==33
    ev['event_id']=[d.strftime('%Y-%m-%d')+(' '+l if l else '') for d,l in zip(ev.peak_date,ev.label)]
    return ts,ev

def panels(ts,ev,mat,h,anchor_kind='argmax'):
    s=ts[mat]; dates=s.index; onsets=list(ev.onset_date); rows=[]; trajectories={}; intervals={}
    for _,e in ev.iterrows():
        eid=e.event_id; reason=None; on=int(dates.searchsorted(e.onset_date))
        anchor=int(dates.searchsorted(e.peak_date)) if anchor_kind=='argmax' else on+5
        pre=s.iloc[:on].dropna().iloc[-250:]
        later=[d for d in onsets if d>e.onset_date]
        end=anchor+h
        if on>=len(s) or anchor>=len(s) or end>=len(s): reason='end_of_data'
        elif dates[on]!=e.onset_date or (anchor_kind=='argmax' and dates[anchor]!=e.peak_date): reason='anchor_or_onset_not_observed_date'
        elif len(pre)<250: reason='prehistory'
        elif later and dates[end]>=min(later): reason='next_onset'
        else:
            try:
                z=core.normalize(pre.values,float(s.iloc[anchor]),s.iloc[anchor:end+1].values)
                trajectories[eid]=z
                intervals[eid]=(pre.index[0].value,dates[end].value)
            except ValueError as ex: reason=str(ex)
        rows.append(dict(event_id=eid,maturity=mat,horizon=h,anchor=anchor_kind,eligible=reason is None,reason=reason,pre_count=len(pre),anchor_index=anchor,onset_index=on))
    return trajectories,intervals,rows

def fit_unit(panel,ids,intervals,key,draws,mode='event',paired_key=None):
    directory=OUT/'units'/key
    directory.mkdir(parents=True,exist_ok=True)
    z=np.array([panel[i] for i in ids]); h=z.shape[1]-1
    groups=core.overlap_groups([intervals[i] for i in ids]); ug=np.unique(groups)
    metadata=dict(key=key,event_ids=ids,event_count=len(ids),group_count=len(ug),horizon=h,mode=mode,draws=draws)
    if not (directory/'panel.json').exists(): write(directory/'panel.json',dict(**metadata,trajectories=z.tolist(),groups=groups.tolist()))
    for rep in range(-1,draws):
        rng=np.random.default_rng(core.seed(paired_key or key,mode,rep,'resample'))
        if rep==-1: ix=np.arange(len(ids))
        elif mode=='event': ix=rng.integers(0,len(ids),len(ids))
        else:
            selected=rng.choice(ug,len(ug),replace=True)
            ix=np.concatenate([np.flatnonzero(groups==g) for g in selected])
        y=z[ix].mean(axis=0)
        for zero in (False,True):
            for start in core.STARTS:
                p=directory/f'r{rep}_b{int(zero)}_s{start}.json'
                if p.exists(): continue
                guard(); begin=time.perf_counter(); fitted={}
                for family in core.BOUNDS:
                    guard()
                    f=core.fit(family,np.arange(start,h+1),y[start:],['S02a',key,mode,rep,start,zero],zero)
                    best=f['best']
                    f['score']=core.score(best['sse'],f['n'],f['k']) if best else None
                    fitted[family]=f
                write(p,dict(key=key,replicate=rep,zero_baseline=zero,start=start,indices=ix.tolist(),models=fitted,wall_seconds=time.perf_counter()-begin))
    return metadata

def stationary(process):
    rng=np.random.default_rng(core.seed('timing',process)); n=15000
    def ou(tau,sd):
        x=np.empty(n); x[0]=rng.normal(0,sd); phi=math.exp(-1/tau)
        innovations=rng.normal(0,sd*math.sqrt(-math.expm1(-2/tau)),n-1)
        for i in range(1,n): x[i]=phi*x[i-1]+innovations[i-1]
        return x
    if process=='OU': x=ou(90,.4)
    elif process=='2OU': x=ou(5,.4*math.sqrt(.25))+ou(90,.4*math.sqrt(.75))
    else:
        r=core.arfima_cov(n,.4); c=np.r_[r,r[-2:0:-1]]
        lam=np.fft.fft(c).real
        if lam.min() < -1e-10: raise ValueError('stationary covariance embedding has negative eigenvalue')
        x=np.fft.ifft(np.sqrt(np.maximum(lam,0))*np.fft.fft(rng.normal(size=len(c)))).real[:n]
    return x[5000:]+math.log(20)

def timing_dates(ts):
    saved=ts.index[ts.VIX30D.notna()]
    offsets=(saved-saved[0]).days.to_numpy()
    period=int(offsets[-1])+1
    while period%7: period+=1
    days=[]; cycle=0
    while len(days)<10000:
        days.extend((saved[0]+pd.to_timedelta(offsets+cycle*period,unit='D')).tolist()); cycle+=1
    return pd.DatetimeIndex(days[:10000])

def selection(key):
    folder=OUT/'units'/key
    point=json.loads((folder/'r-1_b0_s1.json').read_text())['models']
    scores={f:x['score'] for f,x in point.items()}
    if any(v is None for v in scores.values()): return dict(classification='undetermined',reason='fit_failure')
    winner=min(scores,key=scores.get); margin=min(v-scores[winner] for f,v in scores.items() if f!=winner)
    wins=0
    for i in range(199):
        fits=json.loads((folder/f'r{i}_b0_s1.json').read_text())['models']
        if all(fits[f]['score'] is not None for f in fits) and min(fits,key=lambda f:fits[f]['score'])==winner: wins+=1
    reason=None; pars=point[winner]['best']['parameters']
    if margin<2 or wins/199<.9: reason='insufficient_margin_or_stability'
    if winner=='E2' and (pars[2]<.05 or pars[2]>.95 or pars[4]/pars[3]<2): reason='collapsed_E2'
    if winner in ('PL','SE'):
        t=np.arange(1,121); residual_sd=math.sqrt(point[winner]['best']['sse']/120)
        separation=float(np.max(np.abs(core.curve(winner,t,pars)-core.curve('E1',t,point['E1']['best']['parameters']))))
        if separation<.05*residual_sd: reason='EXP_like'
    return dict(classification=winner if reason is None else 'undetermined',reason=reason,point_winner=winner,margin=margin,bootstrap_wins=wins,bootstrap_draws=199)

def pilot(process):
    guard(); started=time.perf_counter(); ts,_=inputs(); x=stationary(process)
    dates=timing_dates(ts); values=np.exp(x); events=core.detect(dates.values,values)
    ev=pd.DataFrame([dict(onset_date=dates[o],peak_date=dates[p],event_id=str(i)) for i,(o,p) in enumerate(events)])
    sim=pd.DataFrame({'VIX30D':values},index=dates); records=[]; counts=[]; scores={}
    for h in HORIZONS:
        a,ia,ra=panels(sim,ev,'VIX30D',h,'argmax'); b,ib,rb=panels(sim,ev,'VIX30D',h,'onset5')
        ids=sorted(set(a)&set(b)); records.extend(ra+rb); counts.append(dict(horizon=h,argmax=len(a),onset5=len(b),common=len(ids)))
        for anchor,panel,interval in [('argmax',a,ia),('onset5',b,ib)]:
            key=f'timing_{process}_{h}_{anchor}'
            if len(ids)<10:
                scores[key]=dict(classification='undetermined',reason='fewer_than_10_common_eligible_events')
            else:
                fit_unit(panel,ids,interval,key,199,paired_key=f'timing_{process}_{h}')
                if h==120: scores[key]=selection(key)
    write(OUT/'pilots'/f'{process}_checkpoint.json',dict(process=process,synthetic=True,seed=core.seed('timing',process),detected=len(events),counts=counts,scores=scores,cohorts=records))
    elapsed=time.perf_counter()-started
    write(OUT/'pilots'/f'{process}_timing.json',dict(process=process,total_wall_seconds=elapsed,completed=True,seed=core.seed('timing',process),detected_events=len(events),synthetic=True))
    print(process,elapsed,flush=True)

def empirical():
    ts,ev=inputs(); allrows=[]; plans=[]
    for h in HORIZONS:
        pp={m:panels(ts,ev,m,h) for m in MATS}
        for m in MATS: allrows.extend(pp[m][2])
        common=sorted(set.intersection(*(set(pp[m][0]) for m in MATS)))
        for m in MATS:
            panel,interval,_=pp[m]
            for cohort,ids in [('full',sorted(panel)),('common',common)]:
                if not ids: plans.append(dict(maturity=m,horizon=h,cohort=cohort,status='coverage_failure')); continue
                for mode in ('event','group'):
                    key=f'empirical_{h}_{m}_{cohort}_{mode}'
                    plans.append(fit_unit(panel,ids,interval,key,499,mode,paired_key=f'paired_{h}_{mode}' if cohort=='common' else None))
    pd.DataFrame(allrows).to_csv(OUT/'cohort_manifest.csv',index=False)
    write(OUT/'empirical_completed.json',plans)

if __name__=='__main__':
    if sys.argv[1]=='pilot': pilot(sys.argv[2])
    elif sys.argv[1]=='empirical': empirical()
