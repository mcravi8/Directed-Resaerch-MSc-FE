"""Frozen S02b v2 calibration. Run once after hashing code and method."""
import csv, hashlib, json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'stages/S02b_recovery'
sys.path.insert(0,str(ROOT/'stages/S01_protocol/attempt_02'))
import protocol_core as core
LAGS=(1,22,63)
ACF_TOL=np.array([.015,.07,.10])
PROCESSES=('OU','2OU','ARFIMA')
K=11

def dump(name,obj):
    p=OUT/name
    if p.exists(): raise FileExistsError(p)
    p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def moments(x):
    x=np.asarray(x,float)
    return np.array([x.mean(),x.std()]+[np.corrcoef(x[:-k],x[k:])[0,1] for k in LAGS])

def theory(process,q):
    if process=='OU': return np.exp(-np.array(LAGS)/q[0])
    if process=='2OU':
        fast,slow,w=q
        return w*np.exp(-np.array(LAGS)/fast)+(1-w)*np.exp(-np.array(LAGS)/slow)
    d=q[0]
    return np.array([np.prod([(j-1+d)/(j-d) for j in range(1,k+1)]) for k in LAGS])

def fit_theory(process,target):
    if process=='OU':
        starts=[[v] for v in (2,10,30,90,300,1000)]; lo=[1];hi=[2000]
    elif process=='2OU':
        starts=[[a,b,w] for a,b in ((2,30),(5,90),(15,300),(30,1000),(100,1500)) for w in (.2,.5,.8)]
        lo=[1,1,.01];hi=[2000,2000,.99]
    else:
        starts=[[v] for v in (.05,.2,.35,.45,.49)];lo=[.01];hi=[.499]
    best=None
    for st in starts:
        sol=least_squares(lambda q:(theory(process,q)-target)/ACF_TOL,st,bounds=(lo,hi),xtol=1e-12,ftol=1e-12,gtol=1e-12,max_nfev=3000)
        q=sol.x
        if process=='2OU' and q[0]>q[1]: q=np.array([q[1],q[0],1-q[2]])
        loss=float(np.sum(((theory(process,q)-target)/ACF_TOL)**2))
        if best is None or loss<best['loss']:best=dict(parameters=q.tolist(),loss=loss,optimizer_success=bool(sol.success),nfev=int(sol.nfev))
    return best

def generate(process,q,mean,sd,n,seed):
    rng=np.random.default_rng(seed)
    def ou(tau,s):
        z=np.empty(n);z[0]=rng.normal(0,s)
        phi=math.exp(-1/tau)
        e=rng.normal(0,s*math.sqrt(-math.expm1(-2/tau)),n-1)
        for j in range(1,n):z[j]=phi*z[j-1]+e[j-1]
        return z
    if process=='OU':z=ou(q[0],sd)
    elif process=='2OU':z=ou(q[0],sd*math.sqrt(q[2]))+ou(q[1],sd*math.sqrt(1-q[2]))
    else:
        r=core.arfima_cov(n,q[0],sd)
        circ=np.r_[r,r[-2:0:-1]]
        lam=np.fft.fft(circ).real
        if lam.min() < -1e-10:raise ValueError('negative circulant eigenvalue')
        z=np.fft.ifft(np.sqrt(np.maximum(lam,0))*np.fft.fft(rng.normal(size=len(circ)))).real[:n]
    return z+mean

def cohort(dates,values,events):
    eligible=[];reasons=[]
    for j,(on,peak) in enumerate(events):
        next_on=events[j+1][0] if j+1<len(events) else len(values)
        for anchor_name,anchor in [('argmax',peak),('onset5',on+5)]:
            reason=None
            if on<250: reason='prehistory'
            elif anchor+120>=len(values):reason='end_of_data'
            elif anchor+120>=next_on:reason='next_onset'
            else:
                pre=values[on-250:on]
                try: core.normalize(pre,values[anchor],values[anchor:anchor+121])
                except ValueError as e:reason=str(e)
            reasons.append(dict(event_index=j,anchor=anchor_name,eligible=reason is None,reason=reason))
            if reason is None:eligible.append((j,anchor_name))
    common=set(j for j,a in eligible if a=='argmax')&set(j for j,a in eligible if a=='onset5')
    return len(common),reasons

def main():
    s=pd.read_csv(ROOT/'omori_pipeline/data/vix_term_structure.csv',parse_dates=['Date']).dropna(subset=['VIX30D'])
    assert len(s)==9234 and s.VIX30D.gt(0).all() and s.Date.is_monotonic_increasing
    dates=s.Date.to_numpy();v=s.VIX30D.to_numpy();x=np.log(v);obs=moments(x)
    empirical_events=core.detect(dates,v)
    empirical_common,empirical_reasons=cohort(dates,v,empirical_events)
    rows=[];params={}
    for process in PROCESSES:
        fit=fit_theory(process,obs[2:]);q=fit['parameters']
        predicted=theory(process,q)
        theoretical_residual=predicted-obs[2:]
        vals=[];counts=[];commons=[]
        for i in range(K):
            seed=core.seed('S02b','calibration',process,i)
            path=generate(process,q,obs[0],obs[1],len(x),seed)
            generated=moments(path);events=core.detect(dates,np.exp(path))
            common,reasons=cohort(dates,np.exp(path),events)
            vals.append(generated);counts.append(len(events));commons.append(common)
            rows.append(dict(process=process,path_id=i,seed=seed,mean=generated[0],sd=generated[1],acf1=generated[2],acf22=generated[3],acf63=generated[4],detected=len(events),eligible_common=common,anchor_exclusions=json.dumps(reasons,separators=(',',':'))))
        med=np.median(vals,axis=0);mc=int(np.median(counts));me=int(np.median(commons))
        gates={'theory_acf':bool(np.all(np.abs(theoretical_residual)<=ACF_TOL)),
               'median_mean_sd':bool(np.all(np.abs(med[:2]-obs[:2])<=[.03,.03])),
               'median_acf':bool(np.all(np.abs(med[2:]-obs[2:])<=ACF_TOL)),
               'detected_20_50':bool(20<=mc<=50),'eligible_at_least_12':bool(me>=12)}
        params[process]=dict(status='CALIBRATABLE' if all(gates.values()) else 'NOT CALIBRATABLE',parameters=q,
            objective=fit['loss'],optimizer_success=fit['optimizer_success'],theory_acf=predicted.tolist(),
            theory_acf_residual=theoretical_residual.tolist(),median_generated_moments=med.tolist(),
            median_detected=mc,median_eligible_common=me,gates=gates,
            path_seeds=[core.seed('S02b','calibration',process,i) for i in range(K)])
    with (OUT/'calibration_diagnostics.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    dump('calibration_parameters.json',dict(observed_moments=obs.tolist(),acf_tolerances=ACF_TOL.tolist(),
        mean_sd_tolerances=[.03,.03],empirical_raw_detected=len(empirical_events),empirical_common_eligible=empirical_common,
        empirical_named_merged_catalog=33,empirical_anchor_exclusions=empirical_reasons,
        processes=params,seed_base=core.SEED,ensemble_size=K,date_grid='exact observed VIX30D dates',
        method='calibration_method_v1.md'))
    with (OUT/'detector_parity.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['series','path_id','raw_detected','common_eligible','detector','merge'])
        w.writeheader();w.writerow(dict(series='empirical',path_id='observed',raw_detected=len(empirical_events),common_eligible=empirical_common,detector='protocol_core.detect',merge='none'))
        for r in rows:w.writerow(dict(series=r['process'],path_id=r['path_id'],raw_detected=r['detected'],common_eligible=r['eligible_common'],detector='protocol_core.detect',merge='none'))
    print(json.dumps({p:{k:params[p][k] for k in ('status','median_detected','median_eligible_common','gates')} for p in PROCESSES}))

if __name__=='__main__': main()
