"""Frozen E3 runner. See E3_spec_v2.md; all paths relative to project root."""
import csv, hashlib, json, math, os, sys, time
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

ROOT=Path('extension/E03_forecast'); RAW=ROOT/'raw'; MODELS=ROOT/'monthly_models'
SPEC=Path('extension/E01_protocol/E3_spec_v2.md')
NAMES=('VIX30D','VIX9D','VIX3M','VIX6M'); HS=(5,10,20); TAUS=(5,10,20,40,80,120)
URLS={n:'https://cdn.cboe.com/api/global/us_indices/daily_prices/'+v for n,v in {'VIX30D':'VIX_History.csv','VIX9D':'VIX9D_History.csv','VIX3M':'VIX3M_History.csv','VIX6M':'VIX6M_History.csv'}.items()}
SEED=5902026
def seed(*key):
    raw=json.dumps([SEED,*key],separators=(',',':'),ensure_ascii=True).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8],'big')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atom(p,obj):
    q=Path(str(p)+'.tmp'); q.write_text(json.dumps(obj,indent=2,default=str)+'\n'); os.replace(q,p)
def savecsv(p,rows): pd.DataFrame(rows).to_csv(p,index=False)
def parse(name):
    p=RAW/(name+'.csv'); df=pd.read_csv(p,skip_blank_lines=True)
    df.columns=[str(c).strip().upper().replace(' ','') for c in df.columns]
    datecol=next((c for c in df if c in ('DATE','TRADEDATE')),df.columns[0])
    valcol=next((c for c in df if c in ('CLOSE','CLOSINGVALUE','VIX','VIX9D','VIX3M','VIX6M')),None)
    if valcol is None: raise ValueError(f'{name}: close column not found: {list(df)}')
    dates=pd.to_datetime(df[datecol],errors='coerce'); values=pd.to_numeric(df[valcol],errors='coerce')
    bad=[]; seen=set(); clean=[]
    for i,(d,v) in enumerate(zip(dates,values)):
        reason=None
        if pd.isna(d): reason='invalid_date'
        elif d in seen: reason='duplicate_date'
        elif not np.isfinite(v) or v<=0: reason='nonpositive_or_missing_close'
        if reason: bad.append(dict(maturity=name,row=i+2,date=str(d),reason=reason))
        else: clean.append((d,float(v))); seen.add(d)
    clean.sort(); dates=pd.DatetimeIndex([x[0] for x in clean]); v=np.array([x[1] for x in clean])
    if len(dates)==0: raise ValueError(name+' no valid rows')
    return dates,v,bad,len(df)
def onset_features(dates,v):
    n=len(v); onset=np.full(n,-1,int); age=np.full(n,-1,int); base=np.full(n,np.nan)
    running=np.full(n,np.nan); maxidx=np.full(n,-1,int); last=-1; mx=-np.inf; mi=-1; events=[]
    for i in range(n):
        trigger=(i>=1 and v[i]-v[i-1]>=7) or (i>=5 and v[i]-v[i-5]>=10)
        if trigger and (last<0 or (dates[i]-dates[last]).days>30):
            last=i; mx=v[i]; mi=i
            events.append(dict(onset_id=len(events),onset_date=str(dates[i].date()),index=i,baseline_start=str(dates[i-20].date()) if i>=20 else '',baseline_end=str(dates[i-1].date()) if i>=20 else '',baseline=float(np.mean(v[i-20:i])) if i>=20 else None))
        if last>=0:
            if v[i]>mx: mx=v[i]; mi=i
            onset[i]=last; age[i]=i-last; running[i]=mx; maxidx[i]=mi
            if last>=20: base[i]=np.mean(v[last-20:last])
    active=(age>=0)&(age<120)
    delta=np.full(n,np.nan); delta[5:]=np.log(v[5:])-np.log(v[:-5])
    core=np.zeros((n,4)); core[:,0]=active.astype(float)
    ok=active&np.isfinite(base)&np.isfinite(delta)
    core[ok,1]=np.log(running[ok]/base[ok]); core[ok,2]=age[ok];core[ok,3]=delta[ok]
    # Active but missing prerequisite is not silently treated as inactive.
    eligible=~active|ok
    return dict(onset=onset,age=age,base=base,running=running,maxidx=maxidx,active=active,core=core,eligible=eligible,events=events)
def read_panel():
    parsed={}; bad=[]; cov=[]
    for n in NAMES:
        p=RAW/(n+'.csv')
        if not p.exists(): cov.append(dict(maturity=n,status='unavailable',reason='official_source_missing'));continue
        try: d,v,b,raw_n=parse(n)
        except Exception as e: cov.append(dict(maturity=n,status='unavailable',reason=str(e)));continue
        parsed[n]=(d,v);bad.extend(b);cov.append(dict(maturity=n,status='available',raw_rows=raw_n,valid_rows=len(d),first=str(d[0].date()),last=str(d[-1].date()),excluded=len(b)))
    savecsv(ROOT/'source_exclusions.csv',bad);savecsv(ROOT/'coverage.csv',cov)
    if 'VIX30D' not in parsed: raise RuntimeError('primary official VIX history unavailable')
    cutoff=parsed['VIX30D'][0][-1]
    for n,(d,v) in list(parsed.items()):
        keep=d<=cutoff;parsed[n]=(d[keep],v[keep])
    return parsed,cutoff
def make_rows(n,d,v,vixdates,feat):
    lookup={a:i for i,a in enumerate(vixdates)}; x=np.log(v); rows=[]
    for i,dt in enumerate(d):
        j=lookup.get(dt,-1); reason=''
        if j<0: reason='no_VIX30D_feature_date'
        elif i<21: reason='HAR_22_lags_unavailable'
        elif not feat['eligible'][j]: reason='shock_feature_history_unavailable'
        elif any(d[i-k] not in lookup for k in range(22)): reason='lag_window_missing_exchange_session'
        elif any(lookup[d[i-k]]!=j-k for k in range(22)): reason='lag_window_gap'
        if not reason:
            z=np.array([1,x[i],np.mean(x[i-4:i+1]),np.mean(x[i-21:i+1])]);core=feat['core'][j]
            target={h:float(np.mean(x[i+1:i+h+1])) if i+h<len(d) and all(d[i+k] in lookup and lookup[d[i+k]]==j+k for k in range(1,h+1)) else np.nan for h in HS}
            rows.append(dict(i=i,j=j,date=dt,har=z,core=core,target=target,onset=int(feat['onset'][j]),active=bool(feat['active'][j])))
        else: rows.append(dict(i=i,j=j,date=dt,reason=reason))
    return rows
def design(rows,tau):
    a=np.array([r['har'] for r in rows]); c=np.array([r['core'] for r in rows]);
    return np.column_stack([a,c[:,0],c[:,1],c[:,2],c[:,0]*np.exp(-c[:,2]/tau),c[:,3]])
def ols(X,y):
    if len(y)<X.shape[1]+1: return None
    u,s,vt=np.linalg.svd(X,full_matrices=False)
    tol=max(X.shape)*np.finfo(float).eps*s[0]
    if np.sum(s>tol)!=X.shape[1]: return None
    return vt.T@((u.T@y)/s)
def month_model(rows,h,C):
    mature=[r for r in rows if 'har' in r and np.isfinite(r['target'][h]) and r['date']<=C and rows[r['i']+h]['date']<=C]
    years=[C.year-3,C.year-2,C.year-1]
    # Last three complete calendar years ending before C; if C is Dec 31, that year is complete.
    if C.month==12 and C.day==31: years=[C.year-2,C.year-1,C.year]
    folds=[]; losses={tau:[] for tau in TAUS}
    for yr in years:
        jan=pd.Timestamp(year=yr,month=1,day=1);dec=pd.Timestamp(year=yr,month=12,day=31)
        tr=[r for r in mature if rows[r['i']+h]['date']<jan]
        va=[r for r in mature if jan<=r['date']<=dec and rows[r['i']+h]['date']<=dec]
        fold=dict(year=yr,train_n=len(tr),validation_n=len(va),train_max_label=str(max((rows[r['i']+h]['date'] for r in tr),default='')),validation_max_label=str(max((rows[r['i']+h]['date'] for r in va),default='')))
        if not tr or not va: return None,dict(reason='inner_fold_history_missing',folds=folds+[fold])
        yy=np.array([r['target'][h] for r in tr]);vy=np.array([r['target'][h] for r in va])
        for tau in TAUS:
            beta=ols(design(tr,tau),yy)
            if beta is None: return None,dict(reason='inner_fold_rank',folds=folds+[fold])
            pred=design(va,tau)@beta;losses[tau].extend((vy-pred)**2)
            fold['tau_'+str(tau)+'_mse']=float(np.mean((vy-pred)**2))
            fold['tau_'+str(tau)+'_coef_sha']=hashlib.sha256(beta.tobytes()).hexdigest()
        folds.append(fold)
    tau=min(TAUS,key=lambda z:(float(np.mean(losses[z])),z))
    yy=np.array([r['target'][h] for r in mature]);bh=ols(np.array([r['har'] for r in mature]),yy);bx=ols(design(mature,tau),yy)
    if bh is None or bx is None:return None,dict(reason='final_rank_or_n',folds=folds)
    model=dict(tau=tau,har=bh.tolist(),harx=bx.tolist(),train_n=len(mature),train_max_label=str(max(rows[r['i']+h]['date'] for r in mature)),folds=folds,tau_mse={str(z):float(np.mean(losses[z])) for z in TAUS})
    return model,None
def crisis_units(vixdates,feat):
    units=[]
    for e in feat['events']:
        a=e['index'];b=min(a+119,len(vixdates)-1)
        if units and a<=units[-1]['end_idx']+1:
            units[-1]['end_idx']=max(b,units[-1]['end_idx']);units[-1]['onset_ids'].append(e['onset_id'])
        else:units.append(dict(crisis_id=len(units),start_idx=a,end_idx=b,onset_ids=[e['onset_id']]))
    for u in units:u.update(start_date=str(vixdates[u['start_idx']].date()),end_date=str(vixdates[u['end_idx']].date()))
    ids=np.full(len(vixdates),-1,int)
    for u in units:ids[u['start_idx']:u['end_idx']+1]=u['crisis_id']
    return units,ids
def tests(pred,units,h,n,subset):
    p=pred.sort_values('date').reset_index(drop=True);T=len(p);active=p.active.to_numpy(dtype=bool)
    d=p.d.to_numpy(dtype=float);mask=active if subset=='active' else np.ones(T,bool);A=int(mask.sum())
    K=int(p.loc[mask & (p.crisis_id>=0),'crisis_id'].nunique());episodes=int(p.loc[active,'onset'].nunique())
    out=dict(maturity=n,h=h,subset=subset,T=T,A=A,paired_n=A if subset=='active' else T,K=K,episodes=episodes,first=str(p.date.iloc[0]),last=str(p.date.iloc[-1]))
    if A==0:out.update(reason='no_paired_origins',p_raw=1);return out,[],[]
    a=p.loc[mask];m=float(a.d.mean());har=float(a.har_loss.mean());hx=float(a.harx_loss.mean());out.update(har_mse=har,harx_mse=hx,mean_d=m,relative_gain=m/har if har>0 else None)
    g=np.where(mask,d-m,0.0)*(T/A if subset=='active' else 1.0)
    V=float(g@g/T+sum(2*(1-l/h)*(g[l:]@g[:-l]/T) for l in range(1,h)))
    Farg=(T+1-2*h+h*(h-1)/T)/T;F=math.sqrt(Farg) if Farg>0 else np.nan
    out.update(V=V,F=F)
    if T<max(h+2,3) or (subset=='active' and (A<max(3,h) or episodes<2)) or V<=0 or not np.isfinite(V*F):
        out.update(reason='DM_not_estimable',p_raw=1,dm_stat=None)
    else:
        stat=F*m/math.sqrt(V/T);out.update(dm_stat=stat,p_raw=float(2*student_t.sf(abs(stat),T-1)),reason='')
    crisis={int(k):z.d.to_numpy(float) for k,z in a[a.crisis_id>=0].groupby('crisis_id')}
    # Missing scored dates split inactive runs, including gaps in the master session index.
    blocks=[];run=[];lastj=None
    for r in a.itertuples():
        if r.crisis_id>=0:
            if run:blocks.extend([np.array(run[i:i+h]) for i in range(0,len(run),h)]);run=[]
            lastj=None;continue
        if lastj is not None and r.j!=lastj+1:
            blocks.extend([np.array(run[i:i+h]) for i in range(0,len(run),h)]);run=[]
        run.append(r.d);lastj=r.j
    if run:blocks.extend([np.array(run[i:i+h]) for i in range(0,len(run),h)])
    out['inactive_blocks']=len(blocks)
    loo=[]
    for k in sorted(crisis):
        z=a[a.crisis_id!=k];u=units[k];mean=float(z.d.mean()) if len(z) else None;den=float(z.har_loss.mean()) if len(z) else None
        loo.append(dict(maturity=n,h=h,subset=subset,omitted_crisis=k,start=u['start_date'],end=u['end_date'],onset_ids=';'.join(map(str,u['onset_ids'])),remaining_n=len(z),remaining_K=K-1,mean_d=mean,relative_gain=mean/den if mean is not None and den>0 else None,reason='' if mean is not None and den>0 else 'empty_or_zero_HAR_MSE'))
    if K<10:out.update(bootstrap_reason='K_below_10',bootstrap_lo=None,bootstrap_hi=None)
    elif subset=='all' and len(blocks)==0 and np.any(a.crisis_id<0):out.update(bootstrap_reason='inactive_stratum_missing')
    else:
        rng=np.random.Generator(np.random.PCG64(seed('Extension','E3','crisis_bootstrap',n,h,subset)))
        c=list(crisis.values());means=[]
        for rep in range(1999):
            chosen=[c[i] for i in rng.integers(0,len(c),len(c))]
            if subset=='all' and blocks:chosen += [blocks[i] for i in rng.integers(0,len(blocks),len(blocks))]
            means.append(float(np.sum([z.sum() for z in chosen])/np.sum([len(z) for z in chosen])))
        lo,hi=np.percentile(means,[2.5,97.5],method='linear');out.update(bootstrap_reason='',bootstrap_lo=float(lo),bootstrap_hi=float(hi),bootstrap_nonpositive=float(np.mean(np.array(means)<=0)),bootstrap_seed=seed('Extension','E3','crisis_bootstrap',n,h,subset))
    out['positive_survives_checks']=bool(K>=10 and out.get('bootstrap_lo') is not None and out['bootstrap_lo']>0 and len(loo)==K and all(z['mean_d'] is not None and z['mean_d']>0 for z in loo))
    return out,loo,[dict(maturity=n,h=h,subset=subset,replicate=i,mean_d=x) for i,x in enumerate(means)] if K>=10 and 'means' in locals() else []
def main():
    start=time.monotonic(); expected='d8d1f831698006ad01f09a926630d323008c1f5af529952f0d98930a76e6e687'
    assert sha(SPEC)==expected
    freeze=json.loads((ROOT/'input_manifest.json').read_text())
    assert sha(__file__)==freeze['code_sha256']
    for n,q in freeze['sources'].items():assert sha(RAW/(n+'.csv'))==q['sha256']
    panel,cutoff=read_panel();assert str(cutoff.date())==freeze['cutoff']
    vd,vv=panel['VIX30D'];feat=onset_features(vd,vv);savecsv(ROOT/'causal_onsets.csv',feat['events']);units,cids=crisis_units(vd,feat);savecsv(ROOT/'crisis_units.csv',units)
    feature_rows=[]
    for j,dt in enumerate(vd):
        o=feat['onset'][j];feature_rows.append(dict(date=str(dt.date()),onset_id=int(np.searchsorted([e['index'] for e in feat['events']],o)) if o>=0 else '',onset_index=int(o),active=int(feat['active'][j]),age=int(feat['age'][j]),baseline=feat['base'][j],running_max=feat['running'][j],max_source_date=str(vd[feat['maxidx'][j]].date()) if o>=0 else '',size=feat['core'][j,1],five_change=feat['core'][j,3],eligible=bool(feat['eligible'][j])))
    savecsv(ROOT/'features.csv',feature_rows)
    predictions=[];ledger=[];models=[];exclusions=[];completed=[]
    for n in NAMES:
        if n not in panel:continue
        d,v=panel[n];rows=make_rows(n,d,v,vd,feat)
        # Count the initial year only when its first observation is in the
        # exchange's opening January week; otherwise ten later full years.
        firstyear=d[0].year+10+(1 if d[0].month!=1 or d[0].day>7 else 0)
        for h in HS:
            for month in sorted(set(r['date'].to_period('M') for r in rows if r['date'].year>=firstyear)):
                C=month.to_timestamp()-pd.Timedelta(days=1)
                if n=='VIX30D' and C<pd.Timestamp('1999-12-31'):continue
                key=f'{n}_{h}_{month}';path=MODELS/(key+'.json')
                m=None;error=None
                if path.exists():
                    saved=json.loads(path.read_text());m=saved.get('model');error=saved.get('error')
                else:
                    m,error=month_model(rows,h,C)
                    atom(path,dict(unit_id=key,model=m,error=error,code_sha256=freeze['code_sha256'],input_manifest_sha256=sha(ROOT/'input_manifest.json'),cutoff=str(C.date())))
                models.append(dict(unit_id=key,maturity=n,h=h,month=str(month),tau=m['tau'] if m else '',train_n=m['train_n'] if m else '',train_max_label=m['train_max_label'] if m else '',error=error['reason'] if error else ''))
                for r in rows:
                    if r['date'].to_period('M')!=month:continue
                    reason=r.get('reason','')
                    if not reason and not np.isfinite(r['target'][h]):reason='incomplete_target_or_gap'
                    if not reason and m is None:reason='monthly_model_unestimable_'+error['reason']
                    j=r['j'];o=int(feat['onset'][j]) if j>=0 else -1
                    source=str(vd[j].date()) if j>=0 else ''
                    led=dict(maturity=n,h=h,date=str(r['date'].date()),reason=reason,feature_max_date=source,onset_date=str(vd[o].date()) if o>=0 else '',baseline_start=str(vd[o-20].date()) if o>=20 else '',baseline_end=str(vd[o-1].date()) if o>=20 else '',running_max_source_date=str(vd[feat['maxidx'][j]].date()) if o>=0 else '',age_source_date=source,five_change_end=source,refit_cutoff=str(C.date()),latest_train_label=m['train_max_label'] if m else '',model_id=key,target_end=str(d[r['i']+h].date()) if r['i']+h<len(d) else '')
                    ledger.append(led)
                    if reason:exclusions.append(dict(maturity=n,h=h,date=led['date'],reason=reason));continue
                    assert pd.Timestamp(led['feature_max_date'])<=r['date'] and pd.Timestamp(led['latest_train_label'])<=C
                    har=float(r['har']@np.array(m['har']));hx=float(design([r],m['tau'])[0]@np.array(m['harx']))
                    predictions.append(dict(maturity=n,h=h,date=led['date'],j=j,model_id=key,cutoff=str(C.date()),target_end=led['target_end'],y=r['target'][h],har=har,harx=hx,active=int(r['active']),onset=o,crisis_id=int(cids[j])))
                completed.append(key)
                atom(ROOT/'checkpoint.json',dict(spec_sha256=expected,code_sha256=freeze['code_sha256'],input_manifest_sha256=sha(ROOT/'input_manifest.json'),completed_units=completed,errors=[z for z in models if z['error']],elapsed_seconds=time.monotonic()-start))
    savecsv(ROOT/'monthly_models.csv',models);savecsv(ROOT/'origin_dependency_ledger.csv',ledger);savecsv(ROOT/'origin_exclusions.csv',exclusions);savecsv(ROOT/'predictions.csv',predictions)
    # Prefix + perturbed-suffix audit is performed before losses are calculated.
    pred=pd.DataFrame(predictions);audit=[]
    if len(pred):
        eligible_dates=sorted(pred.date.unique());rng=np.random.Generator(np.random.PCG64(seed('Extension','E3','truncation_audit')))
        chosen=set(rng.choice(eligible_dates,size=min(200,len(eligible_dates)),replace=False))
        for yr,g in pred.groupby(pred.date.str[:4]):chosen.update([g.date.min(),g.date.max()])
        bydate={r['date']:r for r in feature_rows};pmap=pred.groupby('date')
        for ds in sorted(chosen):
            j=int(np.searchsorted(vd,pd.Timestamp(ds)));prefix=onset_features(vd[:j+1],vv[:j+1]);alter=vv.copy();alter[j+1:]=np.maximum(.01,alter[j+1:]*rng.uniform(.5,1.5,size=len(alter)-j-1));pert=onset_features(vd,alter)
            equal=bool(np.array_equal(prefix['core'][-1],feat['core'][j]) and np.array_equal(pert['core'][j],feat['core'][j]) and prefix['onset'][-1]==feat['onset'][j] and pert['onset'][j]==feat['onset'][j])
            for row in pmap.get_group(ds).itertuples():
                saved=json.loads((MODELS/(row.model_id+'.json')).read_text())['model'];rnext=pred[(pred.date==ds)&(pred.maturity==row.maturity)&(pred.h==row.h)]
                same=bool(saved and len(rnext)==1 and rnext.iloc[0].har==row.har and rnext.iloc[0].harx==row.harx)
                audit.append(dict(date=ds,maturity=row.maturity,h=row.h,prefix_equal=equal,suffix_equal=equal,model_prediction_equal=same,seed=seed('Extension','E3','truncation_audit')))
        savecsv(ROOT/'leakage_audit.csv',audit)
        if not all(z['prefix_equal'] and z['suffix_equal'] and z['model_prediction_equal'] for z in audit):raise RuntimeError('leakage audit mismatch; scoring blocked')
    pred['har_loss']=(pred.y-pred.har)**2;pred['harx_loss']=(pred.y-pred.harx)**2;pred['d']=pred.har_loss-pred.harx
    pred.to_csv(ROOT/'losses.csv',index=False)
    testrows=[];loos=[];boots=[];slots=[('VIX30D',10,'active'),('VIX30D',10,'all')]+[(n,h,s) for n in NAMES for h in HS for s in ('active','all') if (n,h,s) not in [('VIX30D',10,'active'),('VIX30D',10,'all')]]
    for n,h,s in slots:
        sub=pred[(pred.maturity==n)&(pred.h==h)]
        if len(sub):t,l,b=tests(sub,units,h,n,s)
        else:t,l,b=dict(maturity=n,h=h,subset=s,reason='no_paired_forecasts',p_raw=1,K=0,T=0,A=0,paired_n=0),[],[]
        testrows.append(t);loos.extend(l);boots.extend(b)
    sec=testrows[1:];order=sorted(range(23),key=lambda i:(sec[i]['p_raw'],i));cum=0
    for rank,i in enumerate(order):cum=max(cum,(23-rank)*sec[i]['p_raw']);sec[i]['holm_p']=min(1.,cum)
    for t in testrows:t['positive_claim']=bool(t.get('mean_d',0)>0 and t.get('p_raw',1)<.05 and (t['subset']=='active' and t['maturity']=='VIX30D' and t['h']==10 or t.get('holm_p',1)<.05) and t.get('positive_survives_checks',False))
    savecsv(ROOT/'tests.csv',testrows);savecsv(ROOT/'leave_one_crisis_out.csv',loos);savecsv(ROOT/'bootstrap_draws.csv',boots)
    onset5=[]
    for e in feat['events']:
        j=e['index']+5
        if j<len(vd):
            q=pred[(pred.maturity=='VIX30D')&(pred.h==10)&(pred.date==str(vd[j].date()))]
            if len(q):onset5.append(dict(onset_id=e['onset_id'],onset_date=e['onset_date'],origin=str(vd[j].date()),har_loss=float(q.har_loss.iloc[0]),harx_loss=float(q.harx_loss.iloc[0]),d=float(q.d.iloc[0])))
    savecsv(ROOT/'descriptive_onset5.csv',onset5)
    atom(ROOT/'checkpoint.json',dict(status='computed_pending_Vera',spec_sha256=expected,code_sha256=freeze['code_sha256'],input_manifest_sha256=sha(ROOT/'input_manifest.json'),completed_units=completed,unit_count=len(completed),forecast_count=len(pred),audit_count=len(audit),elapsed_seconds=time.monotonic()-start))
if __name__=='__main__':main()
