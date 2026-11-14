"""Independent read-only verification of the archived E3 forecast outputs."""
import csv
import hashlib
import json
import math
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

ROOT = Path(__file__).parent
SPEC = ROOT.parent / 'E01_protocol' / 'E3_spec_v2.md'
RELEASE = ROOT.parent.parent / 'stages' / 'S05_release'
issues = []
inventory = {}

def check(ok, name):
    if not ok: issues.append(name)

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def close(a,b,tol=1e-11): return math.isclose(float(a),float(b),rel_tol=tol,abs_tol=tol)
def table(name): return pd.read_csv(ROOT / name)

manifest=json.loads((ROOT/'input_manifest.json').read_text())
check(sha(SPEC)==manifest['spec_sha256'], 'frozen spec hash')
check(sha(ROOT/'code'/'run_e3.py')==manifest['code_sha256'],'frozen runner hash')
pre=json.loads((ROOT/'pre_source_freeze.json').read_text())
check(pre['spec_sha256']==manifest['spec_sha256'] and pre['code_sha256']==manifest['code_sha256'],'pre-source freeze identifiers')
for name,record in manifest['sources'].items():
    path=ROOT/'raw'/f'{name}.csv'
    check(sha(path)==record['sha256'] and path.stat().st_size==record['bytes'],f'{name} source hash and size')
for line in (RELEASE/'manifest.sha256').read_text().splitlines():
    digest,name=line.split(maxsplit=1)
    check(sha(RELEASE/name)==digest,f'protected release {name}')
output_manifest=json.loads((ROOT/'output_manifest.json').read_text())
for item in output_manifest['files']:
    path=ROOT.parent.parent/item['path']
    check(path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],f'output manifest {item["path"]}')

p=table('predictions.csv'); losses=table('losses.csv'); tests=table('tests.csv')
check(len(p)==len(losses)==29272,'prediction and loss counts')
check(not p.duplicated(['maturity','h','date']).any(),'unique forecast keys')
for col in p:
    if col in losses and pd.api.types.is_numeric_dtype(p[col]):
        check(np.allclose(p[col],losses[col],rtol=0,atol=1e-12),f'loss source {col}')
check(np.allclose((p.y-p.har)**2,losses.har_loss,rtol=0,atol=1e-12),'HAR squared losses')
check(np.allclose((p.y-p.harx)**2,losses.harx_loss,rtol=0,atol=1e-12),'HAR-X squared losses')
check(np.allclose(losses.har_loss-losses.harx_loss,losses.d,rtol=0,atol=1e-12),'paired d identity')

# Source-level targets and causal shock state, independently traversed.
panels={}
for name in ['VIX30D','VIX9D','VIX3M','VIX6M']:
    raw=pd.read_csv(ROOT/'raw'/f'{name}.csv')
    raw['DATE']=pd.to_datetime(raw.DATE)
    raw=raw.sort_values('DATE').reset_index(drop=True)
    check(raw.DATE.is_unique and np.isfinite(raw.CLOSE).all() and (raw.CLOSE>0).all(),f'{name} valid source')
    panels[name]=raw
    for h in [5,10,20]:
        x=p[(p.maturity==name)&(p.h==h)]
        ix=pd.Index(raw.DATE).get_indexer(pd.to_datetime(x.date))
        expected=np.array([np.log(raw.CLOSE.iloc[i+1:i+h+1]).mean() for i in ix])
        check(np.allclose(expected,x.y,rtol=0,atol=1e-12),f'{name}/{h} direct target alignment')
        check((pd.to_datetime(x.target_end).to_numpy()==raw.DATE.iloc[ix+h].to_numpy()).all(),f'{name}/{h} target endpoints')
v=panels['VIX30D']; vals=v.CLOSE.to_numpy(float); vdates=v.DATE.to_numpy()
states=[]; accepted=[]; last=-1; peak=-math.inf; peak_i=-1
for i,value in enumerate(vals):
    trigger=(i>=1 and value-vals[i-1]>=7) or (i>=5 and value-vals[i-5]>=10)
    if trigger and (last<0 or (vdates[i]-vdates[last])/np.timedelta64(1,'D')>30):
        last=i;peak=value;peak_i=i;accepted.append(i)
    if last>=0 and value>peak:peak=value;peak_i=i
    active=last>=0 and i-last<120
    base=np.mean(vals[last-20:last]) if last>=20 else np.nan
    size=np.log(peak/base) if active and np.isfinite(base) else 0.
    change=np.log(value/vals[i-5]) if active and i>=5 else 0.
    states.append((int(active),i-last if last>=0 else -1,base if last>=0 else np.nan,
                   peak if last>=0 else np.nan,size,change))
features=table('features.csv');onsets=table('causal_onsets.csv')
check(accepted==onsets['index'].tolist(),'onsets from official closes')
check(np.array_equal(features.active.to_numpy(),np.array(states)[:,0]),'active state')
for key,col in [('age',1),('baseline',2),('running_max',3),('size',4),('five_change',5)]:
    check(np.allclose(features[key],np.array(states)[:,col],rtol=0,atol=1e-12,equal_nan=True),f'{key} causal feature')

# Rebuild calendar-index HAC, crisis means, and Holm from forecast rows.
results=[]
for row in tests.itertuples():
    q=losses[(losses.maturity==row.maturity)&(losses.h==row.h)].sort_values('date')
    pick=q.active.astype(bool).to_numpy() if row.subset=='active' else np.ones(len(q),bool)
    a=q.iloc[np.flatnonzero(pick)]; T=len(q); A=len(a); h=int(row.h)
    d=q.d.to_numpy(float); m=float(a.d.mean())
    g=np.where(pick,d-m,0)*(T/A if row.subset=='active' else 1)
    V=float(np.dot(g,g)/T+sum(2*(1-k/h)*np.dot(g[k:],g[:-k])/T for k in range(1,h)))
    F=math.sqrt((T+1-2*h+h*(h-1)/T)/T)
    stat=F*m/math.sqrt(V/T); prob=float(2*student_t.sf(abs(stat),T-1))
    K=a.loc[a.crisis_id>=0,'crisis_id'].nunique()
    episodes=q.loc[q.active==1,'onset'].nunique()
    expected={'T':T,'A':A,'paired_n':A,'K':K,'episodes':episodes,'har_mse':a.har_loss.mean(),
              'harx_mse':a.harx_loss.mean(),'mean_d':m,'relative_gain':m/a.har_loss.mean(),
              'V':V,'F':F,'dm_stat':stat,'p_raw':prob}
    for key,value in expected.items(): check(close(getattr(row,key),value),f'{row.maturity}/{h}/{row.subset} {key}')
    check(str(q.date.min())==row.first and str(q.date.max())==row.last,
          f'{row.maturity}/{h}/{row.subset} dates')
    results.append(dict(maturity=row.maturity,h=h,subset=row.subset,**expected))
check(len(results)==24,'24 slots')
secondary=tests.iloc[1:].copy().reset_index(drop=True)
order=sorted(range(len(secondary)),key=lambda i:(secondary.iloc[i].p_raw,i)); running=0
for rank,i in enumerate(order):
    running=max(running,(23-rank)*float(secondary.iloc[i].p_raw))
    check(close(secondary.iloc[i].holm_p,min(1,running)),f'Holm slot {i}')

# Whole crisis units, inactive runs and every saved bootstrap draw.
draws=table('bootstrap_draws.csv'); loo=table('leave_one_crisis_out.csv')
for row in tests.itertuples():
    q=losses[(losses.maturity==row.maturity)&(losses.h==row.h)].sort_values('date')
    a=q[q.active==1] if row.subset=='active' else q
    crisis={int(cid):z for cid,z in a[a.crisis_id>=0].groupby('crisis_id')}
    omitted=loo[(loo.maturity==row.maturity)&(loo.h==row.h)&(loo.subset==row.subset)]
    check(len(omitted)==len(crisis),f'{row.maturity}/{row.h}/{row.subset} omission count')
    for o in omitted.itertuples():
        z=a[a.crisis_id!=o.omitted_crisis]
        check(int(o.remaining_n)==len(z) and int(o.remaining_K)==len(crisis)-1
              and close(o.mean_d,z.d.mean()) and close(o.relative_gain,z.d.mean()/z.har_loss.mean()),
              f'{row.maturity}/{row.h}/{row.subset} omit {o.omitted_crisis}')
    blocks=[]; run=[]; lastj=None
    for rec in a.itertuples():
        if rec.crisis_id>=0:
            if run: blocks.extend([run[k:k+row.h] for k in range(0,len(run),row.h)]);run=[]
            lastj=None;continue
        if lastj is not None and rec.j!=lastj+1:
            blocks.extend([run[k:k+row.h] for k in range(0,len(run),row.h)]);run=[]
        run.append(rec.d);lastj=rec.j
    if run:blocks.extend([run[k:k+row.h] for k in range(0,len(run),row.h)])
    check(int(row.inactive_blocks)==len(blocks),f'{row.maturity}/{row.h}/{row.subset} inactive blocks')
    b=draws[(draws.maturity==row.maturity)&(draws.h==row.h)&(draws.subset==row.subset)]
    if len(crisis)<10:
        check(len(b)==0 and row.bootstrap_reason=='K_below_10',f'{row.maturity}/{row.h}/{row.subset} bootstrap unavailable')
    else:
        check(len(b)==1999 and b.replicate.tolist()==list(range(1999)),f'{row.maturity}/{row.h}/{row.subset} draw IDs')
        vals=b.mean_d.to_numpy(float)
        lo,hi=np.percentile(vals,[2.5,97.5],method='linear')
        check(close(lo,row.bootstrap_lo) and close(hi,row.bootstrap_hi)
              and close(np.mean(vals<=0),row.bootstrap_nonpositive),
              f'{row.maturity}/{row.h}/{row.subset} bootstrap summary')
        # Deterministic seed and resampling from whole, observation-weighted blocks.
        seedbytes=json.dumps([5902026,'Extension','E3','crisis_bootstrap',row.maturity,int(row.h),row.subset],
                             separators=(',',':')).encode()
        seed=int.from_bytes(hashlib.sha256(seedbytes).digest()[:8],'big')
        # pandas promotes this mixed integer/blank CSV column to float; check
        # the serialized decimal's precision, then reproduce the draws.
        check(close(seed,row.bootstrap_seed,1e-15),f'{row.maturity}/{row.h}/{row.subset} bootstrap seed')
        rng=np.random.Generator(np.random.PCG64(seed))
        unitlist=[z.d.to_numpy(float) for z in crisis.values()]
        blocklist=[np.array(z,float) for z in blocks]
        for rep,observed in enumerate(vals):
            chosen=[unitlist[k] for k in rng.integers(0,len(unitlist),len(unitlist))]
            if row.subset=='all':chosen += [blocklist[k] for k in rng.integers(0,len(blocklist),len(blocklist))]
            recomputed=sum(z.sum() for z in chosen)/sum(len(z) for z in chosen)
            if not close(recomputed,observed,1e-10):issues.append(f'{row.maturity}/{row.h}/{row.subset} bootstrap draw {rep}')

# Calendar month and training endpoint boundaries for every fitted vintage.
models=table('monthly_models.csv'); dep=table('origin_dependency_ledger.csv')
check(len(models)==1740 and models.error.isna().all(),'fitted vintage count and errors')
check(len(dep)==36549,'dependency ledger count')
check(len(dep[dep.reason.isna()])==len(p),'paired origin ledger count')
check((pd.to_datetime(dep.feature_max_date.dropna())<=pd.to_datetime(dep.loc[dep.feature_max_date.notna(),'date'])).all(),'feature dates no later than origin')
cut=pd.to_datetime(dep.refit_cutoff); origin=pd.to_datetime(dep.date)
check((cut<origin).all(),'monthly cutoff precedes origins')
end=pd.to_datetime(dep.latest_train_label,errors='coerce')
check((end[end.notna()]<=cut[end.notna()]).all(),'training labels matured by cutoff')
for field in ['onset_date','baseline_start','baseline_end','running_max_source_date','age_source_date','five_change_end']:
    val=pd.to_datetime(dep[field],errors='coerce')
    check((val[val.notna()]<=origin[val.notna()]).all(),f'{field} causal boundary')
check(all(str(pd.Timestamp(d).to_period('M'))==mid.split('_')[-1] for d,mid in zip(p.date,p.model_id)), 'prediction-month vintage alignment')
check(all(pd.Timestamp(c)==pd.Timestamp(d).replace(day=1)-pd.Timedelta(days=1) for c,d in zip(p.cutoff,p.date)), 'previous month end cutoff')
with zipfile.ZipFile(ROOT/'monthly_models.zip') as archive:
    check(len(archive.namelist())==1740,'model archive count')
    model_manifest=json.loads((ROOT/'monthly_models_manifest.json').read_text())
    check(sha(ROOT/'monthly_models.zip')==model_manifest['archive_sha256'],'model archive hash')
    entries={e['name']:e for e in model_manifest['entries']}
    check(set(entries)==set(archive.namelist()),'model archive entry names')
    for item in archive.namelist():
        raw=archive.read(item)
        check(hashlib.sha256(raw).hexdigest()==entries[item]['sha256'] and len(raw)==entries[item]['bytes'],f'{item} manifest hash and size')
        obj=json.loads(raw); model=obj['model']; cutoff=pd.Timestamp(obj['cutoff'])
        check(pd.Timestamp(model['train_max_label'])<=cutoff,f'{item} training endpoint')
        years=[cutoff.year-2,cutoff.year-1,cutoff.year] if cutoff.month==12 and cutoff.day==31 else [cutoff.year-3,cutoff.year-2,cutoff.year-1]
        check([f['year'] for f in model['folds']]==years,f'{item} three complete validation years')
        for fold in model['folds']:
            year=fold['year']
            check(pd.Timestamp(fold['train_max_label'])<pd.Timestamp(year=year,month=1,day=1)
                  and pd.Timestamp(fold['validation_max_label'])<=pd.Timestamp(year=year,month=12,day=31)
                  and fold['train_n']>0 and fold['validation_n']>0,f'{item} fold {year} boundary')
        check(model['tau'] in [5,10,20,40,80,120],f'{item} tau grid')
        pooled={}
        total=sum(f['validation_n'] for f in model['folds'])
        for tau in [5,10,20,40,80,120]:
            pooled[tau]=sum(f['validation_n']*f[f'tau_{tau}_mse'] for f in model['folds'])/total
            check(close(pooled[tau],model['tau_mse'][str(tau)]),f'{item} pooled tau {tau}')
        check(model['tau']==min(pooled,key=lambda tau:(pooled[tau],tau)),f'{item} selected tau')

audit=table('leakage_audit_v2.csv'); summary=json.loads((ROOT/'leakage_audit_v2_summary.json').read_text())
check(len(audit)==1140 and audit.date.nunique()==253,'audit comparison/date count')
check(audit[['exact','model_match','boundary']].all().all(),'saved audit flags')
check(summary['comparison_rows']==len(audit) and summary['selected_dates']==audit.date.nunique(),'audit summary')
check(np.allclose(audit.prefix_har,audit.perturbed_har,rtol=0,atol=1e-12)
      and np.allclose(audit.prefix_harx,audit.perturbed_harx,rtol=0,atol=1e-12),'audit changed-suffix predictions')
exact_saved_har=int((audit.prefix_har==audit.saved_har).sum())
exact_saved_harx=int((audit.prefix_harx==audit.saved_harx).sum())
exact_perturbed_har=int((audit.prefix_har==audit.perturbed_har).sum())
exact_perturbed_harx=int((audit.prefix_harx==audit.perturbed_harx).sum())
seedbytes=json.dumps([5902026,'Extension','E3','truncation_audit'],separators=(',',':')).encode()
audit_seed=int.from_bytes(hashlib.sha256(seedbytes).digest()[:8],'big')
rng=np.random.Generator(np.random.PCG64(audit_seed)); dates=sorted(p.date.unique())
selected=set(rng.choice(dates,size=min(200,len(dates)),replace=False))
for _,grp in p.groupby(p.date.str[:4]):selected.update((grp.date.min(),grp.date.max()))
check(set(audit.date)==selected,'prespecified audit date sample')
check(close(audit.seed.iloc[0],audit_seed,1e-15),'audit seed')
check(all(audit.apply(lambda r: r.model_sha256==entries[r.model_id+'.json']['sha256'],axis=1)),'audited model hashes')
check(len(table('causal_onsets.csv'))==34,'accepted onset count')
check(len(table('coverage.csv'))==4 and (table('coverage.csv').excluded==0).all(),'source coverage')
inventory.update(forecasts=len(p),test_slots=len(tests),bootstrap_draws=len(draws),crisis_omissions=len(loo),
                 primary=results[0],all_tests=tests.replace({np.nan:None}).to_dict(orient='records'),
                 source_coverage=table('coverage.csv').to_dict(orient='records'),
                 first_scored_by_maturity=p.groupby('maturity').date.min().to_dict(),
                 origin_exclusions=table('origin_exclusions.csv').reason.value_counts().to_dict(),
                 holm_below_05=int((secondary.holm_p<.05).sum()),
                 negative_d_slots=int((tests.mean_d<0).sum()),positive_claims=int((tests.positive_claim==True).sum()),
                 primary_active_first=str(losses[(losses.maturity=='VIX30D')&(losses.h==10)&(losses.active==1)].date.min()),
                 primary_active_last=str(losses[(losses.maturity=='VIX30D')&(losses.h==10)&(losses.active==1)].date.max()),
                 audit_dates=audit.date.nunique(),audit_rows=len(audit),monthly_models=len(models),
                 exact_saved_har=exact_saved_har,exact_saved_harx=exact_saved_harx,
                 exact_perturbed_har=exact_perturbed_har,exact_perturbed_harx=exact_perturbed_harx,
                 max_abs_saved_har=float(np.max(np.abs(audit.prefix_har-audit.saved_har))),
                 max_abs_saved_harx=float(np.max(np.abs(audit.prefix_harx-audit.saved_harx))),
                 dependency_rows=len(dep),issues=issues)
(ROOT/'vera_verified_inventory.json').write_text(json.dumps(inventory,indent=2,default=str)+'\n')
print(json.dumps({'issues':len(issues),'first_issues':issues[:30],'inventory':inventory},default=str))
raise SystemExit(bool(issues))
