"""E5 observed-cohort pipeline. Run only after freezing this file and inputs."""
import argparse, csv, hashlib, json, math, os, sys, time
from datetime import datetime, timezone
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'stages/S01_protocol/attempt_02'))
import protocol_core as core
FAMILIES=('E1','E2','PL','SE')
MATS=('VIX9D','VIX30D','VIX3M','VIX6M')
PUBLIC=('VXN','RVX','OVX','GVZ')
SOURCES={**{m:ROOT/f'extension/E03_forecast/raw/{m}.csv' for m in MATS},
         **{m:OUT/f'private_sources/{m}_History.csv' for m in PUBLIC}}
LEGACY=ROOT/'omori_pipeline/data/shock_events.csv'
SPEC=OUT/'frozen_spec_v1.md'
CODE=OUT/'e5_pipeline.py'
CONTRACT=ROOT/'stages/S01_protocol/attempt_02/protocol_core.py'

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False,default=str)+'\n');os.replace(tmp,path)
def write_csv(path,rows,fields):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
    os.replace(tmp,path)
def check_freeze():
    mf=json.loads((OUT/'execution_freeze.json').read_text())
    for p,h in mf['hashes'].items():
        if sha(ROOT/p)!=h: raise RuntimeError(f'freeze mismatch: {p}')
    return mf

def load_series(path):
    d=pd.read_csv(path)
    field='CLOSE' if 'CLOSE' in d else path.stem.split('_')[0]
    if not {'DATE',field} <= set(d): raise ValueError(f'{path}: DATE and {field} required')
    dates=pd.to_datetime(d.DATE,format='%m/%d/%Y',errors='raise')
    vals=pd.to_numeric(d[field],errors='raise').to_numpy(float)
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError(f'{path}: dates invalid')
    if not np.all(np.isfinite(vals)&(vals>0)): raise ValueError(f'{path}: close invalid')
    return dates.to_numpy(dtype='datetime64[D]'),vals

def causal_onsets(dates,v):
    log=np.log(v); trigger=np.zeros(len(v),bool)
    trigger[1:] |= log[1:]-log[:-1]>=np.log(1+7/20)
    trigger[5:] |= log[5:]-log[:-5]>=np.log(1+10/20)
    ids=[]
    for i in np.flatnonzero(trigger):
        if not ids or dates[i]-dates[ids[-1]]>np.timedelta64(30,'D'): ids.append(int(i))
    return ids

def legacy_onsets(dates):
    ev=pd.read_csv(LEGACY)
    # Inherited S02a event namespace combines Lehman and GFC Peak at the Lehman onset.
    ev=ev[ev.label.fillna('')!='GFC Peak']
    lookup={str(d):i for i,d in enumerate(dates)}
    out=[];missing=[]
    for x in ev.onset_date:
        if x not in lookup: missing.append(x)
        else: out.append(lookup[x])
    return sorted(out),missing

def panels_for(index,dates,v,onsets,namespace):
    rows=[]; trajectories={}; intervals={}
    for j,on in enumerate(onsets):
        eid=f'{namespace}:{str(dates[on])}'
        reason=[]; next_on=onsets[j+1] if j+1<len(onsets) else None
        if on<250: reason.append('insufficient_pre_onset')
        if on+10>=len(v): reason.append('incomplete_peak_search')
        anchor=on+int(np.argmax(v[on:min(on+11,len(v))]))
        if anchor+120>=len(v): reason.append('incomplete_anchor_response')
        if next_on is not None and next_on<=anchor+120: reason.append('next_onset_before_response_end')
        den=None
        if not reason:
            mu=float(np.log(v[on-250:on]).mean());den=float(np.log(v[anchor])-mu)
            if den<=1e-8: reason.append('nonpositive_normalization')
        row=dict(index=index,event_id=eid,onset=str(dates[on]),anchor=str(dates[anchor]),
                 footprint_first=str(dates[on-250]) if on>=250 else '',
                 footprint_last=str(dates[anchor+120]) if anchor+120<len(v) else '',
                 next_onset=str(dates[next_on]) if next_on is not None else '',
                 denominator=den if den is not None else '',eligible=int(not reason),
                 exclusion='|'.join(reason))
        rows.append(row)
        if not reason:
            trajectories[(index,eid)]=((np.log(v[anchor:anchor+121])-mu)/den).tolist()
            intervals[(index,eid)]=(dates[on-250],dates[anchor+120])
    return rows,trajectories,intervals

def components(intervals):
    keys=sorted(intervals,key=lambda x:(intervals[x][0],intervals[x][1],x))
    groups=core.overlap_groups([intervals[x] for x in keys])
    members=defaultdict(list)
    for k,g in zip(keys,groups): members[int(g)].append(k)
    return {k:f'C{g:03d}' for g,ks in members.items() for k in ks},members

def prepare():
    check_freeze(); allrows=[]; trajectories={}; intervals={}; source_rows=[]
    for index,path in SOURCES.items():
        dates,v=load_series(path)
        if index in MATS: on,missing=legacy_onsets(dates)
        else: on,missing=causal_onsets(dates,v),[]
        rows,tr,iv=panels_for(index,dates,v,on,'legacy' if index in MATS else 'causal')
        for date in missing:
            rows.append(dict(index=index,event_id=f'legacy:{date}',onset=date,anchor='',
                footprint_first='',footprint_last='',next_onset='',denominator='',
                eligible=0,exclusion='onset_absent_from_source_calendar'))
        allrows+=rows;trajectories.update(tr);intervals.update(iv)
        source_rows.append(dict(index=index,file=str(path.relative_to(ROOT)),sha256=sha(path),
                                rows=len(v),first=str(dates[0]),last=str(dates[-1]),onsets=len(on),
                                eligible=sum(r['eligible'] for r in rows)))
    common=set.intersection(*(set(e for i,e in trajectories if i==m) for m in MATS))
    assignment,members=components(intervals)
    for row in allrows:
        key=(row['index'],row['event_id']);row['cluster']=assignment.get(key,'')
        row['common_vix']=int(row['index'] in MATS and row['event_id'] in common)
    cluster_rows=[]
    for g,ks in members.items():
        cluster_rows.append(dict(cluster=f'C{g:03d}',members=len(ks),
            indices='|'.join(sorted(set(i for i,e in ks))),
            event_ids='|'.join(f'{i}:{e}' for i,e in ks),
            first=str(min(intervals[k][0] for k in ks)),last=str(max(intervals[k][1] for k in ks))))
    write_csv(OUT/'source_manifest.csv',source_rows,list(source_rows[0]))
    write_csv(OUT/'eligibility.csv',allrows,list(allrows[0]))
    write_csv(OUT/'clusters.csv',cluster_rows,list(cluster_rows[0]))
    write_json(OUT/'prepared.json',dict(trajectories={f'{i}|{e}':z for (i,e),z in trajectories.items()},
        assignment={f'{i}|{e}':c for (i,e),c in assignment.items()},common=sorted(common),
        source_hashes={r['index']:r['sha256'] for r in source_rows},
        source_counts=source_rows,cluster_count=len(members)))
    return len(trajectories),len(members),len(common)

def panel_keys(prepared):
    tr=prepared['trajectories'];common=set(prepared['common'])
    out={}
    for m in SOURCES:
        keys=[k for k in tr if k.startswith(m+'|')]
        if keys: out[f'{m}:full']=keys
        if m in MATS:
            keys=[k for k in keys if k.split('|',1)[1] in common]
            if keys: out[f'{m}:common']=keys
    return out

def fit_exact(family,t,y,index,cohort,start,draw_id):
    """protocol_core.fit semantics with E5's specified family-before-draw seed key."""
    lo,hi=map(lambda x:np.asarray(x,float),core.BOUNDS[family])
    rng=np.random.default_rng(core.seed('Extension','E5',index,cohort,start,family,draw_id))
    records=[];best=None
    for i in range(32):
        x=lo+(hi-lo)*rng.random(len(lo))
        for j in {'E1':[2],'E2':[3,4],'PL':[2,3],'SE':[2]}[family]:
            x[j]=math.exp(rng.uniform(math.log(lo[j]),math.log(hi[j])))
        sol=least_squares(lambda p:core.curve(family,t,core.ordered(family,p))-y,x,
            bounds=(lo,hi),method='trf',ftol=1e-8,xtol=1e-8,gtol=1e-8,max_nfev=5000)
        sse=float(np.sum(sol.fun**2));p=core.ordered(family,sol.x)
        ok=bool(sol.success and np.isfinite(sse) and np.all(np.isfinite(p)))
        rec=dict(start=i,success=ok,status=int(sol.status),nfev=int(sol.nfev),
                 sse=sse,parameters=p.tolist(),active_mask=sol.active_mask.tolist())
        records.append(rec)
        if ok and (best is None or sse<best['sse']): best=rec
    return dict(best=best,starts=records,k=len(lo),n=len(t))

def fit_panel(panel,keys,prepared,draw_id=None):
    tr=prepared['trajectories'];assignment=prepared['assignment']
    clusters=sorted(set(assignment[k] for k in keys))
    if draw_id is None: selected=clusters; multiplicity={c:1 for c in clusters}
    else:
        panel_key='VIX:common' if panel.endswith(':common') else panel
        rng=np.random.default_rng(core.seed('Extension','E5','cluster_bootstrap',panel_key,draw_id))
        selected=rng.choice(clusters,size=len(clusters),replace=True).tolist()
        multiplicity=dict(Counter(selected))
    expanded=[k for k in keys for _ in range(multiplicity.get(assignment[k],0))]
    result=dict(panel=panel,draw_id=draw_id,base_clusters=len(clusters),
                sampled_clusters=selected,multiplicity=multiplicity,events=len(expanded),fits=[])
    if not expanded:
        result['failure']='empty_draw';return result
    y=np.mean(np.asarray([tr[k] for k in expanded]),axis=0)
    for start in (1,5):
        for family in FAMILIES:
            try:
                f=fit_exact(family,np.arange(start,121),y[start:],panel.split(':')[0],
                            panel.split(':')[1],start,draw_id)
                best=f['best'];score=core.score(best['sse'],f['n'],f['k']) if best else None
                rec=dict(start=start,family=family,best=best,starts=f['starts'],score=score,
                         tau=best['parameters'][2] if best and family=='E1' else None)
            except Exception as exc: rec=dict(start=start,family=family,error=repr(exc),best=None,starts=[],score=None,tau=None)
            result['fits'].append(rec)
    return result

def fit_path(panel,draw):
    slug=panel.replace(':','_');return OUT/'units'/slug/(f'{draw:04d}.json' if draw is not None else 'point.json')

def run(start_draw,end_draw,point=False):
    mf=check_freeze();prepared=json.loads((OUT/'prepared.json').read_text())
    if prepared['source_hashes']!={r['index']:sha(SOURCES[r['index']]) for r in prepared['source_counts']}:
        raise RuntimeError('source snapshot changed')
    keys=panel_keys(prepared); t0=time.monotonic();done=0
    for panel,pkeys in keys.items():
        for draw in ([None] if point else range(start_draw,end_draw)):
            path=fit_path(panel,draw)
            if path.exists():
                old=json.loads(path.read_text())
                if old.get('freeze_sha')==mf['freeze_sha'] and old.get('panel')==panel and old.get('draw_id')==draw: continue
                raise RuntimeError(f'corrupt/stale unit: {path}')
            obj=fit_panel(panel,pkeys,prepared,draw)
            obj.update(freeze_sha=mf['freeze_sha'],source_hashes=prepared['source_hashes'])
            write_json(path,obj);done+=1
            if time.monotonic()-t0>7*3600: return done
    return done

def fixture():
    d=np.arange('2020-01-01','2022-12-01',dtype='datetime64[D]')
    v=np.full(len(d),20.0);v[300]=30;v[301:]=20
    assert causal_onsets(d,v)==[300]
    rows,tr,iv=panels_for('X',d,v,[300],'fixture')
    assert rows[0]['eligible']==1 and len(next(iter(tr.values())))==121
    ass,mem=components({('A','1'):(d[0],d[250]),('B','2'):(d[250],d[500]),('C','3'):(d[501],d[600])})
    assert ass[('A','1')]==ass[('B','2')] and ass[('C','3')]!=ass[('A','1')]
    p={'trajectories':{'X|fixture:1':list(np.exp(-np.arange(121)/20)),
                       'X|fixture:2':list(np.exp(-np.arange(121)/30))},
       'assignment':{'X|fixture:1':'C000','X|fixture:2':'C001'}}
    r=fit_panel('X:full',list(p['trajectories']),p,0)
    assert len(r['sampled_clusters'])==2 and len(r['fits'])==8
    assert sum(r['multiplicity'].values())==2 and all(len(f['starts'])==32 for f in r['fits'])
    return dict(status='fixture_pass',fit_records=len(r['fits']),starts_per_fit=32,seed=core.SEED)

def summarize():
    check_freeze(); prepared=json.loads((OUT/'prepared.json').read_text()); keys=panel_keys(prepared)
    fitrows=[]; intervalrows=[]; drawrows=[]; point={}; draws=defaultdict(dict)
    for panel in keys:
        path=fit_path(panel,None)
        if not path.exists(): continue
        obj=json.loads(path.read_text());point[panel]=obj
        for f in obj['fits']:
            fitrows.append(dict(panel=panel,start=f['start'],family=f['family'],
                score=f['score'],sse=f['best']['sse'] if f['best'] else '',
                parameters=json.dumps(f['best']['parameters']) if f['best'] else '',
                tau=f['tau'],base_clusters=obj['base_clusters'],events=obj['events'],
                successes=sum(s['success'] for s in f['starts']),
                boundary_starts=sum(any(x!=0 for x in s['active_mask']) for s in f['starts']),
                error=f.get('error','')))
        for d in range(999):
            path=fit_path(panel,d)
            if not path.exists(): continue
            o=json.loads(path.read_text());draws[panel][d]=o
            drawrows.append(dict(panel=panel,draw=d,cluster_ids='|'.join(o['sampled_clusters']),
                multiplicity=json.dumps(o['multiplicity'],sort_keys=True),events=o['events'],
                successful_fits=sum(f['best'] is not None for f in o['fits']),
                failed_fits=sum(f['best'] is None for f in o['fits'])))
    for row in fitrows:
        panel=row['panel'];start=row['start'];family=row['family']
        vals=[next((f['tau'] for f in o['fits'] if f['start']==start and f['family']==family),None)
              for o in draws[panel].values()]
        vals=[v for v in vals if v is not None and np.isfinite(v)]
        estimable=bool(family=='E1' and row['base_clusters']>=10 and len(vals)>=900)
        intervalrows.append(dict(panel=panel,start=start,family=family,
            point_tau=row['tau'],base_clusters=row['base_clusters'],valid_draws=len(vals),
            missing_draws=999-len(draws[panel]),estimable=int(estimable),
            lo=float(np.percentile(vals,2.5)) if estimable else '',
            hi=float(np.percentile(vals,97.5)) if estimable else '',
            reason='' if estimable else ('not_E1' if family!='E1' else 'too_few_clusters_or_draws')))
    contrasts=[]
    for start in (5,1):
        for mat in ('VIX9D','VIX3M','VIX6M'):
            pa=f'{mat}:common';pb='VIX30D:common'
            if pa not in point or pb not in point:
                contrasts.append(dict(start=start,contrast=f'{mat}-VIX30D',estimable=0,reason='missing_common_panel'));continue
            def tau(o):
                return next((f['tau'] for f in o['fits'] if f['start']==start and f['family']=='E1'),None)
            ta,tb=tau(point[pa]),tau(point[pb]);diffs=[]
            for d in range(999):
                if d in draws[pa] and d in draws[pb]:
                    a,b=tau(draws[pa][d]),tau(draws[pb][d])
                    if a is not None and b is not None: diffs.append(a-b)
            k=min(point[pa]['base_clusters'],point[pb]['base_clusters'])
            good=k>=10 and len(diffs)>=900 and ta is not None and tb is not None
            if any(x==0 for x in diffs): good=False
            raw=min(1,2*min((1+sum(x<=0 for x in diffs))/(len(diffs)+1),
                              (1+sum(x>=0 for x in diffs))/(len(diffs)+1))) if good else None
            contrasts.append(dict(start=start,contrast=f'{mat}-VIX30D',point_difference=ta-tb if ta is not None and tb is not None else '',
                base_clusters=k,valid_draws=len(diffs),estimable=int(good),
                lo=float(np.percentile(diffs,2.5)) if good else '',
                hi=float(np.percentile(diffs,97.5)) if good else '',raw_p=raw,holm_p='',
                reason='' if good else 'too_few_clusters_draws_or_zero_ties'))
    eligible=sorted((r for r in contrasts if r['start']==5 and r['estimable']),key=lambda r:r['raw_p'])
    running=0
    for j,r in enumerate(eligible):
        running=max(running,(len(eligible)-j)*r['raw_p']);r['holm_p']=min(1,running)
    if fitrows: write_csv(OUT/'point_fits.csv',fitrows,list(fitrows[0]))
    if intervalrows: write_csv(OUT/'tau_intervals.csv',intervalrows,list(intervalrows[0]))
    if drawrows: write_csv(OUT/'bootstrap_draws.csv',drawrows,list(drawrows[0]))
    write_csv(OUT/'maturity_contrasts.csv',contrasts,
        ['start','contrast','point_difference','base_clusters','valid_draws','estimable','lo','hi','raw_p','holm_p','reason'])
    write_json(OUT/'summary_status.json',dict(point_panels=len(point),expected_panels=len(keys),
        saved_draw_units=sum(map(len,draws.values())),expected_draw_units=999*len(keys),
        base_clusters=prepared['cluster_count'],common_events=len(prepared['common'])))
    return len(point),sum(map(len,draws.values()))

def freeze():
    if (OUT/'execution_freeze.json').exists(): raise RuntimeError('freeze already exists')
    paths=[SPEC,CODE,CONTRACT,LEGACY,*SOURCES.values()]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}
    obj=dict(hashes=hashes,seed=core.SEED,python=sys.version,numpy=np.__version__,
             pandas=pd.__version__,scipy=__import__('scipy').__version__,
             frozen_at_utc=datetime.now(timezone.utc).isoformat(),
             status='new review-informed public execution freeze; original chronology unchanged')
    obj['freeze_sha']=hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()
    write_json(OUT/'execution_freeze.json',obj)
    return obj['freeze_sha']

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['fixture','freeze','prepare','point','bootstrap','summarize'])
    ap.add_argument('--start',type=int,default=0);ap.add_argument('--end',type=int,default=999)
    a=ap.parse_args()
    if a.mode=='fixture': print(json.dumps(fixture()))
    elif a.mode=='freeze': print(freeze())
    elif a.mode=='prepare': print(prepare())
    elif a.mode=='point': print('completed',run(0,0,True))
    elif a.mode=='summarize': print('summarized',summarize())
    else: print('completed',run(a.start,a.end))
