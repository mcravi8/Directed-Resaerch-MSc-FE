"""Frozen E2 executor. Run from the task project root; --mode prepare/run/summarize."""
import argparse, csv, hashlib, json, math, os, platform, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import scipy

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'extension/E02_sample_size'
sys.path.insert(0,str(ROOT/'stages/S01_protocol/attempt_02'))
sys.path.insert(0,str(ROOT/'stages/S02a_pooled/scripts'))
import protocol_core as core
import pipeline

TRUTHS=('E1','E2','PL','SE'); NS=(19,40,80,160,320)
DESIGNS=('independent','calendar_joint'); STARTS=(1,5)
SPEC=ROOT/'extension/E01_protocol/E2_spec_v2.md'
INPUTS=[SPEC, ROOT/'omori_pipeline/data/vix_term_structure.csv',ROOT/'omori_pipeline/data/shock_events.csv',ROOT/'stages/S02a_pooled/pooled_summary.json',ROOT/'stages/S01_protocol/attempt_02/protocol_core.py',ROOT/'stages/S02a_pooled/scripts/pipeline.py',Path(__file__)]

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n')
    os.replace(tmp,path)
def manifest(): return {str(p.relative_to(ROOT)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in INPUTS}
def check():
    saved=json.loads((OUT/'input_manifest.json').read_text())
    assert manifest()==saved['files'], 'input/spec/code hash mismatch'
    return saved
def prepare():
    assert sha(SPEC)=='72ed76057673830ef7d467c043cef9baa0bbeadd50911302c8965ab9c7000604'
    m=dict(files=manifest(),seed=core.SEED,python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,created=time.time())
    atomic(OUT/'input_manifest.json',m)
    return m

def calibration():
    check()
    ts,ev=pipeline.inputs(); panel,_,rows=pipeline.panels(ts,ev,'VIX30D',120,'argmax')
    ids=sorted(panel); assert len(ids)==19, f'cohort {len(ids)} != 19'
    z=np.array([panel[i] for i in ids]); assert z.shape==(19,121) and np.allclose(z[:,0],1)
    summ=json.loads((ROOT/'stages/S02a_pooled/pooled_summary.json').read_text())['summaries']
    def historical(start): return next(x for x in summ if x['unit']=='empirical_120_VIX30D_full_event' and x['start']==start and not x['zero_baseline'])
    e2=historical(1)['models']['E2']['parameters']; b=historical(5)['models']['E1']['parameters'][0]
    w,t1,t2=e2[2:]; y20=float(z.mean(axis=0)[20]); r=(y20-b)/(1-b)
    assert 0<r<1 and t1<=t2
    c=20/(r**(-1/.7)-1); scale=20/(-math.log(r))**(1/.6); t=np.arange(121)
    curves=dict(E1=b+(1-b)*np.exp(-t/27.8),E2=b+(1-b)*(w*np.exp(-t/t1)+(1-w)*np.exp(-t/t2)),PL=b+(1-b)*(1+t/c)**(-.7),SE=b+(1-b)*np.exp(-(t/scale)**.6))
    residual=z-core.curve('E2',t,e2); residual-=residual.mean(axis=0)
    dates=ts.index; lookup={x:i for i,x in enumerate(dates)}
    erows={x['event_id']:x for x in rows if x['eligible']}
    anchors={eid:int(erows[eid]['anchor_index']) for eid in ids}
    assert all(dates[anchors[eid]] in lookup for eid in ids)
    groups=[]; active=[]; end=-1
    for eid in sorted(ids,key=lambda q:(anchors[q],q)):
        a=anchors[eid]
        if a>end and active: groups.append(active);active=[]
        active.append(eid);end=max(end,a+120)
    if active:groups.append(active)
    templates=[]
    for j,g in enumerate(groups):
        base=min(anchors[e] for e in g); span=max(anchors[e]+120 for e in g)-base+1
        masks={e:list(range(anchors[e]-base,anchors[e]-base+121)) for e in g}
        templates.append(dict(id=j,event_ids=sorted(g),base=base,span=span,offsets={e:anchors[e]-base for e in g},masks=masks))
    bank=defaultdict(list)
    for i,eid in enumerate(ids):
        for j,val in enumerate(residual[i]):bank[anchors[eid]+j].append(float(val))
    values={k:float(np.mean(v)) for k,v in bank.items()}
    runs=[]; run=[]; prev=None
    for k in sorted(values):
        if prev is not None and k!=prev+1:runs.append(run);run=[]
        run.append(k);prev=k
    if run:runs.append(run)
    meta=dict(ids=ids,anchors=anchors,templates=templates,runs=runs,calendar_values=values,parameters=dict(baseline=b,empirical_z20=y20,ratio=r,E2_historical=e2,E2_weight=w,E2_fast=t1,E2_slow=t2,PL_c=c,SE_scale=scale),eligible_rows=[erows[e] for e in ids],source_hashes=manifest())
    atomic(OUT/'calibration.json',meta)
    return curves,residual,meta

def calendar_panel(N,rng,audit_rng,meta):
    values={int(k):v for k,v in meta['calendar_values'].items()}; runs=meta['runs']; templates=meta['templates']
    event_noise=[]; draws=[]; boundaries=[]; draw_id=0
    while len(event_noise)<N:
        template=templates[int(rng.integers(len(templates)))]; candidates=template['event_ids']; m=len(candidates); k=min(m,N-len(event_noise))
        chosen=sorted(candidates if k==m else audit_rng.choice(candidates,size=k,replace=False).tolist())
        field=np.empty(template['span']); blocks=[]; cursor=0
        while cursor<len(field):
            ri=int(rng.integers(len(runs))); run=runs[ri]; pos=int(rng.integers(len(run)))
            count=min(10,len(field)-cursor)
            field[cursor:cursor+count]=[values[run[(pos+j)%len(run)]] for j in range(count)]
            blocks.append(dict(run=ri,start=pos,count=count,source_dates=[int(run[(pos+j)%len(run)]) for j in range(count)]));cursor+=count
        for eid in chosen:
            off=template['offsets'][eid]; event_noise.append(field[off:off+121]-field[off])
        record=dict(draw_id=draw_id,source_component=template['id'],candidate_ids=candidates,selected_ids=chosen,excluded_ids=sorted(set(candidates)-set(chosen)),m=m,k=k,inclusion_probability=k/m,offsets={e:template['offsets'][e] for e in chosen},masks={e:template['masks'][e] for e in chosen},blocks=blocks,field=field.tolist())
        draws.append(record)
        if k<m:boundaries.append({x:record[x] for x in ('draw_id','source_component','candidate_ids','selected_ids','excluded_ids','m','k','inclusion_probability','offsets','masks')})
        draw_id+=1
    return np.array(event_noise),draws,boundaries

def independent_panel(N,rng,residual):
    panel=np.empty((N,121)); source=[]; blocks=[]
    for i in range(N):
        row=int(rng.integers(19)); positions=[]; bb=[]
        while len(positions)<121:
            j=int(rng.integers(121));positions.extend((j+np.arange(10)).tolist());bb.append(j)
        raw=residual[row,np.array(positions[:121])%121];panel[i]=raw-raw[0]
        source.append(row);blocks.append(bb)
    return panel,dict(source_rows=source,block_starts=blocks)

def fit_case(y,start,unit):
    models={};scores={}
    for family in TRUTHS:
        result=core.fit(family,np.arange(start,121),y[start:],['Extension','E2',*unit,'fit',start],False)
        best=result['best'];score=core.score(best['sse'],result['n'],result['k']) if best else None
        result['score']=score;models[family]=result;scores[family]=score
    if any(v is None for v in scores.values()):winner='unclassified_fit_failure';ties=[]
    else:
        best=min(scores.values());ties=[f for f in TRUTHS if scores[f]==best];winner=ties[0]
    return dict(winner=winner,ties=ties,models=models,fit_seed=core.seed('Extension','E2',*unit,'fit',start))

def unit_path(truth,N,design,rep):return OUT/'units'/truth/str(N)/design/f'{rep:03d}.json'
def job(truth,N,design,rep,curves,residual,meta):
    unit=(truth,N,design,rep); seed=core.seed('Extension','E2',*unit,'panel')
    rng=np.random.default_rng(seed); audit_rng=np.random.default_rng(core.seed('Extension','E2',*unit,'boundary_audit'))
    if design=='independent':noise,draws=independent_panel(N,rng,residual);boundaries=[];component_draw_count=N;distinct_source_count=None
    else:
        noise,draws,boundaries=calendar_panel(N,rng,audit_rng,meta)
        component_draw_count=len(draws);distinct_source_count=len({d['source_component'] for d in draws})
    y=(curves[truth][None,:]+noise).mean(axis=0)
    starts={str(s):fit_case(y,s,unit) for s in STARTS}
    return dict(unit_id='/'.join(map(str,unit)),truth=truth,N=N,design=design,rep=rep,panel_seed=seed,audit_seed=core.seed('Extension','E2',*unit,'boundary_audit'),spec_hash=sha(SPEC),input_manifest_hash=sha(OUT/'input_manifest.json'),code_hash=sha(__file__),component_draw_count=component_draw_count,distinct_source_component_count=distinct_source_count,draws=draws,boundaries=boundaries,starts=starts)

def fixture():
    rng=np.random.default_rng(core.seed('Extension','E2','fixture'))
    template=dict(id=0,event_ids=['a','b','c'],base=0,span=125,offsets={'a':0,'b':2,'c':4},masks={e:list(range(i,i+121)) for e,i in [('a',0),('b',2),('c',4)]})
    meta=dict(templates=[template],runs=[list(range(125))],calendar_values={i:float(i) for i in range(125)})
    noise,draws,boundary=calendar_panel(2,rng,np.random.default_rng(core.seed('Extension','E2','fixture','boundary_audit')),meta)
    d=draws[0];assert len(boundary)==1 and len(d['selected_ids'])==2 and len(d['excluded_ids'])==1
    a,b=d['selected_ids'];oa,ob=d['offsets'][a],d['offsets'][b]
    shared=oa+max(0,ob-oa);assert shared>=ob and shared<oa+121 and shared<ob+121
    assert np.isclose(noise[0,shared-oa]+d['field'][oa],noise[1,shared-ob]+d['field'][ob])
    independent=np.random.default_rng(core.seed('Extension','E2','fixture','independent')).integers(0,100000,size=2)
    assert independent[0]!=independent[1]
    return dict(seed=core.seed('Extension','E2','fixture'),selected=d['selected_ids'],excluded=d['excluded_ids'],shared_coordinate=shared,shared_value=d['field'][shared],independent_distinct_draws=independent.tolist(),passed=True)

def run(max_units=None,shard=0,shards=1):
    check();curves,residual,meta=calibration();atomic(OUT/'noise_design_checks.json',fixture())
    begin=time.time();done=0;alljobs=[(t,n,d,r) for t in TRUTHS for n in NS for d in DESIGNS for r in range(200)]
    assert 0<=shard<shards
    for index,key in enumerate(alljobs):
        if index%shards!=shard:continue
        p=unit_path(*key)
        if p.exists():
            try:
                old=json.loads(p.read_text());assert old['unit_id']=='/'.join(map(str,key)) and old['code_hash']==sha(__file__) and set(old['starts'])=={'1','5'}
                continue
            except Exception: raise RuntimeError(f'corrupt existing unit: {p}')
        if max_units is not None and done>=max_units:break
        if time.time()-begin>=7*3600:break
        started=time.time()
        try: result=job(*key,curves,residual,meta);result['status']='complete'
        except Exception as exc:result=dict(unit_id='/'.join(map(str,key)),status='error',error=repr(exc),code_hash=sha(__file__),spec_hash=sha(SPEC))
        result['elapsed_seconds']=time.time()-started;atomic(p,result);done+=1
        if done%20==0: checkpoint(begin,alljobs)
    checkpoint(begin,alljobs)

def checkpoint(begin,alljobs):
    completed=[];errors=[]
    for key in alljobs:
        p=unit_path(*key)
        if p.exists():
            v=json.loads(p.read_text());(completed if v['status']=='complete' else errors).append(v['unit_id'])
    atomic(OUT/'checkpoint.json',dict(completed_units=completed,error_units=errors,remaining_units=len(alljobs)-len(completed)-len(errors),elapsed_seconds=time.time()-begin,spec_hash=sha(SPEC),input_manifest_hash=sha(OUT/'input_manifest.json'),code_hash=sha(__file__),seed=core.SEED,updated=time.time()))

def wilson(k,n=200):
    z=1.959963984540054;p=k/n;den=1+z*z/n
    mid=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return mid-half,mid+half
def csvwrite(name,rows):
    if not rows:return
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def summarize():
    check();counts=defaultdict(Counter);completed=Counter();errors=[];runtime=[];boundary=Counter();candidates=Counter();selected=Counter();inclusions=defaultdict(list);draw_stats=[]
    for truth in TRUTHS:
      for N in NS:
       for design in DESIGNS:
        for rep in range(200):
         p=unit_path(truth,N,design,rep)
         if not p.exists():continue
         v=json.loads(p.read_text())
         if v['status']!='complete':errors.append(v['unit_id']);continue
         for s in STARTS:counts[(truth,N,design,s)][v['starts'][str(s)]['winner']]+=1;completed[(truth,N,design,s)]+=1
         runtime.append(dict(truth=truth,N=N,design=design,rep=rep,elapsed_seconds=v['elapsed_seconds']))
         draw_stats.append(dict(truth=truth,N=N,design=design,rep=rep,component_draw_count=v['component_draw_count'],distinct_source_component_count=v['distinct_source_component_count']))
         for b in v['boundaries']:
            assert b['k']==len(b['selected_ids'])==len(set(b['selected_ids'])) and set(b['selected_ids'])<=set(b['candidate_ids'])
            boundary[(truth,N,design)]+=1
            for eid in b['candidate_ids']:candidates[(truth,N,design,eid)]+=1;inclusions[(truth,N,design,eid)].append(b['inclusion_probability'])
            for eid in b['selected_ids']:selected[(truth,N,design,eid)]+=1
    confusion=[];recognition=[];components=[];miss=[]
    for design in DESIGNS:
     for N in NS:
      for s in STARTS:
       for truth in TRUTHS:
        key=(truth,N,design,s);n=completed[key];c=counts[key]
        for winner in (*TRUTHS,'unclassified_fit_failure'):
            confusion.append(dict(truth=truth,N=N,design=design,start=s,winner=winner,count=c[winner],completed=n,planned=200,pending=200-n))
        ok=c[truth];lo,hi=wilson(ok) if n==200 else (None,None)
        conditional_n=sum(c[f] for f in TRUTHS)
        recognition.append(dict(truth=truth,N=N,design=design,start=s,correct=ok,rate=ok/200 if n==200 else None,wilson_lower=lo,wilson_upper=hi,completed=n,planned=200,unclassified=c['unclassified_fit_failure'],conditional_denominator=conditional_n,conditional_rate=ok/conditional_n if conditional_n else None,primary=design=='calendar_joint' and s==5))
        if truth=='PL':num=200-c['PL'] if n==200 else None;label='PL_miss'
        else:num=c['PL'];label=truth+'_false_PL'
        a,b=wilson(num) if num is not None else (None,None)
        row=dict(design=design,N=N,start=s,estimand=label,truth=truth,count=num,rate=num/200 if num is not None else None,wilson_lower=a,wilson_upper=b,completed=n,planned=200,unclassified=c['unclassified_fit_failure'])
        miss.append(row)
        if truth=='PL':components.append({**row,'estimand':'PL_recognition','count':c['PL'] if n==200 else None,'rate':c['PL']/200 if n==200 else None,'wilson_lower':lo,'wilson_upper':hi})
        else:components.append(row)
    n80=[]
    for design in DESIGNS:
        qualifying=[];complete=[]
        for N in NS:
            rows=[r for r in components if r['design']==design and r['N']==N]
            full=all(r['completed']==200 for r in rows)
            good=full and all((r['count']>=160 if r['estimand']=='PL_recognition' else r['count']<=20) for r in rows)
            qualifying.append(good);complete.append(full)
        hit=next((N for i,N in enumerate(NS) if qualifying[i] and all(complete[:i+1])),None)
        state=str(hit) if hit is not None else ('not attained on tested grid through 320' if all(complete) else 'pending')
        n80.append(dict(design=design,N80=state,complete_cells=sum(complete),qualifying_N=';'.join(str(NS[i]) for i,x in enumerate(qualifying) if x)))
    csvwrite('confusion.csv',confusion);csvwrite('recognition.csv',recognition);csvwrite('N80_components.csv',components);csvwrite('PL_errors.csv',miss);csvwrite('N80.csv',n80);csvwrite('runtime.csv',runtime);csvwrite('component_draws.csv',draw_stats)
    csvwrite('boundary_frequencies.csv',[dict(truth=t,N=n,design=d,event_id=e,candidate_count=v,selected_count=selected[(t,n,d,e)],selected_frequency=selected[(t,n,d,e)]/v,mean_conditional_inclusion=float(np.mean(inclusions[(t,n,d,e)]))) for (t,n,d,e),v in sorted(candidates.items())])
    atomic(OUT/'summary_diagnostics.json',dict(completed=sum(1 for x in runtime),errors=errors,boundary_counts={str(k):v for k,v in boundary.items()},N80=n80))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['prepare','run','summarize','fixture'],required=True);ap.add_argument('--max-units',type=int);ap.add_argument('--shard',type=int,default=0);ap.add_argument('--shards',type=int,default=1);a=ap.parse_args()
    if a.mode=='prepare':print(json.dumps(prepare(),indent=2))
    elif a.mode=='run':run(a.max_units,a.shard,a.shards)
    elif a.mode=='summarize':summarize()
    else:print(json.dumps(fixture(),indent=2))
