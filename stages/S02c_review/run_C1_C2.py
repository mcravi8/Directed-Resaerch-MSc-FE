"""Frozen S02c C1/C2 simulations, resumable per replication; never invokes OU pilot."""
import argparse, json, math, sys, time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
import numpy as np
from scipy.stats import norm
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'stages/S02c_review'
sys.path.insert(0,str(ROOT/'stages/S01_protocol/attempt_02'))
import protocol_core as core
sys.path.insert(0,str(ROOT/'stages/S02a_pooled/scripts'))
import pipeline
FAMILIES=['E1','E2','PL','SE']

def source():
    ts,ev=pipeline.inputs();panel,_,_=pipeline.panels(ts,ev,'VIX30D',120)
    ids=sorted(panel);assert len(ids)==19
    z=np.array([panel[i] for i in ids]); y=z.mean(axis=0)
    summaries=json.loads((ROOT/'stages/S02a_pooled/pooled_summary.json').read_text())['summaries']
    def get(start):return next(x for x in summaries if x['unit']=='empirical_120_VIX30D_full_event' and x['start']==start and not x['zero_baseline'])
    return ids,z,y,get(1),get(5)

def truths(y,s1,s5):
    t=np.arange(130);b=s5['models']['E1']['parameters'][0];target=y[20]
    e2=s1['models']['E2']['parameters'];w,t1,t2=e2[2:]
    v={'E1':b+(1-b)*np.exp(-t/27.8),
       'E2':b+(1-b)*(w*np.exp(-t/t1)+(1-w)*np.exp(-t/t2))}
    ratio=(target-b)/(1-b)
    assert 0<ratio<1
    scale=20/(-math.log(ratio))**(1/.6)
    c=20/(ratio**(-1/.7)-1)
    v['SE']=b+(1-b)*np.exp(-(t/scale)**.6)
    v['PL']=b+(1-b)*(1+t/c)**(-.7)
    return v,dict(baseline=b,empirical_z20=target,SE_scale=scale,PL_c=c,E2_weight=w,E2_fast=t1,E2_slow=t2)

def fit(y,start,key,families=FAMILIES):
    models={}
    for f in families:
        result=core.fit(f,np.arange(start,121),y[start:121],['S02c',*key,start],False)
        best=result['best'];models[f]=None if best is None else dict(score=core.score(best['sse'],120-start+1,result['k']),parameters=best['parameters'])
    valid={f:m for f,m in models.items() if m is not None}
    winner=min(valid,key=lambda f:valid[f]['score']) if len(valid)==len(families) else None
    return winner,models

def residuals(z,s1):
    r=z-core.curve('E2',np.arange(121),s1['models']['E2']['parameters'])
    return r-r.mean(axis=0)

def wilson(k,n):
    zz=norm.ppf(.975);p=k/n;den=1+zz**2/n
    return [(p+zz**2/(2*n)-zz*np.sqrt(p*(1-p)/n+zz**2/(4*n*n)))/den,
            (p+zz**2/(2*n)+zz*np.sqrt(p*(1-p)/n+zz**2/(4*n*n)))/den]

def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():return
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def c1(rep,truth,curves,bank):
    rng=np.random.default_rng(core.seed('S02c','C1',truth,rep))
    panel=[]
    for _ in range(19):
        source_event=rng.integers(0,19);r=bank[source_event];blocks=[]
        while len(blocks)<121:
            j=rng.integers(0,121);blocks.extend(r[(j+np.arange(10))%121])
        noise=np.array(blocks[:121]);panel.append(curves[truth][:121]+noise-noise[0])
    panel=np.asarray(panel);out={}
    for start in [1,5]:
        point,models=fit(panel.mean(axis=0),start,[truth,rep,'point'])
        counts=dict.fromkeys(FAMILIES,0);fractions=[];fail=0
        for b in range(199):
            ix=rng.integers(0,19,19)
            win,_=fit(panel[ix].mean(axis=0),start,[truth,rep,'boot',b])
            if win is None:fail+=1
            else:counts[win]+=1
        out[str(start)]=dict(point_winner=point,point_models=models,bootstrap_wins=counts,bootstrap_failures=fail,point_winner_bootstrap_fraction=counts[point]/199 if point else None)
    return dict(truth=truth,replication=rep,starts=out)

def c1_job(job,curves,bank):
    truth,rep=job
    return job,c1(rep,truth,curves,bank)

def c2(rep,curves,phi,sd,b):
    rng=np.random.default_rng(core.seed('S02c','C2',rep));noise=np.empty(130)
    noise[0]=rng.normal(0,sd/math.sqrt(1-phi*phi))
    for i in range(1,130):noise[i]=phi*noise[i-1]+rng.normal(0,sd)
    path=curves['E1']+noise;peak=int(np.argmax(path[:10]));aligned=path[peak:peak+121]
    y=b+(aligned-b)/(aligned[0]-b)*(1-b)
    winner,models=fit(y,1,['C2',rep],['E1','E2'])
    return dict(replication=rep,peak_offset=peak,winner=winner,fast_tau=models['E2']['parameters'][3] if models['E2'] else None,models=models)

def summarize():
    rows=[]
    for truth in FAMILIES:
        reps=[json.loads((OUT/'checkpoints'/'C1'/truth/f'{i:03d}.json').read_text()) for i in range(200)]
        for start in ['1','5']:
            wins=[r['starts'][start]['point_winner'] for r in reps]
            for family in FAMILIES:
                k=wins.count(family);rows.append(dict(truth=truth,start=int(start),point_winner=family,n=k,total=200,share=k/200,wilson_lower=wilson(k,200)[0],wilson_upper=wilson(k,200)[1]))
            frac=[r['starts'][start]['point_winner_bootstrap_fraction'] for r in reps if r['starts'][start]['point_winner']]
            q=np.quantile(frac,[.025,.5,.975]);rows.append(dict(truth=truth,start=int(start),point_winner='WIN_FRACTION_Q025_Q50_Q975',n=len(frac),total=200,share=float(q[1]),wilson_lower=float(q[0]),wilson_upper=float(q[2])))
    import pandas as pd
    pd.DataFrame(rows).to_csv(OUT/'C1_confusion_and_win_fraction.csv',index=False)
    e1=[json.loads((OUT/'checkpoints'/'C1'/'E1'/f'{i:03d}.json').read_text()) for i in range(200)]
    bias=[]
    for start in ['1','5']:
        vals=np.array([r['starts'][start]['point_models']['E1']['parameters'][2] for r in e1]);bias.append(dict(start=int(start),mean_tau=float(vals.mean()),bias=float(vals.mean()-27.8),median_tau=float(np.median(vals)),q025=float(np.quantile(vals,.025)),q975=float(np.quantile(vals,.975))))
    pd.DataFrame(bias).to_csv(OUT/'C1_E1_tau_bias.csv',index=False)
    r=[json.loads((OUT/'checkpoints'/'C2'/f'{i:03d}.json').read_text()) for i in range(200)]
    k=sum(x['winner']=='E2' for x in r);fast=np.array([x['fast_tau'] for x in r if x['fast_tau'] is not None]);winfast=np.array([x['fast_tau'] for x in r if x['winner']=='E2'])
    save(OUT/'C2_summary.json',dict(replicates=200,E2_wins=k,E2_share=k/200,wilson_95=wilson(k,200),peak_offsets=np.bincount([x['peak_offset'] for x in r],minlength=10).tolist(),fast_tau_quantiles=np.quantile(fast,[.025,.5,.975]).tolist(),winner_fast_tau_quantiles=np.quantile(winfast,[.025,.5,.975]).tolist() if len(winfast) else None))

def summarize_c2():
    r=[json.loads((OUT/'checkpoints'/'C2'/f'{i:03d}.json').read_text()) for i in range(200)]
    k=sum(x['winner']=='E2' for x in r);fast=np.array([x['fast_tau'] for x in r if x['fast_tau'] is not None]);winfast=np.array([x['fast_tau'] for x in r if x['winner']=='E2'])
    save(OUT/'C2_summary.json',dict(replicates=200,E2_wins=k,E2_share=k/200,wilson_95=wilson(k,200),peak_offsets=np.bincount([x['peak_offset'] for x in r],minlength=10).tolist(),fast_tau_quantiles=np.quantile(fast,[.025,.5,.975]).tolist(),winner_fast_tau_quantiles=np.quantile(winfast,[.025,.5,.975]).tolist() if len(winfast) else None))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('arm',choices=['C1','C2','summarize','summarize-c2']);ap.add_argument('--max-reps',type=int,default=800);ap.add_argument('--workers',type=int,default=1);ap.add_argument('--shard',metavar='I/N',help='C1 only: one-based shard of the full 800-job list');a=ap.parse_args()
    shard=None
    if a.shard is not None:
        if a.arm!='C1':ap.error('--shard applies only to C1')
        try:
            i,n=map(int,a.shard.split('/'))
        except (ValueError,TypeError):
            ap.error('--shard must have the form I/N')
        if not 1<=i<=n:ap.error('--shard requires 1 <= I <= N')
        shard=(i,n)
    if a.arm=='summarize':summarize();return
    if a.arm=='summarize-c2':summarize_c2();return
    started=time.monotonic();ids,z,y,s1,s5=source();curves,parameters=truths(y,s1,s5);bank=residuals(z,s1)
    x=bank[:,:-1].ravel();yy=bank[:,1:].ravel();phi=float(x@yy/(x@x));sd=float(np.std(yy-phi*x,ddof=1))
    save(OUT/'simulation_parameters.json',dict(cohort_ids=ids,truth_parameters=parameters,C2_AR1_phi=phi,C2_innovation_sd=sd,seed=core.SEED))
    jobs=[(truth,rep) for truth in FAMILIES for rep in range(200)] if a.arm=='C1' else [(None,rep) for rep in range(200)]
    if shard is not None:
        i,n=shard
        jobs=[job for position,job in enumerate(jobs) if position%n==i-1]
    done=0
    if a.arm=='C1' and a.workers>1:
        pending=[job for job in jobs if not (OUT/'checkpoints'/'C1'/job[0]/f'{job[1]:03d}.json').exists()]
        with ProcessPoolExecutor(max_workers=a.workers) as executor:
            active={};cursor=0
            while active or cursor<len(pending):
                while cursor<len(pending) and len(active)<a.workers and done+len(active)<a.max_reps and time.monotonic()-started<7*3600+30*60:
                    job=pending[cursor];cursor+=1
                    active[executor.submit(c1_job,job,curves,bank)]=job
                if not active:break
                finished,_=wait(active,return_when=FIRST_COMPLETED)
                for future in finished:
                    truth,rep=active.pop(future);_,result=future.result()
                    save(OUT/'checkpoints'/'C1'/truth/f'{rep:03d}.json',result);done+=1
                    print(json.dumps(dict(arm='C1',truth=truth,rep=rep,completed_this_run=done)),flush=True)
        save(OUT/f'C1_run_checkpoint_{int(time.time())}.json',dict(arm='C1',completed_this_run=done,wall_seconds=time.monotonic()-started,remaining=sum(not (OUT/'checkpoints'/'C1'/truth/f'{rep:03d}.json').exists() for truth,rep in jobs)))
        return
    for truth,rep in jobs:
        path=OUT/'checkpoints'/a.arm/(truth or '')/f'{rep:03d}.json'
        if path.exists():continue
        if time.monotonic()-started>=7*3600+45*60 or done>=a.max_reps:break
        t0=time.monotonic(); result=c1(rep,truth,curves,bank) if truth else c2(rep,curves,phi,sd,parameters['baseline'])
        save(path,result);done+=1
        print(json.dumps(dict(arm=a.arm,truth=truth,rep=rep,seconds=round(time.monotonic()-t0,2))),flush=True)
    save(OUT/f'{a.arm}_run_checkpoint_{int(time.time())}.json',dict(arm=a.arm,completed_this_run=done,wall_seconds=time.monotonic()-started,remaining=sum(not (OUT/'checkpoints'/a.arm/(truth or '')/f'{rep:03d}.json').exists() for truth,rep in jobs)))

if __name__=='__main__':main()
