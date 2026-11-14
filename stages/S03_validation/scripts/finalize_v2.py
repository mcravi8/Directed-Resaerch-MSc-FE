"""Crash-only S03 continuation: reuse attempt_01 curve predictions, repair benchmark state and scoring."""
import json, math
from pathlib import Path
import numpy as np
import pandas as pd
import run

ROOT=run.ROOT;OUT=run.OUT

def repaired_benchmarks(ts,ev,manifests,preds,coefs,audits):
    existing={(x['event_id'],x['maturity'],x['design'],x['model']) for x in coefs}
    for b in manifests:
        if not b['eligible']:continue
        e=ev.loc[ev.event_id==b['event_id']].iloc[0]
        design=b['design'];mat=b['maturity'];_,tr0,tr1,te0,te1=run.DESIGNS[design]
        s=ts[mat];dates=s.index;on=int(dates.searchsorted(e.onset_date))
        a=int(dates.searchsorted(e.peak_date)) if b['anchor_kind']=='argmax' else on+5
        pre756=s.iloc[:on].dropna().iloc[-756:]
        # Only lag state through forecast origin is needed; earlier missing coverage must not exclude the benchmark.
        history=s.iloc[:a+tr1+1].dropna().iloc[-22:]
        for fam in run.BENCH:
            if (e.event_id,mat,design,fam) in existing:continue
            try:
                lp,coef,rows=run.benchmark(pre756,history,te1-tr1,fam)
                coefs.append(dict(event_id=e.event_id,maturity=mat,design=design,model=fam,cutoff=str(pre756.index[-1].date()) if len(pre756) else '',first=str(pre756.index[0].date()) if len(pre756) else '',pre_n=len(pre756),rows=rows,origin=str(dates[a+tr1].date()),coefficients=json.dumps(coef)))
                for j,t in enumerate(range(te0,te1+1)):
                    preds.append(dict(event_id=e.event_id,maturity=mat,design=design,model=fam,t=t,date=str(dates[a+t].date()),pred_log=float(lp[j]),actual_log=float(np.log(s.iloc[a+t])),pred_level=float(np.exp(lp[j])),actual_level=float(s.iloc[a+t]),group=b['group']))
            except ValueError as ex:audits.append(dict(event_id=e.event_id,maturity=mat,design=design,model=fam,status='benchmark_exclusion',detail=str(ex)))
    for b in manifests:
        if b['eligible']:b['benchmark_fits']=sum(x['event_id']==b['event_id'] and x['maturity']==b['maturity'] and x['design']==b['design'] for x in coefs)
    return manifests,preds,coefs,audits

def score(manifests,preds):
    d=pd.DataFrame(preds); records=[];summary={}
    for (design,mat,model,eid),g in d.groupby(['design','maturity','model','event_id']):
        records.append(dict(design=design,maturity=mat,model=model,event_id=eid,group=int(g.group.iloc[0]),n=len(g),log_mse=float(np.mean((g.pred_log-g.actual_log)**2)),raw_mae=float(np.mean(abs(g.pred_level-g.actual_level)))))
    run.write_csv(OUT/'event_losses.csv',records,['design','maturity','model','event_id','group','n','log_mse','raw_mae'])
    z=pd.DataFrame(records)
    for (design,mat),g in z.groupby(['design','maturity']):
        key=design+'/'+mat; summary[key]={'models':{},'paired_vs_RW':{}}
        for model,h in g.groupby('model'):
            summary[key]['models'][model]=dict(events=len(h),groups=int(h.group.nunique()),log_rmse=float(math.sqrt(h.log_mse.mean())),raw_mae=float(h.raw_mae.mean()))
        complete=g.pivot(index='event_id',columns='model',values='log_mse').dropna()
        summary[key]['paired_events']=len(complete)
        if 'RW' not in complete.columns or complete.empty:
            summary[key]['paired_status']='no_complete_RW_overlap';continue
        group_map=g.drop_duplicates('event_id').set_index('event_id').group.to_dict();groups=np.array([group_map[e] for e in complete.index]);ug=np.unique(groups)
        for model in [x for x in complete.columns if x!='RW']:
            diff=(complete[model]-complete['RW']).to_numpy(float)
            obj=dict(mean_log_mse_difference=float(diff.mean()),groups=len(ug),group_losses={str(u):float(diff[groups==u].mean()) for u in ug})
            if len(ug)>=10:
                rng=np.random.default_rng(run.core.seed('S03','oos_bootstrap',design,mat,model));draws=[]
                for _ in range(9999):
                    selected=rng.choice(ug,len(ug),replace=True);ix=np.concatenate([np.flatnonzero(groups==u) for u in selected]);draws.append(float(diff[ix].mean()))
                obj['bootstrap_95pct']=np.quantile(draws,[.025,.975]).tolist();obj['draws']=9999
            else:obj['interval_status']='descriptive_fewer_than_10_groups'
            summary[key]['paired_vs_RW'][model]=obj
    return summary,records

def main():
    checks=run.input_check();ts,ev=run.inputs()
    base=OUT/'attempt_01'
    manifests=pd.read_csv(base/'train_test_manifest.csv').to_dict('records')
    preds=pd.read_csv(base/'predictions.csv').to_dict('records')
    coefs=pd.read_csv(base/'benchmark_coefficients.csv').to_dict('records')
    fits=pd.read_csv(base/'curve_fits.csv')
    audits=[]
    manifests,preds,coefs,audits=repaired_benchmarks(ts,ev,manifests,preds,coefs,audits)
    run.write_csv(OUT/'predictions.csv',preds,['event_id','maturity','design','model','t','date','pred_log','actual_log','pred_level','actual_level','group'])
    run.write_csv(OUT/'train_test_manifest.csv',manifests,sorted(set().union(*(r.keys() for r in manifests))))
    run.write_csv(OUT/'benchmark_coefficients.csv',coefs,['event_id','maturity','design','model','cutoff','first','pre_n','rows','origin','coefficients'])
    summary,losses=score(manifests,preds)
    pr,features,folds,pp=run.predictor(ts,ev)
    for r in manifests:
        audits.append(dict(event_id=r['event_id'],maturity=r['maturity'],design=r['design'],model='cohort',status='eligible' if r['eligible'] else 'excluded',detail=r['reason'],onset=r['onset'],pre_last=r.get('pre_last',''),train_last=r.get('train_last',''),test_first=r.get('test_first',''),group=r.get('group','')))
    for c in coefs:
        audits.append(dict(event_id=c['event_id'],maturity=c['maturity'],design=c['design'],model=c['model'],status='pre_onset_cutoff',detail='rows='+str(c['rows']),onset=next(r['onset'] for r in manifests if r['event_id']==c['event_id'] and r['maturity']==c['maturity'] and r['design']==c['design']),pre_last=c['cutoff'],train_last=c['origin'],test_first='',group=''))
    run.write_csv(OUT/'leakage_audit.csv',audits,['event_id','maturity','design','model','status','detail','onset','pre_last','train_last','test_first','group'])
    metrics=dict(prerequisite_hashes=checks,seed=run.SEED,event_catalog_n=len(ev),oos=summary,counts=dict(manifest_rows=len(manifests),eligible_panels=sum(r['eligible'] for r in manifests),prediction_rows=len(preds),benchmark_coefficient_rows=len(coefs),curve_fit_rows=len(fits),event_loss_rows=len(losses)),predictor_status=pr['status'])
    run.write_json(OUT/'metrics.json',metrics)
    files=[p for p in OUT.iterdir() if p.is_file() and p.name!='output_manifest.json']
    run.write_json(OUT/'output_manifest.json',dict(inputs=checks,scripts={'frozen_fit':run.sha(OUT/'scripts/run.py'),'continuation':run.sha(__file__)},outputs={p.name:run.sha(p) for p in files}))
    print(json.dumps(metrics['counts']|{'predictor':pr['status'],'predictor_events':pr['complete_events'],'predictor_groups':pr['groups']}))

if __name__=='__main__':main()
