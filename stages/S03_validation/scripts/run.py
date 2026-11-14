"""S03 frozen held-out validation. Run from project root: python3 stages/S03_validation/scripts/run.py."""
import csv, hashlib, json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'stages/S03_validation'
sys.path.insert(0,str(ROOT/'stages/S01_protocol/attempt_02'))
sys.path.insert(0,str(ROOT/'stages/S02a_pooled/scripts'))
import protocol_core as core
from pipeline import inputs, MATS
SEED=5902026
DESIGNS={'peak_primary':('argmax',5,24,25,44),'peak_sensitivity':('argmax',1,20,21,40),'onset5_primary':('onset5',5,24,25,44),'onset5_sensitivity':('onset5',1,20,21,40)}
FAMILIES=tuple(core.BOUNDS)
BENCH=('RW','AR1','HAR')

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,obj): Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def write_csv(path,rows,columns):
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');w.writeheader();w.writerows(rows)

def input_check():
    frozen=json.loads((ROOT/'stages/S02a_pooled/frozen_manifest.json').read_text())
    paths=['stages/S01_protocol/spec_v1.md','stages/S01_protocol/validation_protocol.md','stages/S01_protocol/attempt_02/protocol_core.py','stages/S02a_pooled/spec_v1.md','stages/S02a_pooled/scripts/pipeline.py','omori_pipeline/data/vix_term_structure.csv','omori_pipeline/data/shock_events.csv']
    checks={p:{'actual':sha(ROOT/p),'expected':frozen.get(p,frozen.get('stages/S02a_pooled/reproduction_root/'+p))} for p in paths}
    if any(v['actual']!=v['expected'] for v in checks.values()): raise ValueError('frozen prerequisite mismatch')
    return checks

def event_panel(ts,ev,e,mat,design):
    anchor_kind,tr0,tr1,te0,te1=DESIGNS[design]
    s=ts[mat]; dates=s.index; on=int(dates.searchsorted(e.onset_date))
    a=int(dates.searchsorted(e.peak_date)) if anchor_kind=='argmax' else on+5
    base=dict(event_id=e.event_id,maturity=mat,design=design,onset=str(e.onset_date.date()),anchor_kind=anchor_kind,
              train_start=tr0,train_end=tr1,test_start=te0,test_end=te1)
    if on>=len(s) or dates[on]!=e.onset_date: return base,None,'onset_not_observed'
    if a>=len(s) or (anchor_kind=='argmax' and dates[a]!=e.peak_date): return base,None,'anchor_not_observed'
    if a+te1>=len(s): return base,None,'end_of_data'
    next_on=ev.onset_date[ev.onset_date>e.onset_date].min()
    if pd.notna(next_on) and dates[a+te1]>=next_on: return base,None,'next_onset'
    pre=s.iloc[:on].dropna().iloc[-250:]
    if len(pre)<250: return base,None,'prehistory_250'
    vals=s.iloc[a:a+te1+1].to_numpy(float)
    try: z=core.normalize(pre.to_numpy(float),float(s.iloc[a]),vals)
    except ValueError as ex: return base,None,'normalization_'+str(ex)
    b=dict(base,anchor=str(dates[a].date()),pre_first=str(pre.index[0].date()),pre_last=str(pre.index[-1].date()),
           train_first=str(dates[a+tr0].date()),train_last=str(dates[a+tr1].date()),
           test_first=str(dates[a+te0].date()),test_last=str(dates[a+te1].date()),
           mu=float(np.log(pre).mean()),den=float(np.log(s.iloc[a])-np.log(pre).mean()),pre_n=len(pre),
           next_onset=str(next_on.date()) if pd.notna(next_on) else '')
    return b,(s,dates,on,a,pre,vals,z),None

def benchmark(pre,history,horizon,family):
    logs=np.log(pre.to_numpy(float)); hist=np.log(history.to_numpy(float))
    if len(logs)<756 or not np.isfinite(logs).all() or not np.isfinite(hist).all(): raise ValueError('prehistory_756_or_missing')
    rows=len(logs)-(1 if family=='AR1' else 22 if family=='HAR' else 0)
    if family!='RW' and rows<252: raise ValueError('regression_rows_below_252')
    pred=core.forecast_benchmark(logs,hist,horizon,family)
    if family=='RW': coef={'last_log':float(hist[-1])}
    elif family=='AR1':
        x=np.column_stack([np.ones(len(logs)-1),logs[:-1]]); coef=dict(zip(['intercept','lag1'],np.linalg.lstsq(x,logs[1:],rcond=None)[0].tolist()))
    else:
        x=np.array([[1,logs[i-1],np.mean(logs[i-5:i]),np.mean(logs[i-22:i])] for i in range(22,len(logs))]);
        coef=dict(zip(['intercept','lag1','mean5','mean22'],np.linalg.lstsq(x,logs[22:],rcond=None)[0].tolist()))
    return pred,coef,rows

def oos(ts,ev):
    manifests=[]; preds=[]; coefs=[]; fits=[]; audits=[]
    for design in DESIGNS:
      for _,e in ev.iterrows():
       for mat in MATS:
        b,p,reason=event_panel(ts,ev,e,mat,design)
        if reason: manifests.append(dict(b,eligible=False,reason=reason));continue
        s,dates,on,a,pre,vals,z=p
        _,tr0,tr1,te0,te1=DESIGNS[design]
        history=s.iloc[:a+tr1+1]
        # Exact last 756 observed pre-onset sessions; missing maturity history is an explicit failure.
        pre756=s.iloc[:on].dropna().iloc[-756:]
        fp_first=pre756.index[0] if len(pre756)==756 else pre.index[0]
        group_fp=(int(fp_first.value),int(dates[a+te1].value))
        b.update(footprint_first=str(fp_first.date()),footprint_last=str(dates[a+te1].date()))
        fit_ok=0
        for family in FAMILIES:
            key=['S03',design,e.event_id,mat,tr0,tr1]
            f=core.fit(family,np.arange(tr0,tr1+1),z[tr0:tr1+1],key)
            best=f['best']
            fits.append(dict(event_id=e.event_id,maturity=mat,design=design,model=family,success=best is not None,
                             parameters=json.dumps(best['parameters']) if best else '',sse=best['sse'] if best else '',
                             successful_starts=sum(x['success'] for x in f['starts']),boundary_starts=sum(any(x['active_mask']) for x in f['starts'])))
            if best is None: audits.append(dict(event_id=e.event_id,maturity=mat,design=design,model=family,status='fit_failure',detail='no finite converged start'));continue
            fit_ok+=1; zz=core.curve(family,np.arange(te0,te1+1),best['parameters']); logp=b['mu']+b['den']*zz
            for j,t in enumerate(range(te0,te1+1)):
                preds.append(dict(event_id=e.event_id,maturity=mat,design=design,model=family,t=t,date=str(dates[a+t].date()),pred_log=float(logp[j]),actual_log=float(np.log(vals[t])),pred_level=float(np.exp(logp[j])),actual_level=float(vals[t]),group=''))
        for fam in BENCH:
            try:
                lp,coef,rows=benchmark(pre756,history,te1-tr1,fam)
                # Recursion starts at training end; test begins next session for both frozen splits.
                assert te0==tr1+1
                coefs.append(dict(event_id=e.event_id,maturity=mat,design=design,model=fam,cutoff=str(pre756.index[-1].date()) if len(pre756) else '',first=str(pre756.index[0].date()) if len(pre756) else '',pre_n=len(pre756),rows=rows,origin=str(dates[a+tr1].date()),coefficients=json.dumps(coef)))
                for j,t in enumerate(range(te0,te1+1)):
                    preds.append(dict(event_id=e.event_id,maturity=mat,design=design,model=fam,t=t,date=str(dates[a+t].date()),pred_log=float(lp[j]),actual_log=float(np.log(vals[t])),pred_level=float(np.exp(lp[j])),actual_level=float(vals[t]),group=''))
            except ValueError as ex: audits.append(dict(event_id=e.event_id,maturity=mat,design=design,model=fam,status='benchmark_exclusion',detail=str(ex)))
        manifests.append(dict(b,eligible=True,reason='',curve_fits=fit_ok,benchmark_fits=sum(x['event_id']==e.event_id and x['maturity']==mat and x['design']==design for x in coefs),footprint_a=group_fp[0],footprint_b=group_fp[1]))
    # Group on raw footprints across all maturities, within each design.
    for design in DESIGNS:
        rr=[r for r in manifests if r['design']==design and r['eligible']]
        if rr:
            gg=core.overlap_groups([(r['footprint_a'],r['footprint_b']) for r in rr]); lookup={(r['event_id'],r['maturity']):int(g) for r,g in zip(rr,gg)}
            for r in rr:r['group']=lookup[r['event_id'],r['maturity']]
            for r in preds:
                if r['design']==design:r['group']=lookup[r['event_id'],r['maturity']]
    columns=['event_id','maturity','design','model','t','date','pred_log','actual_log','pred_level','actual_level','group']
    write_csv(OUT/'predictions.csv',preds,columns)
    write_csv(OUT/'train_test_manifest.csv',manifests,sorted(set().union(*(r.keys() for r in manifests))))
    write_csv(OUT/'benchmark_coefficients.csv',coefs,['event_id','maturity','design','model','cutoff','first','pre_n','rows','origin','coefficients'])
    write_csv(OUT/'curve_fits.csv',fits,['event_id','maturity','design','model','success','parameters','sse','successful_starts','boundary_starts'])
    return manifests,preds,coefs,fits,audits

def score_oos(manifests,preds):
    d=pd.DataFrame(preds); records=[]; summary={}
    if d.empty:return {},[]
    for (design,mat,model,eid),g in d.groupby(['design','maturity','model','event_id']):
        records.append(dict(design=design,maturity=mat,model=model,event_id=eid,group=int(g.group.iloc[0]),n=len(g),log_mse=float(np.mean((g.pred_log-g.actual_log)**2)),raw_mae=float(np.mean(abs(g.pred_level-g.actual_level)))))
    write_csv(OUT/'event_losses.csv',records,['design','maturity','model','event_id','group','n','log_mse','raw_mae'])
    z=pd.DataFrame(records)
    for (design,mat),g in z.groupby(['design','maturity']):
        complete=g.pivot(index='event_id',columns='model',values='log_mse').dropna()
        key=design+'/'+mat; summary[key]={'paired_events':len(complete),'models':{},'paired_vs_RW':{}}
        for model,h in g.groupby('model'):
            summary[key]['models'][model]=dict(events=len(h),groups=int(h.group.nunique()),log_rmse=float(math.sqrt(h.log_mse.mean())),raw_mae=float(h.raw_mae.mean()))
        if len(complete):
            group_map=g.drop_duplicates('event_id').set_index('event_id').group.to_dict(); groups=np.array([group_map[e] for e in complete.index]); ug=np.unique(groups)
            for model in [m for m in complete.columns if m!='RW']:
                diff=(complete[model]-complete['RW']).to_numpy(float); obj={'mean_log_mse_difference':float(diff.mean()),'groups':len(ug),'group_losses':{str(u):float(diff[groups==u].mean()) for u in ug}}
                if len(ug)>=10:
                    rng=np.random.default_rng(core.seed('S03','oos_bootstrap',design,mat,model)); draws=[]
                    for _ in range(9999):
                        selected=rng.choice(ug,len(ug),replace=True); ix=np.concatenate([np.flatnonzero(groups==u) for u in selected]);draws.append(float(diff[ix].mean()))
                    obj['bootstrap_95pct']=np.quantile(draws,[.025,.975]).tolist();obj['draws']=9999
                else:obj['interval_status']='descriptive_fewer_than_10_groups'
                summary[key]['paired_vs_RW'][model]=obj
    return summary,records

def predictor(ts,ev):
    s=ts.VIX30D; dates=s.index; features=[]; exclusions=[]
    for _,e in ev.iterrows():
        on=int(dates.searchsorted(e.onset_date));peak=int(dates.searchsorted(e.peak_date));reason=''
        if on>=len(s) or peak>=len(s) or dates[on]!=e.onset_date or dates[peak]!=e.peak_date:reason='unobserved_onset_or_peak'
        elif peak+20>=len(s):reason='end_of_data'
        else:
            next_on=ev.onset_date[ev.onset_date>e.onset_date].min()
            if pd.notna(next_on) and dates[peak+20]>=next_on:reason='next_onset_within_target'
        pre=s.iloc[:on].dropna().iloc[-20:] if not reason else pd.Series(dtype=float)
        if not reason and len(pre)!=20:reason='prehistory_20'
        if not reason:
            post=s.iloc[peak+1:peak+21].to_numpy(float); vp=float(s.iloc[peak]); b20=float(pre.mean())
            if not np.isfinite(post).all() or (post<=0).any() or not np.isfinite(vp) or vp<=b20 or b20<=0:reason='nonpositive_denominator_or_missing'
        if reason:exclusions.append(dict(event_id=e.event_id,reason=reason));continue
        features.append(dict(event_id=e.event_id,onset=str(e.onset_date.date()),peak=str(e.peak_date.date()),feature_first=str(pre.index[0].date()),feature_last=str(pre.index[-1].date()),target_first=str(dates[peak+1].date()),target_last=str(dates[peak+20].date()),b20=b20,vpeak=vp,shock_size=float(np.log(vp/b20)),build_up=core.buildup(pre.to_numpy(float)),target=float(np.mean((post-b20)/(vp-b20))),footprint_a=int(pre.index[0].value),footprint_b=int(dates[peak+20].value)))
    if features:
        gr=core.overlap_groups([(r['footprint_a'],r['footprint_b']) for r in features]);
        for r,g in zip(features,gr):r['group']=int(g)
    write_csv(OUT/'build_up_features.csv',features,['event_id','onset','peak','feature_first','feature_last','target_first','target_last','b20','vpeak','shock_size','build_up','target','footprint_a','footprint_b','group'])
    write_csv(OUT/'predictor_exclusions.csv',exclusions,['event_id','reason'])
    n=len(features); groups=len(set(r['group'] for r in features)); folds=[]; pp=[]
    eligible=n>=20 and groups>=10 and all(n-sum(x['group']==g for x in features)>=15 for g in set(r['group'] for r in features))
    if eligible:
        x=np.array([[r['shock_size'],r['build_up']] for r in features]); y=np.array([r['target'] for r in features]); gg=np.array([r['group'] for r in features])
        for group in np.unique(gg):
            train=gg!=group; test=~train
            folds.append(dict(group=int(group),train_events=int(train.sum()),test_events=int(test.sum()),train_ids='|'.join(np.array([r['event_id'] for r in features])[train]),test_ids='|'.join(np.array([r['event_id'] for r in features])[test])))
            for name,cols in [('size_only',[0]),('size_plus_build_up',[0,1])]:
                yh=core.ridge_predict(x[train][:,cols],y[train],x[test][:,cols]);
                for i,p in zip(np.flatnonzero(test),yh):pp.append(dict(event_id=features[i]['event_id'],group=int(group),model=name,prediction=float(p),actual=float(y[i]),abs_error=float(abs(p-y[i]))))
    write_csv(OUT/'logo_folds.csv',folds,['group','train_events','test_events','train_ids','test_ids'])
    write_csv(OUT/'predictor_predictions.csv',pp,['event_id','group','model','prediction','actual','abs_error'])
    result=dict(status='estimable' if eligible else 'prespecified_non_estimable',complete_events=n,groups=groups,min_fold_train=min((n-sum(x['group']==g for x in features) for g in set(r['group'] for r in features)),default=0),exclusions=exclusions,seed=SEED,lambda_ridge=1,bootstrap_draws=9999 if eligible else 0)
    if eligible:
        q=pd.DataFrame(pp); piv=q.pivot(index='event_id',columns='model',values='abs_error');result['paired_events']=len(piv);result['mae']={m:float(q[q.model==m].abs_error.mean()) for m in ['size_only','size_plus_build_up']}
        gmap={r['event_id']:r['group'] for r in features};grp=np.array([gmap[e] for e in piv.index]);ug=np.unique(grp); dif=(piv['size_plus_build_up']-piv['size_only']).to_numpy();result['mae_difference']=float(dif.mean());result['group_differences']={str(g):float(dif[grp==g].mean()) for g in ug}
        rng=np.random.default_rng(SEED);draws=[]
        for _ in range(9999):
            selected=rng.choice(ug,len(ug),replace=True);ix=np.concatenate([np.flatnonzero(grp==g) for g in selected]);draws.append(float(dif[ix].mean()))
        result['bootstrap_95pct']=np.quantile(draws,[.025,.975]).tolist()
    write_json(OUT/'predictor_results.json',result)
    return result,features,folds,pp

def main():
    checks=input_check();ts,ev=inputs(); assert len(ev)==33
    manifests,preds,coefs,fits,audits=oos(ts,ev)
    summary,losses=score_oos(manifests,preds)
    pr,features,folds,pp=predictor(ts,ev)
    for r in manifests:
        audits.append(dict(event_id=r['event_id'],maturity=r['maturity'],design=r['design'],model='cohort',status='eligible' if r['eligible'] else 'excluded',detail=r['reason'],onset=r['onset'],pre_last=r.get('pre_last',''),train_last=r.get('train_last',''),test_first=r.get('test_first',''),group=r.get('group','')))
    for c in coefs:
        audits.append(dict(event_id=c['event_id'],maturity=c['maturity'],design=c['design'],model=c['model'],status='pre_onset_cutoff',detail='rows='+str(c['rows']),onset=next(r['onset'] for r in manifests if r['event_id']==c['event_id'] and r['maturity']==c['maturity'] and r['design']==c['design']),pre_last=c['cutoff'],train_last=c['origin'],test_first='',group=''))
    write_csv(OUT/'leakage_audit.csv',audits,['event_id','maturity','design','model','status','detail','onset','pre_last','train_last','test_first','group'])
    metrics=dict(prerequisite_hashes=checks,seed=SEED,event_catalog_n=len(ev),oos=summary,counts=dict(manifest_rows=len(manifests),eligible_panels=sum(r['eligible'] for r in manifests),prediction_rows=len(preds),benchmark_coefficient_rows=len(coefs),curve_fit_rows=len(fits),event_loss_rows=len(losses)),predictor_status=pr['status'])
    write_json(OUT/'metrics.json',metrics)
    files=[p for p in OUT.iterdir() if p.is_file() and p.name not in ('output_manifest.json',)]
    write_json(OUT/'output_manifest.json',{'inputs':checks,'script_sha256':sha(__file__),'outputs':{p.name:sha(p) for p in files}})
    print(json.dumps(metrics['counts']|{'predictor':pr['status'],'predictor_events':pr['complete_events'],'predictor_groups':pr['groups']}))

if __name__=='__main__':main()
