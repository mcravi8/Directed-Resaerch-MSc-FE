"""Independent read-only checks of frozen S03 exports; one local perturbation probe."""
import hashlib, json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]; OUT=ROOT/'stages/S03_validation'
sys.path.insert(0,str(ROOT/'stages/S01_protocol/attempt_02'))
sys.path.insert(0,str(ROOT/'stages/S02a_pooled/scripts'))
import protocol_core as core
from pipeline import inputs

def main():
    f=json.loads((OUT/'freeze_manifest.json').read_text())
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in f['sha256'].items())
    p=pd.read_csv(OUT/'predictions.csv');m=pd.read_csv(OUT/'train_test_manifest.csv');c=pd.read_csv(OUT/'benchmark_coefficients.csv');feat=pd.read_csv(OUT/'build_up_features.csv');fold=pd.read_csv(OUT/'logo_folds.csv');q=pd.read_csv(OUT/'predictor_predictions.csv');met=json.loads((OUT/'metrics.json').read_text());pr=json.loads((OUT/'predictor_results.json').read_text())
    assert len(p)==met['counts']['prediction_rows'] and p[['event_id','maturity','design','model','t']].duplicated().sum()==0
    assert np.allclose(np.log(p.pred_level),p.pred_log,rtol=0,atol=1e-12)
    assert np.allclose(np.log(p.actual_level),p.actual_log,rtol=0,atol=1e-12)
    assert (pd.to_datetime(c.cutoff)<pd.to_datetime(c.merge(m,on=['event_id','maturity','design']).onset)).all()
    assert (c[c.model=='HAR'].rows>=252).all() and (c[c.model=='AR1'].rows>=252).all()
    assert (c.pre_n==756).all()
    assert (pd.to_datetime(m[m.eligible].train_last)<pd.to_datetime(m[m.eligible].test_first)).all()
    # Independent equal-event loss recomputation for every scored model.
    for (design,mat,model),g in p.groupby(['design','maturity','model']):
        e=g.assign(sq=(g.pred_log-g.actual_log)**2,ab=abs(g.pred_level-g.actual_level)).groupby('event_id')[['sq','ab']].mean()
        saved=met['oos'][design+'/'+mat]['models'][model]
        assert math.isclose(math.sqrt(e.sq.mean()),saved['log_rmse'],rel_tol=1e-12)
        assert math.isclose(e.ab.mean(),saved['raw_mae'],rel_tol=1e-12)
    ts,ev=inputs();s=ts.VIX30D
    event_onsets=ev.set_index('event_id').onset_date.to_dict()
    for _,r in c.iterrows():
        logs=np.log(ts.loc[ts.index<event_onsets[r.event_id],r.maturity].dropna().iloc[-756:].to_numpy(float))
        saved_coef=json.loads(r.coefficients)
        if r.model=='RW':
            assert math.isclose(saved_coef['last_log'],np.log(ts.loc[:r.origin,r.maturity].dropna().iloc[-1]),rel_tol=1e-12)
        else:
            lag=1 if r.model=='AR1' else 22
            if lag==1:X=np.column_stack([np.ones(len(logs)-1),logs[:-1]]); names=['intercept','lag1']
            else:
                X=np.array([[1,logs[i-1],sum(logs[i-5:i])/5,sum(logs[i-22:i])/22] for i in range(22,len(logs))]);names=['intercept','lag1','mean5','mean22']
            beta=np.linalg.lstsq(X,logs[lag:],rcond=None)[0]
            assert r.rows==len(logs)-lag
            assert np.allclose(beta,[saved_coef[k] for k in names],rtol=1e-11,atol=1e-12)
    for _,r in feat.iterrows():
        pre=s.loc[r.feature_first:r.feature_last].dropna().to_numpy(float)
        post=s.loc[r.target_first:r.target_last].to_numpy(float)
        assert len(pre)==20 and len(post)==20
        x=np.arange(20);slope=np.polyfit(x,np.log(pre),1)[0]
        assert math.isclose(slope,r.build_up,rel_tol=1e-10,abs_tol=1e-12)
        assert math.isclose(math.log(r.vpeak/pre.mean()),r.shock_size,rel_tol=1e-10)
        assert math.isclose(np.mean((post-pre.mean())/(r.vpeak-pre.mean())),r.target,rel_tol=1e-10)
    if pr['status']=='estimable':
        assert len(q)==2*len(feat) and len(fold)==pr['groups']
        for _,r in fold.iterrows():
            train=set(r.train_ids.split('|'));test=set(r.test_ids.split('|'))
            assert not train&test and len(train)==r.train_events and len(test)==r.test_events
            assert set(feat.loc[feat.group==r.group,'event_id'])==test
        for name in ('size_only','size_plus_build_up'):
            assert math.isclose(q[q.model==name].abs_error.mean(),pr['mae'][name],rel_tol=1e-12)
        for _,fold_row in fold.iterrows():
            train=feat.group!=fold_row.group;test=~train;y=feat.target.to_numpy(float)
            for name,cols in [('size_only',['shock_size']),('size_plus_build_up',['shock_size','build_up'])]:
                X=feat.loc[train,cols].to_numpy(float);Z=feat.loc[test,cols].to_numpy(float)
                mu=X.mean(0);sd=X.std(0);sd[sd==0]=1;T=(X-mu)/sd
                beta=np.linalg.solve(T.T@T+np.eye(len(cols)),T.T@(y[train]-y[train].mean()))
                independent=y[train].mean()+(Z-mu)/sd@beta
                saved=q[(q.group==fold_row.group)&(q.model==name)].set_index('event_id').prediction
                for eid,val in zip(feat.loc[test,'event_id'],independent):assert math.isclose(val,saved[eid],rel_tol=1e-11,abs_tol=1e-12)
        # Held-out target perturbation leaves a fold's train-only predictions unchanged.
        first=fold.iloc[0]; train=feat.group!=first.group; test=~train
        x=feat[['shock_size','build_up']].to_numpy(float);y=feat.target.to_numpy(float);y2=y.copy();y2[test]+=999
        a=core.ridge_predict(x[train],y[train],x[test]);b=core.ridge_predict(x[train],y2[train],x[test]);assert np.array_equal(a,b)
        saved=q[(q.group==first.group)&(q.model=='size_plus_build_up')].set_index('event_id').prediction
        for eid,v in zip(feat.loc[test,'event_id'],a): assert math.isclose(v,saved[eid],rel_tol=1e-12,abs_tol=1e-12)
    # Test-outcome perturbation: prediction from an unchanged training slice is identical.
    first=p[(p.model=='E1')&(p.design=='peak_primary')].iloc[0]
    panel=m[(m.event_id==first.event_id)&(m.maturity==first.maturity)&(m.design==first.design)].iloc[0]
    series=ts[first.maturity];a=int(ts.index.searchsorted(pd.Timestamp(panel.anchor)));on=int(ts.index.searchsorted(pd.Timestamp(panel.onset)))
    pre=series.iloc[:on].dropna().iloc[-250:].to_numpy(float);v=series.iloc[a:a+45].to_numpy(float);z=core.normalize(pre,v[0],v)
    key=['S03','peak_primary',first.event_id,first.maturity,5,24]
    fit=core.fit('E1',np.arange(5,25),z[5:25],key)['best']['parameters']
    v2=v.copy();v2[25:45]*=3;z2=core.normalize(pre,v2[0],v2)
    fit2=core.fit('E1',np.arange(5,25),z2[5:25],key)['best']['parameters']
    assert fit==fit2
    expected=np.log(pre).mean()+(math.log(v[0])-np.log(pre).mean())*core.curve('E1',np.arange(25,45),fit)
    saved=p[(p.event_id==first.event_id)&(p.maturity==first.maturity)&(p.design=='peak_primary')&(p.model=='E1')].sort_values('t').pred_log.to_numpy()
    assert np.allclose(expected,saved,rtol=1e-12,atol=1e-12)
    result={'status':'PASS','frozen_hashes':len(f['sha256']),'predictions_checked':len(p),'coefficient_cutoffs_checked':len(c),'features_checked':len(feat),'folds_checked':len(fold),'test_perturbation':'unchanged','heldout_target_perturbation':'unchanged' if pr['status']=='estimable' else 'non_estimable'}
    (OUT/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
