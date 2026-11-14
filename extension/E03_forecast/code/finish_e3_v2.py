"""Score frozen paired forecasts only after corrected independent audit passes."""
import json
from pathlib import Path
from importlib.machinery import SourceFileLoader
import numpy as np
import pandas as pd
e=SourceFileLoader('frozen_runner','extension/E03_forecast/code/run_e3.py').load_module()
root=Path('extension/E03_forecast');m=json.loads((root/'input_manifest.json').read_text())
assert e.sha(e.__file__)==m['code_sha256']
audit=json.loads((root/'leakage_audit_v2_summary.json').read_text())
assert audit['status']=='passed' and audit['comparison_rows']>0
ledger=pd.read_csv(root/'origin_dependency_ledger.csv').fillna('')
assert len(ledger)>0
for r in ledger.itertuples():
    t=pd.Timestamp(r.date);cutoff=pd.Timestamp(r.refit_cutoff)
    for field in ('feature_max_date','onset_date','baseline_start','baseline_end','running_max_source_date','age_source_date','five_change_end'):
        value=getattr(r,field)
        if value:assert pd.Timestamp(value)<=t,(field,r)
    if r.latest_train_label:assert pd.Timestamp(r.latest_train_label)<=cutoff
for path in (root/'monthly_models').glob('*.json'):
    q=json.loads(path.read_text());assert q['code_sha256']==m['code_sha256']
    model=q['model'];cutoff=pd.Timestamp(q['cutoff'])
    if model:
        assert pd.Timestamp(model['train_max_label'])<=cutoff
        for fold in model['folds']:
            year=fold['year']
            assert pd.Timestamp(fold['train_max_label'])<pd.Timestamp(year=year,month=1,day=1)
            assert pd.Timestamp(fold['validation_max_label'])<=pd.Timestamp(year=year,month=12,day=31)
pred=pd.read_csv(root/'predictions.csv');assert not pred.duplicated(['maturity','h','date']).any()
vd,vv=e.parse('VIX30D')[:2];feat=e.onset_features(vd,vv);units,cids=e.crisis_units(vd,feat)
assert all(pred.apply(lambda r:cids[int(r.j)]==int(r.crisis_id),axis=1))
pred['har_loss']=(pred.y-pred.har)**2;pred['harx_loss']=(pred.y-pred.harx)**2;pred['d']=pred.har_loss-pred.harx_loss
assert np.allclose(pred.d, pred.har_loss-pred.harx_loss,rtol=0,atol=1e-14)
pred.to_csv(root/'losses.csv',index=False)
slots=[('VIX30D',10,'active'),('VIX30D',10,'all')]+[(n,h,s) for n in e.NAMES for h in e.HS for s in ('active','all') if (n,h,s) not in [('VIX30D',10,'active'),('VIX30D',10,'all')]]
assert len(slots)==24
testrows=[];loos=[];boots=[]
for n,h,s in slots:
    q=pred[(pred.maturity==n)&(pred.h==h)]
    if len(q):t,l,b=e.tests(q,units,h,n,s)
    else:t,l,b=dict(maturity=n,h=h,subset=s,reason='no_paired_forecasts',p_raw=1,K=0,T=0,A=0,paired_n=0),[],[]
    if 'mean_d' in t:assert np.isclose(t['mean_d'],t['har_mse']-t['harx_mse'],rtol=0,atol=1e-12),(n,h,s)
    testrows.append(t);loos.extend(l);boots.extend(b)
sec=testrows[1:];order=sorted(range(23),key=lambda i:(sec[i]['p_raw'],i));cum=0.
for rank,i in enumerate(order):
    cum=max(cum,(23-rank)*sec[i]['p_raw']);sec[i]['holm_p']=min(1.,cum)
for idx,t in enumerate(testrows):
    t['positive_claim']=bool(t.get('mean_d',0)>0 and t.get('p_raw',1)<.05 and (idx==0 or t.get('holm_p',1)<.05) and t.get('positive_survives_checks',False))
e.savecsv(root/'tests.csv',testrows);e.savecsv(root/'leave_one_crisis_out.csv',loos);e.savecsv(root/'bootstrap_draws.csv',boots)
onset5=[]
for event in feat['events']:
    j=event['index']+5
    if j<len(vd):
        q=pred[(pred.maturity=='VIX30D')&(pred.h==10)&(pred.date==str(vd[j].date()))]
        if len(q):onset5.append(dict(onset_id=event['onset_id'],onset_date=event['onset_date'],origin=str(vd[j].date()),har_loss=float(q.har_loss.iloc[0]),harx_loss=float(q.harx_loss.iloc[0]),d=float(q.d.iloc[0])))
e.savecsv(root/'descriptive_onset5.csv',onset5)
e.atom(root/'score_summary.json',dict(forecasts=len(pred),test_slots=len(testrows),bootstrap_draw_rows=len(boots),leave_one_crisis_rows=len(loos),audit=audit,primary=testrows[0],secondary_claims=sum(z['positive_claim'] for z in sec)))
print(json.dumps(dict(forecasts=len(pred),primary=testrows[0],secondary_claims=sum(z['positive_claim'] for z in sec)),default=str))
