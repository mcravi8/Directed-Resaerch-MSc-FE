"""Local cross-checks of saved E3 forecast and inference tables; Vera review separate."""
import json,math
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import t as student_t
p=Path('extension/E03_forecast');loss=pd.read_csv(p/'losses.csv');tests=pd.read_csv(p/'tests.csv');draws=pd.read_csv(p/'bootstrap_draws.csv');loo=pd.read_csv(p/'leave_one_crisis_out.csv')
assert len(tests)==24 and len(loss)==29272
assert not loss.duplicated(['maturity','h','date']).any()
assert np.allclose(loss.d,loss.har_loss-loss.harx_loss,rtol=0,atol=1e-14)
checks=[]
for row in tests.itertuples():
    a=loss[(loss.maturity==row.maturity)&(loss.h==row.h)].sort_values('date');mask=a.active.to_numpy(bool) if row.subset=='active' else np.ones(len(a),bool)
    d=a.d.to_numpy();selected=a[mask];T=len(a);A=len(selected)
    assert T==row.T and A==row.paired_n and (A if row.subset=='active' else T)==row.paired_n
    assert selected.crisis_id[selected.crisis_id>=0].nunique()==row.K
    mean=float(selected.d.mean());har=float(selected.har_loss.mean());hx=float(selected.harx_loss.mean())
    assert np.isclose(mean,row.mean_d,atol=1e-12,rtol=0) and np.isclose(mean,har-hx,atol=1e-12,rtol=0)
    assert np.isclose(har,row.har_mse,atol=1e-12,rtol=0) and np.isclose(hx,row.harx_mse,atol=1e-12,rtol=0)
    scores=np.where(mask,d-mean,0)*(T/A if row.subset=='active' else 1)
    variance=float(scores@scores/T+sum(2*(1-l/row.h)*(scores[l:]@scores[:-l]/T) for l in range(1,row.h)))
    factor=math.sqrt((T+1-2*row.h+row.h*(row.h-1)/T)/T)
    stat=factor*mean/math.sqrt(variance/T);pv=float(2*student_t.sf(abs(stat),T-1))
    assert np.isclose(variance,row.V,atol=1e-12,rtol=0) and np.isclose(factor,row.F,atol=1e-12,rtol=0)
    assert np.isclose(stat,row.dm_stat,atol=1e-10,rtol=0) and np.isclose(pv,row.p_raw,atol=1e-12,rtol=0)
    z=draws[(draws.maturity==row.maturity)&(draws.h==row.h)&(draws.subset==row.subset)]
    if row.K>=10:
        assert len(z)==1999 and z.replicate.nunique()==1999
        limits=np.percentile(z.mean_d,[2.5,97.5],method='linear')
        assert np.allclose(limits,[row.bootstrap_lo,row.bootstrap_hi],atol=1e-12,rtol=0)
    else:assert len(z)==0
    l=loo[(loo.maturity==row.maturity)&(loo.h==row.h)&(loo.subset==row.subset)]
    assert len(l)==row.K
    for item in l.itertuples():
        remaining=selected[selected.crisis_id!=item.omitted_crisis]
        assert len(remaining)==item.remaining_n and np.isclose(remaining.d.mean(),item.mean_d,atol=1e-12,rtol=0)
    checks.append(dict(maturity=row.maturity,h=row.h,subset=row.subset,mean_mse_dm_bootstrap_loo='matched'))
secondary=tests.iloc[1:].copy();order=sorted(range(23),key=lambda i:(secondary.iloc[i].p_raw,i));adj=np.zeros(23);prior=0
for rank,i in enumerate(order):prior=max(prior,(23-rank)*secondary.iloc[i].p_raw);adj[i]=min(1,prior)
assert np.allclose(adj,secondary.holm_p.to_numpy(),atol=1e-12,rtol=0)
summary=dict(status='local_checks_passed_pending_Vera',forecast_rows=len(loss),test_slots=len(tests),holm_slots=len(secondary),bootstrap_draw_rows=len(draws),loo_rows=len(loo),checks=checks)
(p/'local_verification.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='checks'}))
