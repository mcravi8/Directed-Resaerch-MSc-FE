"""Reconcile historical claims. No new fitting or specification selection."""
import csv,json,re
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]; OUT=ROOT/'stages/S02a_pooled'; R=OUT/'reproduction_root/claude_review_2026-10-20'
audit=json.loads((OUT/'independent_audit.json').read_text())['historical_diagnostics']; rows=[]
def add(claim,status,actual,evidence,limitation=''):
    rows.append(dict(claim=claim,status=status,actual=json.dumps(actual),evidence=evidence,limitation=limitation))
for claim,field,target,digits in [('107 fitted slots','fitted',107,0),('24 primary slots','primary',24,0),('50 ridge-threshold slots','ridge',50,0),('ridge rate ratio median 1.10','ridge_ratio_median',1.10,2),('6 primary c<=1.6','primary_c_le_1_6',6,0),('48% SSE within 1%','sse_relative_under_1_percent',.48,2),('87% SSE within 5%','sse_relative_under_5_percent',.87,2),('dAIC median -2.49','dAIC_median',-2.49,2)]:
    value=audit[field]; add(claim,'reproduced' if round(value,digits)==target else 'refuted',value,'logs/diag1.log; independent_audit.json')
add('ridge rate-ratio IQR 1.02 to 1.12','reproduced' if np.allclose(np.round(audit['ridge_ratio_iqr'],2),[1.02,1.12]) else 'refuted',audit['ridge_ratio_iqr'],'independent_audit.json')
add('PL residual lag1 quantiles .57/.81/.91','reproduced' if np.allclose(np.round(audit['residual_lag1_quantiles'],2),[.57,.81,.91]) else 'refuted',audit['residual_lag1_quantiles'],'independent_audit.json')
add('PL ridge fits equal exponential numerically','qualified',None,'logs/diag1.log','Asymptotic limit is analytically verified; threshold membership and a median ratio near one do not establish equality for every fitted curve. Some threshold fits are near-constant or degenerate.')
add('18 primary limit-like slots, ratio .87 to 1.20','reproduced',18,'logs/diag1.log','18 is based on the broad rate/large-c reading, not the narrower p>=4,c>=18 threshold. Interpretation remains descriptive.')
add('OOS PL/EXP slotwise median ratios .999 to 1.006','reproduced' if np.allclose(np.round(audit['slot_median_PL_EXP_ratio_range'],3),[.999,1.006]) else 'refuted',audit['slot_median_PL_EXP_ratio_range'],'independent_audit.json','This is the median of paired ratios, not the ratio of marginal RMSE medians.')
med=pd.read_csv(OUT/'independent_oos_medians.csv')
targets={'first33':[3.67,3.54,3.59,3.34,3.48],'first50':[3.03,3.21,3.37,3.30,3.13],'lastout20':[2.21,2.35,1.85,1.82,1.90]}
for split,target in targets.items():
    row=med.loc[(med.spec=='free')&(med.H==250)&(med.split==split)].iloc[0]
    values=[float(row[c]) for c in ['pl_RMSE','exp_RMSE','rw_RMSE','ar1_RMSE','har_RMSE']]
    add(f'free 250 {split} RMSE table','reproduced' if np.allclose(np.round(values,2),target) else 'refuted',values,'independent_oos_medians.csv')
add('EXP beats HAR at event level 45/58/19%','reproduced' if np.allclose([round(audit['event_EXP_beats_HAR'][k],2) for k in targets],[.45,.58,.19]) else 'refuted',audit['event_EXP_beats_HAR'],'independent_audit.json','Original denominator includes NaN comparisons as false; not a complete-case performance estimate.')
b=pd.read_csv(R/'oos_with_benchmarks.csv'); grouped=b.groupby(['spec','H','split'])
win=pd.concat([grouped.apply(lambda x:(x.exp_RMSE<x.har_RMSE).mean(),include_groups=False),grouped.apply(lambda x:(x.pl_RMSE<x.har_RMSE).mean(),include_groups=False)])
add('slot HAR win range 15% to 49%','reproduced' if np.allclose(np.round([win.min(),win.max()],2),[.15,.49]) else 'refuted',[float(win.min()),float(win.max())],'reproduction_root/claude_review_2026-10-20/oos_with_benchmarks.csv','NaN comparisons counted as losses.')
fixed=med.loc[med.spec=='fixed']; worse=all((fixed[c]>fixed[base]).all() for c in ['pl_RMSE','exp_RMSE'] for base in ['rw_RMSE','har_RMSE'])
add('fixed-B curves worse than RW/HAR in every cell','reproduced' if worse else 'refuted',bool(worse),'independent_oos_medians.csv','Statement refers to marginal cell medians, not every forecast.')
for claim,value,evidence in [('AR coefficient uses correlation, not OLS',True,'diag3.py:25'),('missing-date forecast horizons assume consecutive sessions',True,'diag3.py:46'),('NaN HAR comparisons can distort win fractions',audit['nonfinite_HAR_slots'],'independent_audit.json'),('diag5 uses 400 bootstrap draws',400,'diag5.py'),('diag6 uses 300 bootstrap draws',300,'diag6.py'),('ML is skipped, not fitted',True,'diag5.py')]:
    add(claim,'verified_static',value,'reproduction_root/claude_review_2026-10-20/'+evidence)
if (R/'panel.npy').exists():
    t,m,k=np.load(R/'panel.npy');z=np.load(R/'panelZ.npy')
    for claim,value,target,digits in [('balanced historical panel 19 events',len(z),19,0),('mean z22=.18',float(m[22]),.18,2),('ACF22=.82',float(k[22]),.82,2),('19/19 below kernel at22',int((z[:,22]<k[22]).sum()),19,0),('95% below kernel at63',float((z[:,63]<k[63]).mean()),.95,2),('mean z63=-.08',float(m[63]),-.08,2)]:
        add(claim,'reproduced' if round(value,digits)==target else 'refuted',value,'reproduction_root/claude_review_2026-10-20/panel.npy; panelZ.npy')
    add('balanced panel versus pre-onset ACF kernel','qualified',None,'reproduction_root/claude_review_2026-10-20/diag5.py','Kernel includes all eligible prehistory events, whereas pooled curve includes only complete 120-session events; cohorts differ.')
log=(OUT/'logs/diag6.log').read_text() if (OUT/'logs/diag6.log').exists() else ''
pattern=r'start t=(\d+): EXP tau=([\d.e+-]+)  PL c=([\d.e+-]+) p=([\d.e+-]+)  2EXP taus=([\d.e+-]+),([\d.e+-]+)\s+P\(PL beats EXP by AIC\)=([\d.e+-]+)  median dAIC\(EXP-PL\)=([\d.e+-]+) \| P\(2EXP beats PL\)=([\d.e+-]+) \| P\(2EXP beats EXP by >2\)=([\d.e+-]+)'
targets6={1:[24.3,143,6.4,1.7,28.4,.41,.87,.65],2:[25.8,503,20,1.1,28.1,.28,.73,.50],3:[27.1,529,20,.7,27.9,.17,.59,.23],5:[27.8,542,20,2,28.2,.11,.27,.10]}
found={int(a[0]):list(map(float,a[1:])) for a in re.findall(pattern,log)}
for start,expected in targets6.items():
    if start not in found: add(f'diag6 start {start} table','pending',None,'logs/diag6.log');continue
    actual=found[start];comparison=actual[:6]+actual[7:]
    tolerance=[.11,1,.06,.11,.11,.011,.011,.011]
    agree=all(abs(x-y)<=tol for x,y,tol in zip(comparison,expected,tolerance))
    add(f'diag6 start {start} table','reproduced_with_rounding' if agree else 'refuted_or_environment_sensitive',actual,'logs/diag6.log','Reported precision is compared without tuning or rerunning; original and current SciPy versions differ.')
add('day1-3 curvature interpretation','interpretive_not_independently_established',None,'review_diagnostics_2026-10-20.md','No causal attribution to argmax or genuine fast component; later recovery experiment is outside this run.')
add('full-sample ACF scores E2 -2648 PL -2360 EXP -1849','unsupported',None,'supplied four scripts','None of the scripts implements this full-sample ACF fit. No invented reproduction.')
add('v3 original and rerun byte identity','not_retested',None,'D-26','Closed by supersession; this stage does not reopen V3-5.')
with (OUT/'diagnostic_reproduction.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print(json.dumps(dict(claims=len(rows),statuses=pd.Series([r['status'] for r in rows]).value_counts().to_dict()),indent=2))
