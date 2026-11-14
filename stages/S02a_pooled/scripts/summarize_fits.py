"""Descriptive summaries only; requires independently checked immutable fits."""
import json,os
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'stages/S02a_pooled'
os.environ['MPLCONFIGDIR']=str(OUT/'plot_cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from verify_fits import predict

verification=json.loads((OUT/'fit_verification.json').read_text())
if verification['errors']: raise SystemExit('Cannot summarize unverified fits')
source=OUT/'fits.csv'
if not source.exists(): raise SystemExit('No completed fits to summarize')
fits=pd.read_csv(source);summary=[];ledger=[]
(OUT/'figures').mkdir(exist_ok=True)
for unit,v in fits.groupby('unit'):
    panel=json.loads((OUT/'units'/unit/'panel.json').read_text());z=np.array(panel['trajectories']);mean=z.mean(0)
    for (zero,start),part in v.groupby(['zero_baseline','start']):
        points=part.loc[part.replicate==-1];boots=part.loc[part.replicate>=0]
        comparison=boots.pivot(index='replicate',columns='family',values='score').dropna()
        record=dict(unit=unit,zero_baseline=bool(zero),start=int(start),events=panel['event_count'],groups=panel['group_count'],planned_draws=panel['draws'],paired_complete_draws=len(comparison),descriptive_only=True,models={})
        for family in ['E1','E2','PL','SE']:
            point=points.loc[points.family==family];bootstrap=boots.loc[boots.family==family]
            if point.empty: continue
            row=point.iloc[0];pars=json.loads(row.parameters)
            params=np.array([json.loads(x) for x in bootstrap.parameters])
            record['models'][family]=dict(parameters=pars,score=float(row.score),sse=float(row.sse),complete_refits=len(bootstrap),parameter_percentiles_2_5_50_97_5=np.percentile(params,[2.5,50,97.5],axis=0).tolist() if len(params) else None,win_fraction=float((comparison.idxmin(axis=1)==family).mean()) if len(comparison) else None,total_failed_starts=int(bootstrap.failed_starts.sum()),total_boundary_starts=int(bootstrap.boundary_hits.sum()))
            ledger.append(dict(number=f'{unit}/{int(zero)}/{start}/{family}/SSE',value=float(row.sse),source='fits.csv',generation_script='scripts/pipeline.py',independent_check='scripts/verify_fits.py'))
        summary.append(record)
        if not zero and start==1 and unit.startswith('empirical') and unit.endswith('_event'):
            t=np.arange(len(mean));fig,ax=plt.subplots(figsize=(7,4));ax.plot(t,mean,color='black',label='equal-event mean')
            for family,model in record['models'].items(): ax.plot(t,predict(family,t,np.array(model['parameters'])),label=family)
            ax.axhline(0,color='gray',linewidth=.5);ax.set(xlabel='Sessions after anchor',ylabel='Normalized log volatility',title=unit);ax.legend();fig.tight_layout();fig.savefig(OUT/'figures'/f'{unit}.svg');plt.close(fig)
paired=[]
for h in [120,61,250]:
    for zero in [False,True]:
        for start in [1,3,5]:
            for mode in ['event','group']:
                reference=fits.loc[(fits.unit==f'empirical_{h}_VIX30D_common_{mode}')&(fits.family=='E1')&(fits.zero_baseline==zero)&(fits.start==start)&(fits.replicate>=0)]
                for mat in ['VIX9D','VIX3M','VIX6M']:
                    target=fits.loc[(fits.unit==f'empirical_{h}_{mat}_common_{mode}')&(fits.family=='E1')&(fits.zero_baseline==zero)&(fits.start==start)&(fits.replicate>=0)]
                    joined=target.merge(reference,on='replicate',suffixes=('_target','_reference'))
                    if joined.empty: continue
                    differences=np.array([json.loads(a)[2]-json.loads(b)[2] for a,b in zip(joined.parameters_target,joined.parameters_reference)])
                    paired.append(dict(horizon=h,maturity=mat,reference='VIX30D',zero_baseline=zero,start=start,mode=mode,paired_draws=len(joined),E1_tau_difference_percentiles=np.percentile(differences,[2.5,50,97.5]).tolist(),interpretation='Descriptive paired common-event contrast, not causal or model-family proof.'))
(OUT/'pooled_summary.json').write_text(json.dumps(dict(summaries=summary,paired_maturity_contrasts=paired),indent=2)+'\n')
pd.DataFrame(ledger).to_csv(OUT/'number_ledger.csv',index=False)
print('Summarized',len(summary),'verified fit comparisons; paired contrasts',len(paired))
