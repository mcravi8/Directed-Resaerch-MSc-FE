"""S02c descriptive outputs and post-hoc grouping sensitivity. No S02a/S03 mutation."""
import json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'stages/S02c_review'
sys.path.insert(0,str(ROOT/'stages/S01_protocol/attempt_02'))
import protocol_core as core
sys.path.insert(0,str(ROOT/'stages/S02a_pooled/scripts'))
import pipeline
MATS=['VIX9D','VIX30D','VIX3M','VIX6M']

def write_csv(name, rows):
    pd.DataFrame(rows).to_csv(OUT/name,index=False)

def main():
    OUT.mkdir(exist_ok=True)
    summary=json.loads((ROOT/'stages/S02a_pooled/pooled_summary.json').read_text())['summaries']
    selected={m:next(x for x in summary if x['unit']==f'empirical_120_{m}_full_event' and x['start']==5 and not x['zero_baseline']) for m in MATS}
    tau=[]
    for m,x in selected.items():
        e=x['models']['E1']; p=e['parameter_percentiles_2_5_50_97_5']
        tau.append(dict(maturity=m,events=x['events'],point_tau=e['parameters'][2],lower_2_5=p[0][2],median_50=p[1][2],upper_97_5=p[2][2],complete_refits=e['complete_refits']))
    write_csv('C6_E1_tau_intervals.csv',tau)

    ts,ev=pipeline.inputs(); cohort=pd.read_csv(ROOT/'stages/S02a_pooled/cohort_manifest.csv')
    flags=cohort[(cohort.horizon==120)&(cohort.anchor=='argmax')].set_index(['event_id','maturity']).eligible
    rows=[]
    for _,e in ev.iterrows():
        onset=pd.Timestamp(e.onset_date); peak=pd.Timestamp(e.peak_date)
        ix=int(ts.index.searchsorted(onset)); pre=ts.VIX30D.iloc[:ix].dropna().iloc[-250:]
        baseline=float(np.exp(np.log(pre).mean())) if len(pre)==250 else np.nan
        row=dict(event_id=e.event_id,onset_date=onset.date(),peak_date=peak.date(),peak_VIX30D=float(ts.loc[peak,'VIX30D']),pre_onset_baseline_geometric_250=baseline,label=e.label)
        for m in MATS: row[f'{m}_complete_120']=bool(flags.loc[(e.event_id,m)])
        rows.append(row)
    assert len(rows)==33
    write_csv('C5_event_table.csv',rows)

    losses=pd.read_csv(ROOT/'stages/S03_validation/event_losses.csv')
    manifest=pd.read_csv(ROOT/'stages/S03_validation/train_test_manifest.csv')
    manifest=manifest[manifest.eligible==True].copy()
    sign=[]; group_results=[]; rmse=[]
    for (design,mat),g in losses.groupby(['design','maturity'],sort=True):
        q=g.pivot(index='event_id',columns='model',values='log_mse').dropna()
        assert len(q)>0 and {'HAR','E1','E2','PL','SE'}<=set(q.columns)
        h=q.HAR; e=q.E1; n=len(q)
        sign.append(dict(design=design,maturity=mat,n=n,HAR_better_E1=int((h<e).sum()),ties=int((h==e).sum()),E1_better_HAR=int((h>e).sum())))
        for model in q.columns:
            rmse.append(dict(design=design,maturity=mat,model=model,n=n,matched_RMSE=float(np.sqrt(q[model].mean()))))
        sub=manifest[(manifest.design==design)&(manifest.maturity==mat)].set_index('event_id').loc[q.index]
        # Disjoint closed intervals; touching windows are assigned to one component.
        dates=[(pd.Timestamp(z.onset).value,pd.Timestamp(z.test_last).value) for _,z in sub.iterrows()]
        groups=core.overlap_groups(dates)
        ng=len(np.unique(groups)); fixed_best=min(['E1','E2','PL','SE'],key=lambda model:q[model].mean())
        base=dict(design=design,maturity=mat,n=n,frozen_group_count=int(sub.group.nunique()),posthoc_group_count=ng,best_curve=fixed_best,method='test_window_overlap_post_hoc')
        if ng<10:
            group_results.extend([dict(**base,comparison=comp,point_HAR_minus_curve=float(np.sqrt(h.mean())-np.sqrt(q[model].mean())),lower_2_5=np.nan,upper_97_5=np.nan,draws=0,status='fewer_than_10_groups') for comp,model in [('E1','E1'),('best_curve',fixed_best)]]);continue
        rng=np.random.default_rng(core.seed('S02c','C3',design,mat,'group-bootstrap'))
        unique=np.unique(groups); boot={'E1':[],'best_curve':[]}
        for _ in range(9999):
            draw=rng.choice(unique,len(unique),replace=True)
            idx=np.concatenate([np.flatnonzero(groups==j) for j in draw])
            for name,model in [('E1','E1'),('best_curve',fixed_best)]:
                boot[name].append(float(np.sqrt(h.iloc[idx].mean())-np.sqrt(q[model].iloc[idx].mean())))
        for name,model in [('E1','E1'),('best_curve',fixed_best)]:
            lo,hi=np.percentile(boot[name],[2.5,97.5])
            group_results.append(dict(**base,comparison=name,point_HAR_minus_curve=float(np.sqrt(h.mean())-np.sqrt(q[model].mean())),lower_2_5=lo,upper_97_5=hi,draws=9999,status='computed'))
    write_csv('C3_per_event_signs.csv',sign)
    write_csv('C3_group_sensitivity.csv',group_results)
    write_csv('C4_matched_RMSE.csv',rmse)

    figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
    panels={m:pipeline.panels(ts,ev,m,120)[0] for m in MATS}
    paths={m:np.mean(np.array(list(panels[m].values())),axis=0) for m in MATS}
    t=np.arange(121); s1=next(x for x in summary if x['unit']=='empirical_120_VIX30D_full_event' and x['start']==1 and not x['zero_baseline'])
    fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    baseline=s1['models']['E1']['parameters'][0]
    axs[0].plot(t,paths['VIX30D'],color='black',lw=2,label='Pooled normalized path')
    axs[1].plot(t,np.where(paths['VIX30D']>baseline,paths['VIX30D']-baseline,np.nan),color='black',lw=2,label='Pooled normalized excess')
    for family in ['E1','E2','PL','SE']:
        fit=core.curve(family,t,s1['models'][family]['parameters'])
        axs[0].plot(t,fit,label=family)
        axs[1].plot(t,np.where(fit>baseline,fit-baseline,np.nan),label=family)
    axs[0].set(xlabel='Sessions since peak',ylabel='Normalized log VIX30D')
    axs[1].set(xlabel='Sessions since peak',ylabel='Positive excess above E1 fitted baseline')
    for ax in axs:ax.grid(alpha=.2)
    axs[1].set_yscale('log');axs[1].set_ylim(bottom=.01)
    axs[0].legend(fontsize=8)
    fig.savefig(figdir/'C4_figure1_linear_log_baseline_corrected.png',dpi=180);plt.close(fig)

    fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained',gridspec_kw={'width_ratios':[2,1]})
    colors=plt.cm.tab10.colors
    for i,m in enumerate(MATS):
        axs[0].plot(t,paths[m],color=colors[i],alpha=.5,label=m+' pooled')
        axs[0].plot(t,core.curve('E1',t,selected[m]['models']['E1']['parameters']),color=colors[i],linestyle='--',label=m+' E1')
        x=tau[i];axs[1].errorbar(i,x['point_tau'],yerr=[[x['point_tau']-x['lower_2_5']],[x['upper_97_5']-x['point_tau']]],fmt='o',color=colors[i])
    axs[0].set(xlabel='Sessions since peak',ylabel='Normalized log VIX')
    axs[0].legend(ncol=2,fontsize=7)
    axs[1].set_xticks(range(4),MATS);axs[1].set(ylabel='E1 tau, sessions')
    fig.savefig(figdir/'C4_term_structure.png',dpi=180);plt.close(fig)

    r=pd.DataFrame(rmse);primary=r[r.design.isin(['peak_primary','onset5_primary'])]
    fig,ax=plt.subplots(figsize=(12,5),layout='constrained')
    cells=primary[['design','maturity']].drop_duplicates().sort_values(['design','maturity']); labels=[f'{d}\n{m}' for d,m in cells.itertuples(index=False,name=None)]
    for i,model in enumerate(['RW','AR1','HAR','E1','E2','PL','SE']):
        vals=[float(primary[(primary.design==d)&(primary.maturity==m)&(primary.model==model)].matched_RMSE.iloc[0]) for d,m in cells.itertuples(index=False,name=None)]
        ax.scatter(np.arange(len(cells))+(i-3)*.08,vals,label=model,s=24)
    ax.set_xticks(range(len(cells)),labels);ax.set(ylabel='Matched-event RMSE, log VIX',xlabel='Primary design and maturity');ax.legend(ncol=7,fontsize=8)
    fig.savefig(figdir/'C4_primary_matched_RMSE.png',dpi=180);plt.close(fig)
    print(json.dumps({'event_rows':len(rows),'sign_cells':len(sign),'group_cells':len(group_results)//2,'intervals':tau},default=str))

if __name__=='__main__':main()
