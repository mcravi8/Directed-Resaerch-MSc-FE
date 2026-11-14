"""Trace reported E5 numbers to saved input and output tables."""
import csv, json
from pathlib import Path
import pandas as pd

HERE=Path(__file__).resolve().parent

def main():
    source=pd.read_csv(HERE/'source_manifest.csv')
    fit=pd.read_csv(HERE/'point_fits.csv')
    contrast=pd.read_csv(HERE/'maturity_contrasts.csv')
    status=json.loads((HERE/'summary_status.json').read_text())
    audit=json.loads((HERE/'local_verification.json').read_text())
    pool=pd.read_csv(HERE/'pooled_family_comparison.csv')
    pool_status=json.loads((HERE/'pooled_summary.json').read_text())
    rows=[]
    def add(section,item,value,file,rule):
        rows.append(dict(section=section,item=item,value=value,source_file=file,derivation=rule,
                         verification='provisional_pending_Vera'))
    for r in source.itertuples():
        for field in ('rows','first','last','onsets','eligible'):
            add('source',f'{r.index}.{field}',getattr(r,field),'source_manifest.csv','saved source screening')
    add('cohort','eligible_index_event_panels',int(source.eligible.sum()),'source_manifest.csv','sum eligible by index')
    add('cohort','global_overlap_components',status['base_clusters'],'clusters.csv','full-footprint connected components')
    add('cohort','common_vix_events',status['common_events'],'prepared.json','same legacy onset ID in all four maturities')
    for panel,g in fit[fit.start==5].groupby('panel'):
        best=g.sort_values('score').iloc[0]
        tau=g[g.family=='E1'].iloc[0]
        add('point_fit',f'{panel}.winner',best.family,'point_fits.csv','minimum four-family point score')
        add('point_fit',f'{panel}.E1_tau',tau.tau,'point_fits.csv','start-5 E1 tau parameter')
        add('point_fit',f'{panel}.base_clusters',int(tau.base_clusters),'point_fits.csv','distinct observed components in panel')
    for r in fit.itertuples():
        add('family_score',f'{r.panel}.start{r.start}.{r.family}.score',r.score,
            'point_fits.csv','n log(SSE/n)+2k on saved best fit')
    for r in contrast.itertuples():
        add('maturity',f'start{r.start}.{r.contrast}.point_difference',r.point_difference,
            'maturity_contrasts.csv','paired common-event E1 tau difference')
        add('maturity',f'start{r.start}.{r.contrast}.estimable',int(r.estimable),
            'maturity_contrasts.csv','frozen >=10 base components and >=900 valid draws')
    for field in ('point_panels','expected_panels','saved_draw_units','expected_draw_units'):
        add('completion',field,status[field],'summary_status.json','unit-file enumeration')
    for r in pool.itertuples():
        add('pooled_score',f'start{r.start}.{r.family}.score',r.point_score,
            'pooled_family_comparison.csv','equal-index pooled response; n log(SSE/n)+2k')
        if r.family=='E1':
            add('pooled_E1_tau',f'start{r.start}.tau',r.E1_tau,
                'pooled_family_comparison.csv','E1 fitted tau in sessions')
    for field in ('saved_draws','valid_draws','failed_draws','base_clusters'):
        add('pooled_completion',field,pool_status[field],'pooled_summary.json','whole-global-cluster draw accounting')
    for field in ('raw_panels_rebuilt','point_fits_checked','bootstrap_units_checked',
                  'paired_common_draws_checked','missing_draw_units','failed_bootstrap_fits',
                  'boundary_best_bootstrap_fits'):
        add('local_audit',field,audit[field],'local_verification.json','independent local arithmetic audit')
    with (HERE/'number_ledger.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,rows[0].keys());w.writeheader();w.writerows(rows)
    print(len(rows))

if __name__=='__main__': main()
