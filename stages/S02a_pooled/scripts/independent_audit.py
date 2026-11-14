"""Independent read-only audit of frozen implementation and available outputs.
Added after freeze as a verifier, never used to select or modify scientific rules.
"""
import csv,hashlib,json,math
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]; OUT=ROOT/'stages/S02a_pooled'
def check_sources():
    raw=pd.read_csv(ROOT/'omori_pipeline/data/vix_term_structure.csv',parse_dates=['Date']).set_index('Date')
    v=raw.VIX30D.dropna(); catalog=pd.read_csv(ROOT/'omori_pipeline/data/shock_events.csv',parse_dates=['onset_date','peak_date'])
    detected=[]; previous=None
    for i in range(len(v)):
        crossed=(i>=1 and v.iloc[i]-v.iloc[i-1]>=7) or (i>=5 and v.iloc[i]-v.iloc[i-5]>=10)
        if crossed and (previous is None or (v.index[i]-previous).days>30):
            previous=v.index[i]; peak=v.iloc[i:i+11].idxmax(); detected.append((str(previous.date()),str(peak.date())))
    saved=[(str(r.onset_date.date()),str(r.peak_date.date())) for r in catalog.itertuples()]
    return dict(raw_detected_count=len(detected),catalog_count=len(saved),detector_matches_catalog=detected==saved,missing_from_catalog=[x for x in detected if x not in saved],not_in_detector=[x for x in saved if x not in detected])

def diagnostic_numbers():
    f=pd.read_csv(ROOT/'v3_run_2026-10-20/v3_out/v3_fits.csv')
    d=f.loc[(f.spec=='free')&(f.H==250)&(~f.fit_failed.astype(bool))].copy()
    ridge=d.loc[(d.pl_p>=4)&(d.pl_c>=18)]
    p=d.loc[d.primary.astype(bool)]
    ratio=ridge.pl_p/ridge.pl_c*ridge.exp_tau
    result=dict(fitted=len(d),primary=len(p),ridge=len(ridge),ridge_ratio_median=float(ratio.median()),ridge_ratio_iqr=ratio.quantile([.25,.75]).tolist(),primary_c_le_1_6=int((p.pl_c<=1.6).sum()),sse_relative_under_1_percent=float((abs((d.exp_SSE-d.pl_SSE)/d.pl_SSE)<.01).mean()),sse_relative_under_5_percent=float((abs((d.exp_SSE-d.pl_SSE)/d.pl_SSE)<.05).mean()),dAIC_median=float(d.dAIC.median()),residual_lag1_quantiles=d.resid_lag1_PL.quantile([.1,.5,.9]).tolist())
    bpath=OUT/'reproduction_root/claude_review_2026-10-20/oos_with_benchmarks.csv'
    if bpath.exists():
        b=pd.read_csv(bpath); groups=b.groupby(['spec','H','split'])
        ratios=groups.apply(lambda x: (x.pl_RMSE/x.exp_RMSE).median(),include_groups=False)
        result['oos_rows']=len(b);result['slot_median_PL_EXP_ratio_range']=[float(ratios.min()),float(ratios.max())]
        med=groups[['pl_RMSE','exp_RMSE','rw_RMSE','ar1_RMSE','har_RMSE']].median()
        med.to_csv(OUT/'independent_oos_medians.csv')
        result['nonfinite_HAR_slots']=int((~np.isfinite(b.har_RMSE)).sum())
        event=b.loc[(b.spec=='free')&(b.H==250)].groupby(['split','event_id'])[['exp_RMSE','har_RMSE']].mean()
        result['event_EXP_beats_HAR']=event.groupby('split').apply(lambda x:float((x.exp_RMSE<x.har_RMSE).mean()),include_groups=False).to_dict()
    return result

if __name__=='__main__':
    frozen=json.loads((OUT/'frozen_manifest.json').read_text())
    verified={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in frozen.items()}
    output=dict(frozen_hashes_checked=len(verified),frozen_hash_failures=[p for p,ok in verified.items() if not ok],source_detector=check_sources(),historical_diagnostics=diagnostic_numbers())
    (OUT/'independent_audit.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))
