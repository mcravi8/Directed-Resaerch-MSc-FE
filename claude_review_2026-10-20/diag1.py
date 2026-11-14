import pandas as pd, numpy as np
import os, sys
# Run from anywhere: python3 diagX.py [path/to/Directed Research]
U=os.path.join(sys.argv[1] if len(sys.argv)>1 else os.path.join(os.path.dirname(os.path.abspath(__file__)),".."),"")
OUT=os.path.dirname(os.path.abspath(__file__))
f=pd.read_csv(U+"v3_run_2026-10-20/v3_out/v3_fits.csv")
print(f.groupby(['spec','window_rule','H']).size())
d=f[(f.spec=='free')&(f.H==250)&(~f.fit_failed.astype(bool))].copy()
print("free250 fitted", len(d), "primary", d.primary.sum())
# 1. PL->EXP limit: compare p/c with 1/tau
d['rate_pl']=d.pl_p/d.pl_c
d['rate_ex']=1/d.exp_tau
d['ratio']=d.rate_pl/d.rate_ex
big=d[(d.pl_p>=4)&(d.pl_c>=18)]
print("\nridge fits (p>=4,c>=18):",len(big)," of fitted",len(d))
print(big[['event_id','maturity','pl_p','pl_c','exp_tau','ratio','dAIC','pl_SSE','exp_SSE','primary']].to_string())
print("\nmedian ratio p/c / (1/tau) on ridge:", big.ratio.median(), "IQR", big.ratio.quantile([.25,.75]).values)
# 2. dAIC near -2 => identical SSE
d['sse_rel']=(d.exp_SSE-d.pl_SSE)/d.pl_SSE
print("\nfree250 all fitted: dAIC quantiles", d.dAIC.quantile([.05,.25,.5,.75,.95]).round(2).values)
print("share with |SSE diff|<1%:", (d.sse_rel.abs()<0.01).mean().round(3), " <5%:",(d.sse_rel.abs()<0.05).mean().round(3))
p=d[d.primary.astype(bool)]
print("primary: dAIC", p.dAIC.round(2).values)
print("primary SSE rel diff", p.sse_rel.round(4).values)
print("primary ratio", p.ratio.round(2).values)
print("primary p,c", list(zip(p.pl_p.round(2),p.pl_c.round(1))))
# baseline
print("\nfree250 primary: pl_B - exp_B ", (p.pl_B-p.exp_B).round(2).values)
print("pl_B vs base_pre", (p.pl_B-p.base_pre).round(2).values)
print("resid lag1 PL quantiles all", d.resid_lag1_PL.quantile([.1,.5,.9]).round(2).values)
