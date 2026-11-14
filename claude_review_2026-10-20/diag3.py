import pandas as pd, numpy as np
import os, sys
# Run from anywhere: python3 diagX.py [path/to/Directed Research]
U=os.path.join(sys.argv[1] if len(sys.argv)>1 else os.path.join(os.path.dirname(os.path.abspath(__file__)),".."),"")
OUT=os.path.dirname(os.path.abspath(__file__))
MATS=["VIX9D","VIX30D","VIX3M","VIX6M"]
ts=pd.read_csv(U+"omori_pipeline/data/vix_term_structure.csv",parse_dates=["Date"],index_col="Date").sort_index()
ev=pd.read_csv(U+"omori_pipeline/data/shock_events.csv",parse_dates=["onset_date","peak_date"])
ev["label"]=ev["label"].fillna("").astype(str)
leh,gfc=ev[ev.label=="Lehman"],ev[ev.label=="GFC Peak"]
row=dict(onset_date=leh.onset_date.iloc[0],peak_date=gfc.peak_date.iloc[0],label="GFC (merged Lehman + GFC Peak)")
ev=pd.concat([ev.drop(index=list(leh.index)+list(gfc.index)),pd.DataFrame([row])],ignore_index=True).sort_values("onset_date").reset_index(drop=True)
ev["event_id"]=[d.strftime("%Y-%m-%d")+(" "+l if l else "") for d,l in zip(ev.peak_date,ev.label)]
onset_locs=[int(ts.index.searchsorted(d)) for d in ev.onset_date]
o=pd.read_csv(U+"v3_run_2026-10-20/v3_out_oos/v3_oos.csv")
o=o[(o.window_rule=="trunc")]
recs=[]
for ei,e in ev.iterrows():
    pk=int(ts.index.searchsorted(e.peak_date)); on=int(ts.index.searchsorted(e.onset_date))
    later=[x for x in onset_locs if x>pk]
    for mat in MATS:
        s=ts[mat]
        # pre-event history for AR(1)/HAR in logs
        hist=np.log(s.iloc[:on].dropna())
        ok_hist=len(hist)>500
        if ok_hist:
            x=hist.values; mu=x.mean()
            phi=np.corrcoef(x[1:]-mu,x[:-1]-mu)[0,1]
            # HAR on log: y_{t+1}=b0+b1 y_t + b5 mean5 + b22 mean22
            hs=pd.Series(x); m5=hs.rolling(5).mean(); m22=hs.rolling(22).mean()
            X=np.column_stack([np.ones(len(x)),x,m5,m22])[21:-1]; Y=x[22:]
            beta=np.linalg.lstsq(X,Y,rcond=None)[0]
        for H in (61,120,250):
            end=pk+H
            if later and min(later)<end: end=min(later)
            end=min(end,len(ts))
            seg=s.values[pk:end]; mask=~np.isnan(seg); y=seg[mask]; n=len(y)
            if n<15: continue
            idx_abs=np.arange(pk,end)[mask]
            for name,ntr in (("first33",int(round(0.33*n))),("first50",int(round(0.5*n))),("lastout20",n-20)):
                if ntr<8 or n-ntr<5: continue
                te=y[ntr:]; origin=y[ntr-1]
                rw=np.sqrt(np.mean((te-origin)**2))
                ar=har=np.nan
                if ok_hist:
                    h=np.arange(1,len(te)+1)  # assumes consecutive sessions
                    f=np.exp(mu+phi**h*(np.log(origin)-mu))
                    ar=np.sqrt(np.mean((te-f)**2))
                    # iterate HAR from history up to origin (log), using actual data to origin
                    oabs=idx_abs[ntr-1]
                    past=list(np.log(s.iloc[:oabs+1].dropna().values[-22:]))
                    fc=[]
                    for _ in range(len(te)):
                        v=beta@np.array([1,past[-1],np.mean(past[-5:]),np.mean(past[-22:])])
                        fc.append(v); past.append(v)
                    har=np.sqrt(np.mean((te-np.exp(fc))**2))
                recs.append(dict(event_id=e.event_id,maturity=mat,H=H,split=name,rw_RMSE=rw,ar1_RMSE=ar,har_RMSE=har))
B=pd.DataFrame(recs)
m=o.merge(B,on=["event_id","maturity","H","split"],how="inner")
print("merged",len(m),"of",len(o))
m['best_curve']=m[['pl_RMSE','exp_RMSE']].min(axis=1)
def summ(x):
    return pd.Series({'n':len(x),
     'PL':x.pl_RMSE.median(),'EXP':x.exp_RMSE.median(),'RW':x.rw_RMSE.median(),'AR1pre':x.ar1_RMSE.median(),'HARpre':x.har_RMSE.median(),
     'EXP<RW':(x.exp_RMSE<x.rw_RMSE).mean(),'PL<RW':(x.pl_RMSE<x.rw_RMSE).mean(),'EXP<HAR':(x.exp_RMSE<x.har_RMSE).mean(),'PL<HAR':(x.pl_RMSE<x.har_RMSE).mean()})
pd.set_option('display.width',250)
print(m.groupby(['spec','H','split']).apply(summ).round(2).to_string())
m.to_csv(os.path.join(OUT,"oos_with_benchmarks.csv"),index=False)
