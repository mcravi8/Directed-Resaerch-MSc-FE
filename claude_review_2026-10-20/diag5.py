import pandas as pd, numpy as np
import os, sys
# Run from anywhere: python3 diagX.py [path/to/Directed Research]
U=os.path.join(sys.argv[1] if len(sys.argv)>1 else os.path.join(os.path.dirname(os.path.abspath(__file__)),".."),"")
OUT=os.path.dirname(os.path.abspath(__file__))
from scipy.optimize import least_squares
ts=pd.read_csv(U+"omori_pipeline/data/vix_term_structure.csv",parse_dates=["Date"],index_col="Date").sort_index()
ev=pd.read_csv(U+"omori_pipeline/data/shock_events.csv",parse_dates=["onset_date","peak_date"])
ev["label"]=ev["label"].fillna("").astype(str)
leh,gfc=ev[ev.label=="Lehman"],ev[ev.label=="GFC Peak"]
ev=pd.concat([ev.drop(index=list(leh.index)+list(gfc.index)),pd.DataFrame([dict(onset_date=leh.onset_date.iloc[0],peak_date=gfc.peak_date.iloc[0],label="GFC")])],ignore_index=True).sort_values("onset_date").reset_index(drop=True)
s=ts["VIX30D"].dropna(); x=np.log(s.values); T=120
onsets=[int(s.index.searchsorted(d)) for d in ev.onset_date]
Z=np.full((len(ev),T+1),np.nan); kernel=[]
for i,e in ev.iterrows():
    pk=int(s.index.searchsorted(e.peak_date)); on=onsets[i]
    if on<260: continue
    mu=x[on-250:on].mean()                     # 1y pre-onset mean of log VIX
    end=min([o for o in onsets if o>pk]+[pk+T+1,len(x)])
    seg=(x[pk:end]-mu)/(x[pk]-mu); Z[i,:len(seg)]=seg
    # linear-response kernel from pre-onset history only: ACF of log VIX up to onset
    h=x[:on]-x[:on].mean()
    kernel.append([1.0]+[np.corrcoef(h[k:],h[:-k])[0,1] for k in range(1,T+1)])
K=np.nanmean(np.array(kernel),0)
full=~np.isnan(Z[:,T])
print("events with full 120d window and >=1y history:",full.sum())
Zb=Z[full]; t=np.arange(T+1.0); m=Zb.mean(0)
print("balanced-panel mean z at t=1,5,10,22,44,63,120:",m[[1,5,10,22,44,63,120]].round(3))
print("pre-onset ACF kernel  at t=1,5,10,22,44,63,120:",K[[1,5,10,22,44,63,120]].round(3))
# per-event: fraction of events whose z(22) below kernel(22)
print("share events with z(22)<ACF(22):",(Zb[:,22]<K[22]).mean().round(2), " z(63)<ACF(63):",(Zb[:,63]<K[63]).mean().round(2))
# fits on balanced panel, asymptote fixed at 0 (mu = 1y mean), t=1..120
M={"EXP":(lambda q,t:np.exp(-t/q[0]),[20.],[0.2],[5000]),
   "PL":(lambda q,t:(1+t/q[0])**(-q[1]),[3.,0.5],[1e-3,0.01],[1e4,20]),
   "2EXP":(lambda q,t:q[0]*np.exp(-t/q[1])+(1-q[0])*np.exp(-t/q[2]),[0.6,3.,80.],[0,0.2,0.2],[1,5000,5000]),
   "ML":(None,None,None,None)}
from scipy.special import gamma
def mittag(a,z,N=200):
    k=np.arange(N); 
    out=np.zeros_like(z)
    for kk in k: out+= z**kk/gamma(a*kk+1)
    return out
def fit(f,p0,lo,hi,tt,y):
    best=None
    for j in range(15):
        rng=np.random.default_rng(j); q0=np.clip(np.array(p0)*np.exp(0.7*rng.standard_normal(len(p0))*(j>0)),np.array(lo)+1e-6,np.array(hi)-1e-6)
        r=least_squares(lambda q:f(q,tt)-y,q0,bounds=(lo,hi))
        if best is None or r.cost<best.cost: best=r
    n=len(y); return best.x, n*np.log(2*best.cost/n)+2*len(p0)
tt=t[1:]
def allfits(y):
    o={}
    for k,(f,p0,lo,hi) in M.items():
        if k=="ML": continue
        o[k]=fit(f,p0,lo,hi,tt,y)
    return o
r=allfits(m[1:])
for k,(q,a) in r.items(): print(k,np.round(q,3),"AIC",round(a,1))
rng=np.random.default_rng(590); D=[]
for b in range(400):
    idx=rng.integers(0,len(Zb),len(Zb)); rb=allfits(Zb[idx].mean(0)[1:])
    D.append([rb["EXP"][1]-rb["PL"][1], rb["2EXP"][1]-rb["PL"][1], rb["PL"][0][1], rb["2EXP"][0][1], rb["2EXP"][0][2]])
D=np.array(D)
print("event-bootstrap (n=%d events) AIC_EXP-AIC_PL:"%len(Zb),np.percentile(D[:,0],[2.5,50,97.5]).round(1))
print("AIC_2EXP-AIC_PL:",np.percentile(D[:,1],[2.5,50,97.5]).round(1), " share 2EXP better:",(D[:,1]<0).mean().round(2))
print("PL p:",np.percentile(D[:,2],[2.5,50,97.5]).round(2)," 2EXP fast/slow tau:",np.percentile(D[:,3],[2.5,50,97.5]).round(1),np.percentile(D[:,4],[2.5,50,97.5]).round(1))
np.save(os.path.join(OUT,"panel.npy"),np.vstack([t,m,K]))
np.save(os.path.join(OUT,"panelZ.npy"),Zb)
