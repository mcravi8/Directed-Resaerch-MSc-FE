"""S01 executable contracts. No stage runner; only synthetic tests are authorized here.

Later stages must implement their orchestration and freeze its code before outcomes.
This module implements reusable contracts, not completed empirical or recovery work.
"""
import hashlib, json, math
import numpy as np
from scipy.optimize import least_squares

SEED=5902026
STARTS=(1,3,5)
GRID=tuple([('OU',t) for t in (10,30,90)]+
           [('2OU',a,b,w) for a,b in ((2,30),(5,90)) for w in (.25,.75)]+
           [('ARFIMA',d) for d in (.2,.4)])
BOUNDS={
 'E1':([-2,0,.2],[2,5,5000]),
 'E2':([-2,0,0,.2,.2],[2,5,1,5000,5000]),
 'PL':([-2,0,.001,.01],[2,5,100000,1000]),
 'SE':([-2,0,.2,.1],[2,5,5000,1])}

def seed(*key):
    raw=json.dumps([SEED,*key],separators=(',',':'),ensure_ascii=True).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8],'big')

def curve(family,t,p):
    t=np.asarray(t,dtype=float); b,a,*q=p
    if np.any(t<0): raise ValueError('time must remain measured from anchor')
    if family=='E1': return b+a*np.exp(-t/q[0])
    if family=='E2':
        w,t1,t2=q
        if t1>t2: raise ValueError('ordered time constants required')
        return b+a*(w*np.exp(-t/t1)+(1-w)*np.exp(-t/t2))
    if family=='PL': return b+a*np.exp(-q[1]*np.log1p(t/q[0]))
    if family=='SE': return b+a*np.exp(-(t/q[0])**q[1])
    raise ValueError(family)

def ordered(family,p):
    p=np.array(p,copy=True)
    if family=='E2' and p[3]>p[4]:
        p[3],p[4]=p[4],p[3]; p[2]=1-p[2]
    return p

def fit(family,t,y,key,fixed_zero=False):
    """32 bounded TRF starts, preserve original t; pooled mean passed by caller."""
    t,y=np.asarray(t,float),np.asarray(y,float)
    if t.shape!=y.shape or not np.all(np.isfinite(y)): raise ValueError('invalid curve')
    lo,hi=map(lambda x:np.asarray(x,float),BOUNDS[family])
    rng=np.random.default_rng(seed(*key,family)); records=[]; best=None
    def expand(x): return np.r_[0.,x] if fixed_zero else x
    def predict(x): return curve(family,t,ordered(family,expand(x)))
    for i in range(32):
        x=lo+(hi-lo)*rng.random(len(lo))
        # Log-uniform positive scale/shape starts; fixed before any outcomes.
        indices={'E1':[2],'E2':[3,4],'PL':[2,3],'SE':[2]}[family]
        for j in indices: x[j]=math.exp(rng.uniform(math.log(lo[j]),math.log(hi[j])))
        if fixed_zero: x=x[1:]
        sol=least_squares(lambda p:predict(p)-y,x,bounds=(lo[1:],hi[1:]) if fixed_zero else (lo,hi),
                          method='trf',ftol=1e-8,xtol=1e-8,gtol=1e-8,max_nfev=5000)
        sse=float(np.sum(sol.fun**2)); p=ordered(family,expand(sol.x))
        ok=bool(sol.success and np.isfinite(sse) and np.all(np.isfinite(p)))
        rec=dict(start=i,success=ok,status=int(sol.status),nfev=int(sol.nfev),sse=sse,
                 parameters=p.tolist(),active_mask=sol.active_mask.tolist())
        records.append(rec)
        if ok and (best is None or sse<best['sse']): best=rec
    return dict(best=best,starts=records,k=len(lo)-int(fixed_zero),n=len(t))

def score(sse,n,k):
    if n<=0 or sse<0 or not np.isfinite(sse): raise ValueError('invalid SSE or n')
    return -math.inf if sse==0 else n*math.log(sse/n)+2*k

def normalize(pre,anchor,post):
    pre,post=np.asarray(pre,float),np.asarray(post,float)
    if len(pre)!=250 or np.any(pre<=0) or not np.all(np.isfinite(pre)): raise ValueError('250 valid pre-onset closes required')
    if anchor<=0 or not np.isfinite(anchor) or np.any(post<=0) or not np.all(np.isfinite(post)): raise ValueError('invalid closes')
    mu=np.log(pre).mean(); den=np.log(anchor)-mu
    if den<=1e-8: raise ValueError('nonpositive normalization denominator')
    return (np.log(post)-mu)/den

def detect(dates,values):
    """Raw detector only; no manual named merges. Dates correspond to observed values."""
    v=np.asarray(values,float); days=np.asarray(dates,dtype='datetime64[D]')
    if len(v)!=len(days) or np.any(np.diff(days)<=np.timedelta64(0,'D')): raise ValueError('dates not unique increasing')
    if not np.all(np.isfinite(v)): raise ValueError('pass observed closes only')
    triggers=np.zeros(len(v),bool)
    triggers[1:] |= np.diff(v)>=7
    triggers[5:] |= v[5:]-v[:-5]>=10
    out=[]; last=None
    for i in np.flatnonzero(triggers):
        if last is None or days[i]-days[last]>np.timedelta64(30,'D'):
            out.append((int(i),int(i+np.argmax(v[i:i+11]))));last=i
    return out

def allocation(t_ou,t_2ou,t_lm):
    times=np.asarray([t_ou,t_2ou,t_lm],float)
    if np.any(times<=0) or not np.all(np.isfinite(times)): raise ValueError('complete positive pilot times required')
    return min(100,math.floor(21600/float(np.dot([3,4,2],2*times))))

def arfima_cov(n,d,sd=.4):
    if not 0<d<.5: raise ValueError('stationary d required')
    r=np.empty(n);r[0]=sd**2
    for k in range(1,n): r[k]=r[k-1]*(k-1+d)/(k-d)
    return r

def overlap_groups(intervals):
    """Closed raw-footprint intervals; transitive intersections share a group."""
    groups=np.empty(len(intervals),int); end=None; group=-1
    for i in sorted(range(len(intervals)),key=lambda i:intervals[i][0]):
        a,b=intervals[i]
        if a>b: raise ValueError('reversed footprint')
        if end is None or a>end: group+=1;end=b
        else: end=max(end,b)
        groups[i]=group
    return groups

def buildup(pre20):
    x=np.asarray(pre20,float)
    if len(x)!=20 or np.any(x<=0) or not np.all(np.isfinite(x)): raise ValueError('20 valid strictly pre-onset observations required')
    t=np.arange(20)-9.5
    return float(t@np.log(x)/(t@t))

def ridge_predict(train_x,train_y,test_x):
    x=np.asarray(train_x,float); y=np.asarray(train_y,float); z=np.asarray(test_x,float)
    mean=x.mean(axis=0); scale=x.std(axis=0); scale=np.where(scale==0,1,scale)
    x=(x-mean)/scale; z=(z-mean)/scale; ym=y.mean()
    coef=np.linalg.solve(x.T@x+np.eye(x.shape[1]),x.T@(y-ym))
    return ym+z@coef

def logo_predictions(x,y,groups):
    x,y,g=np.asarray(x,float),np.asarray(y,float),np.asarray(groups)
    if len(y)<20 or len(np.unique(g))<10: raise ValueError('non-estimable: events/groups')
    out=np.full(len(y),np.nan)
    for group in np.unique(g):
        train=g!=group
        if train.sum()<15: raise ValueError('non-estimable: training count')
        out[~train]=ridge_predict(x[train],y[train],x[~train])
    return out

def forecast_benchmark(pre_onset,observed_to_origin,horizon,family):
    """Log observations. Coefficients use pre-onset data only; recursive state uses origin."""
    pre=np.asarray(pre_onset,float)[-756:]; hist=list(np.asarray(observed_to_origin,float))
    if not np.all(np.isfinite(pre)) or not np.all(np.isfinite(hist)): raise ValueError('missing benchmark dates')
    if family=='RW': return np.repeat(hist[-1],horizon)
    lag=1 if family=='AR1' else 22
    if family not in ('AR1','HAR') or len(pre)-lag<252: raise ValueError('insufficient regression rows')
    def row(h): return [1,h[-1]] if family=='AR1' else [1,h[-1],np.mean(h[-5:]),np.mean(h[-22:])]
    coef=np.linalg.lstsq(np.array([row(pre[:i]) for i in range(lag,len(pre))]),pre[lag:],rcond=None)[0]
    if len(hist)<lag: raise ValueError('missing lag state')
    result=[]
    for _ in range(horizon):
        pred=float(np.dot(row(hist),coef));result.append(pred);hist.append(pred)
    return np.array(result)

def stage_runner(*args,**kwargs):
    raise RuntimeError('S01 contracts only: later-stage execution requires explicit trigger and frozen orchestration')
