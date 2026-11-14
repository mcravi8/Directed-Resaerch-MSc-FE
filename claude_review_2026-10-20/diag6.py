import numpy as np
import os, sys
# Run from anywhere: python3 diagX.py [path/to/Directed Research]
U=os.path.join(sys.argv[1] if len(sys.argv)>1 else os.path.join(os.path.dirname(os.path.abspath(__file__)),".."),"")
OUT=os.path.dirname(os.path.abspath(__file__))
from scipy.optimize import least_squares
t,m,K=np.load(os.path.join(OUT,"panel.npy")); Zb=np.load(os.path.join(OUT,"panelZ.npy"))
M={"EXP":(lambda q,t:q[0]+q[1]*np.exp(-t/q[2]),[0.,0.5,15.],[-1,0,0.2],[1,5,5000]),
   "PL":(lambda q,t:q[0]+q[1]*(1+t/q[2])**(-q[3]),[0.,0.5,5.,1.],[-1,0,1e-3,0.01],[1,50,1e4,20]),
   "2EXP":(lambda q,t:q[0]+q[1]*np.exp(-t/q[2])+q[3]*np.exp(-t/q[4]),[0.,0.3,2.,0.4,20.],[-1,0,0.2,0,0.2],[1,5,5000,5,5000])}
def fit(f,p0,lo,hi,tt,y):
    best=None
    for j in range(20):
        rng=np.random.default_rng(j); q0=np.clip(np.array(p0)+0.5*np.abs(np.array(p0)+0.1)*rng.standard_normal(len(p0))*(j>0),np.array(lo)+1e-6,np.array(hi)-1e-6)
        r=least_squares(lambda q:f(q,tt)-y,q0,bounds=(lo,hi))
        if best is None or r.cost<best.cost: best=r
    n=len(y); return best.x, n*np.log(2*best.cost/n)+2*len(p0)
for start in (1,2,3,5):
    tt=t[start:]; 
    r={k:fit(*v,tt,m[start:]) for k,v in M.items()}
    rng=np.random.default_rng(590); D=[]
    for b in range(300):
        idx=rng.integers(0,len(Zb),len(Zb)); y=Zb[idx].mean(0)[start:]
        rb={k:fit(*v,tt,y) for k,v in M.items()}
        D.append([rb["EXP"][1]-rb["PL"][1], rb["2EXP"][1]-rb["PL"][1], rb["2EXP"][1]-rb["EXP"][1]])
    D=np.array(D)
    print(f"start t={start}: EXP tau={r['EXP'][0][2]:.1f}  PL c={r['PL'][0][2]:.1f} p={r['PL'][0][3]:.2f}  2EXP taus={r['2EXP'][0][2]:.1f},{r['2EXP'][0][4]:.1f}")
    print("   P(PL beats EXP by AIC)=%.2f  median dAIC(EXP-PL)=%.1f | P(2EXP beats PL)=%.2f | P(2EXP beats EXP by >2)=%.2f"%((D[:,0]>0).mean(),np.median(D[:,0]),(D[:,1]<0).mean(),(D[:,2]<-2).mean()))
