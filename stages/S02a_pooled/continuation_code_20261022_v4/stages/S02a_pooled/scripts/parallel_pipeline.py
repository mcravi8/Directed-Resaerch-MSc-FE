"""Runtime v2: identical frozen numerical fits, deterministic parallel batches."""
import os
for variable in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[variable]='1'
import concurrent.futures as cf
import multiprocessing as mp
import time
import numpy as np
import pipeline as p
WORKERS=int(os.environ.get('S02A_WORKERS','5'))
assert 1 <= WORKERS <= 10

def task(args):
    key,mode,rep,zero,start,h,y,ix,filename=args
    p.guard();begin=time.perf_counter(); fitted={}
    for family in p.core.BOUNDS:
        p.guard()
        fit=p.core.fit(family,np.arange(start,h+1),y[start:],['S02a',key,mode,rep,start,zero],zero)
        best=fit['best'];fit['score']=p.core.score(best['sse'],fit['n'],fit['k']) if best else None
        fitted[family]=fit
    record=dict(key=key,replicate=rep,zero_baseline=zero,start=start,indices=ix.tolist(),models=fitted,wall_seconds=time.perf_counter()-begin)
    p.write(filename,record)
    return filename

def fit_unit(panel,ids,intervals,key,draws,mode='event',paired_key=None):
    directory=p.OUT/'units'/key;directory.mkdir(parents=True,exist_ok=True)
    z=np.array([panel[i] for i in ids]);h=z.shape[1]-1
    groups=p.core.overlap_groups([intervals[i] for i in ids]);ug=np.unique(groups)
    metadata=dict(key=key,event_ids=ids,event_count=len(ids),group_count=len(ug),horizon=h,mode=mode,draws=draws)
    if not (directory/'panel.json').exists(): p.write(directory/'panel.json',dict(**metadata,trajectories=z.tolist(),groups=groups.tolist()))
    jobs=[]
    for rep in range(-1,draws):
        rng=np.random.default_rng(p.core.seed(paired_key or key,mode,rep,'resample'))
        if rep==-1: ix=np.arange(len(ids))
        elif mode=='event': ix=rng.integers(0,len(ids),len(ids))
        else: ix=np.concatenate([np.flatnonzero(groups==g) for g in rng.choice(ug,len(ug),replace=True)])
        y=z[ix].mean(axis=0)
        for zero in (False,True):
            for start in p.core.STARTS:
                filename=directory/f'r{rep}_b{int(zero)}_s{start}.json'
                if not filename.exists(): jobs.append((key,mode,rep,zero,start,h,y,ix,str(filename)))
    p.guard()
    with cf.ProcessPoolExecutor(max_workers=WORKERS,mp_context=mp.get_context('spawn')) as pool:
        for _ in pool.map(task,jobs,chunksize=1): p.guard()
    return metadata

if __name__=='__main__':
    import sys
    p.fit_unit=fit_unit
    if sys.argv[1]=='pilot': p.pilot(sys.argv[2])
    elif sys.argv[1]=='empirical': p.empirical()
