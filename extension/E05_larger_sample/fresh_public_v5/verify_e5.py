"""Independent saved-output arithmetic checks for Vera's handoff."""
import hashlib, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'stages/S01_protocol/attempt_02'))
import protocol_core as core

def main():
    freeze=json.loads((HERE/'execution_freeze.json').read_text())
    for rel,digest in freeze['hashes'].items():
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==digest,rel
    prep=json.loads((HERE/'prepared.json').read_text())
    elig=pd.read_csv(HERE/'eligibility.csv')
    assert len(prep['trajectories'])==int(elig.eligible.sum())
    assert all(len(z)==121 and np.isfinite(z).all() for z in prep['trajectories'].values())
    assert set(prep['assignment'])==set(prep['trajectories'])
    assert len(set(prep['assignment'].values()))==prep['cluster_count']
    footprints=[]
    for row in elig[elig.eligible==1].itertuples():
        footprints.append((np.datetime64(row.footprint_first),np.datetime64(row.footprint_last),
                           f'{row.index}|{row.event_id}'))
    footprints.sort()
    rebuilt={};last=None;component=-1
    for first,final,key in footprints:
        if last is None or first>last: component+=1;last=final
        else: last=max(last,final)
        rebuilt[key]=f'C{component:03d}'
    assert rebuilt==prep['assignment']
    raw_rebuilt=0
    for index,group in elig[elig.eligible==1].groupby('index'):
        source=next(r for r in prep['source_counts'] if r['index']==index)
        raw=pd.read_csv(ROOT/source['file'])
        field='CLOSE' if 'CLOSE' in raw else index
        dates=pd.to_datetime(raw.DATE,format='%m/%d/%Y').dt.strftime('%Y-%m-%d')
        lookup={d:i for i,d in enumerate(dates)}
        v=raw[field].to_numpy(float)
        for row in group.itertuples():
            on=lookup[row.onset];anchor=lookup[row.anchor]
            mu=np.log(v[on-250:on]).mean()
            den=np.log(v[anchor])-mu
            assert np.isclose(den,row.denominator,rtol=1e-14,atol=1e-14)
            z=(np.log(v[anchor:anchor+121])-mu)/den
            saved=prep['trajectories'][f'{index}|{row.event_id}']
            assert np.allclose(z,saved,rtol=1e-14,atol=1e-14)
            raw_rebuilt+=1
    panels=defaultdict(list)
    for key in prep['trajectories']:
        index,event=key.split('|',1);panels[f'{index}:full'].append(key)
        if index.startswith('VIX') and event in prep['common']: panels[f'{index}:common'].append(key)
    point_checked=0; draws_checked=0; paired=0;missing=[];failed_fits=0;boundary_best=0
    for panel,keys in panels.items():
        folder=HERE/'units'/panel.replace(':','_')
        point=json.loads((folder/'point.json').read_text())
        assert point['freeze_sha']==freeze['freeze_sha']
        assert point['events']==len(keys)
        y=np.mean([prep['trajectories'][k] for k in keys],axis=0)
        for f in point['fits']:
            assert len(f['starts'])==32
            if f['best']:
                t=np.arange(f['start'],121)
                sse=float(np.sum((core.curve(f['family'],t,f['best']['parameters'])-y[f['start']:])**2))
                assert np.isclose(sse,f['best']['sse'],rtol=1e-8,atol=1e-9)
                assert np.isclose(core.score(sse,len(t),len(core.BOUNDS[f['family']][0])),f['score'],rtol=1e-8,atol=1e-8)
            point_checked+=1
        cluster_set=set(prep['assignment'][k] for k in keys)
        for d in range(999):
            path=folder/f'{d:04d}.json'
            if not path.exists(): missing.append(f'{panel}:{d}');continue
            o=json.loads(path.read_text())
            assert o['draw_id']==d and o['freeze_sha']==freeze['freeze_sha']
            assert len(o['sampled_clusters'])==len(cluster_set)
            assert set(o['sampled_clusters'])<=cluster_set
            assert sum(o['multiplicity'].values())==len(cluster_set)
            assert o['events']==sum(o['multiplicity'].get(prep['assignment'][k],0) for k in keys)
            expanded=[k for k in keys for _ in range(o['multiplicity'].get(prep['assignment'][k],0))]
            if expanded:
                yy=np.mean([prep['trajectories'][k] for k in expanded],axis=0)
                assert len(o['fits'])==8
                for f in o['fits']:
                    assert len(f['starts'])==32
                    if f['best']:
                        tt=np.arange(f['start'],121)
                        ss=float(np.sum((core.curve(f['family'],tt,f['best']['parameters'])-yy[f['start']:])**2))
                        assert np.isclose(ss,f['best']['sse'],rtol=1e-8,atol=1e-9)
                        assert np.isclose(core.score(ss,len(tt),len(core.BOUNDS[f['family']][0])),f['score'],rtol=1e-8,atol=1e-8)
                        boundary_best+=int(any(x!=0 for x in f['best']['active_mask']))
                    else: failed_fits+=1
            draws_checked+=1
    for d in range(999):
        paths=[HERE/'units'/f'{m}_common'/f'{d:04d}.json' for m in ('VIX9D','VIX30D','VIX3M','VIX6M')]
        if all(p.exists() for p in paths):
            samples=[json.loads(p.read_text())['sampled_clusters'] for p in paths]
            assert all(s==samples[0] for s in samples)
            paired+=1
    complete=len(missing)==0 and paired==999
    if '--require-complete' in sys.argv: assert complete,(len(missing),paired)
    result=dict(status='local_checks_pass_complete' if complete else 'local_checks_pass_partial',eligible=int(elig.eligible.sum()),
        clusters=prep['cluster_count'],common_events=len(prep['common']),
        raw_panels_rebuilt=raw_rebuilt,point_fits_checked=point_checked,bootstrap_units_checked=draws_checked,
        paired_common_draws_checked=paired,expected_bootstrap_units=999*len(panels),
        missing_draw_units=len(missing),missing_ids=missing if len(missing)<100 else missing[:100],
        failed_bootstrap_fits=failed_fits,boundary_best_bootstrap_fits=boundary_best,
        freeze_sha=freeze['freeze_sha'])
    (HERE/'local_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__': main()
