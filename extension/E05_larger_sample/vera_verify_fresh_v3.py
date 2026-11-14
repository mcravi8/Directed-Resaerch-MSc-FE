"""Independent arithmetic audit of the saved fresh public E5 run; no fitting or network."""
import csv, hashlib, json, math, tarfile
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'extension/E05_larger_sample'
P = BASE/'fresh_public_v5'
IND = ('VIX9D','VIX30D','VIX3M','VIX6M','VXN','RVX','OVX','GVZ')
FAMS = ('E1','E2','PL','SE')
K = {'E1':3,'E2':5,'PL':4,'SE':4}

def obj(p): return json.loads(p.read_text())
def rows(p):
    with p.open(newline='') as f: return list(csv.DictReader(f))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b,tol=1e-7): return math.isclose(float(a),float(b),rel_tol=tol,abs_tol=tol)
def seed(*args):
    data=json.dumps([5902026,*args],separators=(',',':'),ensure_ascii=True).encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:8],'big')
def curve(f,t,p):
    b,a,*q=p
    if f=='E1':return b+a*np.exp(-t/q[0])
    if f=='E2':return b+a*(q[0]*np.exp(-t/q[1])+(1-q[0])*np.exp(-t/q[2]))
    if f=='PL':return b+a*np.exp(-q[1]*np.log1p(t/q[0]))
    if f=='SE':return b+a*np.exp(-(t/q[0])**q[1])
    raise AssertionError(f)
def verify_freeze(p):
    z=obj(p); stated=z.pop('freeze_sha')
    assert hashlib.sha256(json.dumps(z,sort_keys=True).encode()).hexdigest()==stated
    missing=[]; checked=0
    for name,h in z['hashes'].items():
        q=ROOT/name
        if not q.exists(): missing.append(name);continue
        assert sha(q)==h,name
        checked+=1
    return stated,checked,missing,z['frozen_at_utc']

def main():
    evidence={}
    package,pc,missing,pt=verify_freeze(P/'public_package_freeze.json')
    pipeline,fc,missing2,ft=verify_freeze(P/'execution_freeze.json')
    pool,vc,missing3,vt=verify_freeze(P/'freeze_v2.json')
    assert not missing3
    evidence['freezes']={'package':package,'pipeline':pipeline,'pool':pool,'verified_hashes':[pc,fc,vc], 'unavailable_private_scalar_hashes':missing+missing2}
    start=obj(P/'public_run_start.json');end=obj(P/'public_run_end.json')
    assert pt<ft<vt<start['started_at_utc']<end['ended_at_utc']
    assert start['freeze_sha']==pipeline and end['worker_exit_codes']==[0]*4
    from datetime import datetime
    wall=(datetime.fromisoformat(end['ended_at_utc'])-datetime.fromisoformat(start['started_at_utc'])).total_seconds()
    assert wall<28800 and end['elapsed_seconds']<28800
    evidence['timing']={'wall_seconds':wall,'recorded_monotonic_seconds':end['elapsed_seconds'],'clock_discrepancy_seconds':wall-end['elapsed_seconds']}
    journal=[json.loads(line) for line in (P/'checkpoint_journal.jsonl').read_text().splitlines()]
    assert len(journal)>1 and journal[0]['observed_at_utc']>start['started_at_utc']
    assert [x['observed_at_utc'] for x in journal]==sorted(x['observed_at_utc'] for x in journal)
    assert journal[-1]['public_checkpoint.json']['completed_draw_units']==11988
    assert journal[-1]['checkpoint.json']['completed_draw_units']==999
    assert obj(P/'public_checkpoint.json')['completed_draw_units']==11988
    assert obj(P/'checkpoint.json')['completed_draw_units']==999
    evidence['timing']['checkpoint_observations']=len(journal)
    mf=obj(P/'output_manifest.json')
    for name,entry in {**mf['files'],**mf['archives']}.items():
        q=P/name
        assert q.stat().st_size==entry['bytes'] and sha(q)==entry['sha256'],name
    evidence['manifest_files']=len(mf['files']);evidence['manifest_archives']=len(mf['archives'])
    old=obj(BASE/'public_run_complete_v1/prepared.json'); new=obj(P/'prepared.json')
    assert old['trajectories']==new['trajectories'] and old['assignment']==new['assignment'] and old['common']==new['common']
    assert (BASE/'public_run_complete_v1/eligibility.csv').read_bytes()==(P/'eligibility.csv').read_bytes()
    tr={k:np.array(v) for k,v in new['trajectories'].items()};ass=new['assignment'];common=set(new['common'])
    counts={i:sum(k.startswith(i+'|') for k in tr) for i in IND}
    assert list(counts.values())==[9,19,10,12,11,8,8,7] and len(tr)==84 and len(set(ass.values()))==7 and len(common)==9
    evidence['cohort']={'counts':counts,'trajectories':len(tr),'clusters':len(set(ass.values())),'common_events':len(common)}
    panels={i+':full':[k for k in tr if k.startswith(i+'|')] for i in IND}
    for i in IND[:4]: panels[i+':common']=[k for k in panels[i+':full'] if k.split('|',1)[1] in common]
    points={(r['panel'],int(r['start']),r['family']):r for r in rows(P/'point_fits.csv')}
    draws={(r['panel'],int(r['draw'])):r for r in rows(P/'bootstrap_draws.csv')}
    assert len(points)==96 and len(draws)==11988
    stats=Counter();point_units={};draw_vectors={};pool_point=None
    for archive in sorted(mf['archives']):
        slug=Path(archive).name.removesuffix('.tar.gz');is_pool=slug=='equal_index_pool'
        panel=slug.replace('_',':',1) if not is_pool else None
        keys=panels.get(panel,[]);base=sorted(set(ass[k] for k in keys)) if not is_pool else sorted(set(ass.values()))
        seen=set()
        with tarfile.open(P/archive,'r:gz') as tf:
            for member in tf:
                if not member.isfile() or not member.name.endswith('.json'):continue
                name=Path(member.name).stem;d=None if name=='point' else int(name);seen.add(d)
                u=json.load(tf.extractfile(member));assert u['draw_id']==d and u['freeze_sha']==(pool if is_pool else pipeline)
                if d is None: selected=base
                else:
                    key='equal_index_pool' if is_pool else ('VIX:common' if panel.endswith(':common') else panel)
                    selected=np.random.default_rng(seed('Extension','E5','cluster_bootstrap',key,d)).choice(base,size=len(base),replace=True).tolist()
                assert u['sampled_clusters' if not is_pool else 'selected_clusters']==selected
                mult=dict(Counter(selected));assert u['multiplicity']==mult
                if is_pool:
                    assert u['created_at_utc']>vt
                    index_groups={i:[k for k in tr if k.startswith(i+'|') for _ in range(mult.get(ass[k],0))] for i in IND}
                    missing=[i for i,g in index_groups.items() if not g]
                    if missing:
                        assert d is not None and u['failure']=='missing_index' and u['missing_indices']==missing and not u['fits']
                        stats['pool_failed']+=1;continue
                    y=np.mean([np.mean([tr[k] for k in index_groups[i]],axis=0) for i in IND],axis=0)
                    assert np.allclose(y,u['pooled_response'],rtol=1e-12,atol=1e-12)
                    stats['pool_valid']+=1
                    if d is None:pool_point=u
                else:
                    expanded=[k for k in keys for _ in range(mult.get(ass[k],0))]
                    assert u['events']==len(expanded) and u['base_clusters']==len(base)
                    y=np.mean([tr[k] for k in expanded],axis=0)
                    if d is None:point_units[panel]=u
                    else:
                        stats['panel_draws']+=1;draw_vectors[(panel,d)]=selected
                        r=draws[(panel,d)];assert r['cluster_ids']=='|'.join(selected) and json.loads(r['multiplicity'])==mult
                assert len(u['fits'])==8
                for fit in u['fits']:
                    f=fit['family'];s=int(fit['start']);assert f in FAMS and s in (1,5) and len(fit['starts'])==32
                    good=[x for x in fit['starts'] if x['success']]
                    best=fit['best'];assert good and best and close(best['sse'],min(x['sse'] for x in good),1e-8)
                    if f=='E2':assert best['parameters'][3]<=best['parameters'][4]
                    t=np.arange(s,121);p=best['parameters'];sse=float(np.sum((curve(f,t,p)-y[s:])**2))
                    score=(121-s)*math.log(sse/(121-s))+2*K[f]
                    assert close(sse,best['sse']) and close(score,fit['score']), (archive,d,s,f)
                    assert (fit['tau']==p[2]) if f=='E1' else fit['tau'] is None
                    stats['fit_scores']+=1
                    if not is_pool and d is not None:stats['panel_best_bounds']+=int(any(best['active_mask']))
                    if d is None and not is_pool:
                        row=points[(panel,s,f)]
                        assert close(row['score'],score) and close(row['sse'],sse)
                        if f=='E1':assert close(row['tau'],p[2],1e-12)
                if is_pool and d is not None:stats['pool_draws']+=1
        assert seen=={None,*range(999)},archive
    assert stats['panel_draws']==11988 and stats['panel_best_bounds']==39714
    assert stats['pool_valid']==667 and stats['pool_failed']==333 and stats['pool_draws']==666
    assert stats['fit_scores']==95904+96+5336
    evidence['units']=dict(stats)
    winners={p:min(FAMS,key=lambda f:float(points[(p,5,f)]['score'])) for p in panels}
    assert Counter(winners.values())=={'E1':11,'E2':1} and winners['RVX:full']=='E2'
    tau={i:float(points[(i+':full',5,'E1')]['tau']) for i in IND}
    evidence['start5_winners']=dict(Counter(winners.values()));evidence['start5_tau']=tau
    for i,v in zip(IND,[20.642594,27.969916,24.162025,40.453590,168.677729,69.461954,212.966766,24.329413]):assert close(tau[i],v,1e-7)
    poolrows=rows(P/'pooled_family_comparison.csv');assert len(poolrows)==8
    pscores={f:float(next(r['point_score'] for r in poolrows if r['start']=='5' and r['family']==f)) for f in FAMS}
    for f,v in zip(FAMS,[-671.2227844657115,-668.4155998402442,-669.2022667462325,-669.2227844676042]):assert close(pscores[f],v,1e-10)
    assert close(next(r['E1_tau'] for r in poolrows if r['start']=='5' and r['family']=='E1'),42.13606114557731,1e-10)
    assert min(pscores,key=pscores.get)=='E1'
    assert min(FAMS,key=lambda f:float(next(r['point_score'] for r in poolrows if r['start']=='1' and r['family']==f)))=='E2'
    evidence['pool_start5_scores']=pscores
    contrasts=rows(P/'maturity_contrasts.csv');assert len(contrasts)==6
    diffs={}
    for r in contrasts:
        s=int(r['start']);mat=r['contrast'].split('-')[0];pa=mat+':common';pb='VIX30D:common'
        def get_tau(panel):return next(x['tau'] for x in point_units[panel]['fits'] if x['start']==s and x['family']=='E1')
        delta=get_tau(pa)-get_tau(pb)
        assert close(delta,r['point_difference'],1e-11) and int(r['base_clusters'])==1 and int(r['valid_draws'])==999 and int(r['estimable'])==0
        assert all(not r[k] for k in ('lo','hi','raw_p','holm_p'))
        assert all(draw_vectors[(pa,d)]==draw_vectors[(pb,d)] for d in range(999))
        diffs[f'{s}:{mat}']=delta
    evidence['maturity_differences']=diffs
    assert len(rows(P/'tau_intervals.csv'))==96
    for r in rows(P/'tau_intervals.csv'):assert int(r['estimable'])==0 and not r['lo'] and not r['hi']
    historical=obj(ROOT/'stages/S02a_pooled/pooled_summary.json')['summaries'];hist={}
    for i in IND[:4]:
        for cohort in ('full_group','full_event'):
            match=[x for x in historical if x['unit']==f'empirical_120_{i}_{cohort}' and x['start']==5 and not x['zero_baseline']]
            assert len(match)==1;hist[f'{i}_{cohort}']=match[0]['models']['E1']['parameters'][2]
    evidence['historical_tau']=hist
    with (ROOT/'stages/S05_release/manifest.sha256').open() as f:
        lines=[x.split() for x in f if x.strip()]
    assert len(lines)==4
    for digest,name in lines:assert sha(ROOT/'stages/S05_release'/name)==digest
    evidence['protected_release_hashes']=4
    ledger=rows(P/'number_ledger.csv');source={r['index']:r for r in rows(P/'source_manifest.csv')}
    local=obj(P/'local_verification.json');summary=obj(P/'summary_status.json');pooled=obj(P/'pooled_summary.json')
    for r in ledger:
        section,item,value=r['section'],r['item'],r['value']
        if section=='source': ix,field=item.split('.');expected=source[ix][field]
        elif section=='cohort': expected={'eligible_index_event_panels':84,'global_overlap_components':7,'common_vix_events':9}[item]
        elif section=='point_fit':
            panel,field=item.split('.');expected={'winner':winners[panel],'E1_tau':points[(panel,5,'E1')]['tau'],
                'base_clusters':points[(panel,5,'E1')]['base_clusters']}[field]
        elif section=='family_score':
            panel,start,fam,field=item.split('.');assert field=='score';expected=points[(panel,int(start[5:]),fam)]['score']
        elif section=='maturity':
            start,contrast,field=item.split('.');expected=next(x[field] for x in contrasts if x['start']==start[5:] and x['contrast']==contrast)
        elif section=='completion': expected=summary[item]
        elif section=='pooled_score':
            start,fam,field=item.split('.');assert field=='score';expected=next(x['point_score'] for x in poolrows if x['start']==start[5:] and x['family']==fam)
        elif section=='pooled_E1_tau':
            start,field=item.split('.');assert field=='tau';expected=next(x['E1_tau'] for x in poolrows if x['start']==start[5:] and x['family']=='E1')
        elif section=='pooled_completion': expected=pooled[item]
        elif section=='local_audit': expected=local[item]
        else: raise AssertionError(section)
        assert value==str(expected) or (value.replace('.','',1).replace('-','',1).isdigit() and close(value,expected,1e-10)), (section,item,value,expected)
    evidence['number_ledger_rows']=len(ledger)
    assert evidence['number_ledger_rows']==212
    return evidence

if __name__=='__main__':
    out=main()
    destination=BASE/'vera_fresh_evidence_v3.json'
    destination.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps(out,sort_keys=True))
