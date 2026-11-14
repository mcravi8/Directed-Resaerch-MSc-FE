"""Read-only replication audit and prespecified secondary diagnostics."""
import csv, json, statistics
from collections import defaultdict
import e2

e2.check()
cal=json.loads((e2.OUT/'calibration.json').read_text())
templates=cal['templates'];anchors=cal['anchors']
pairs=[(a,b) for i,a in enumerate(cal['ids']) for b in cal['ids'][i+1:]]
overlap=[(a,b) for a,b in pairs if max(anchors[a],anchors[b])<=min(anchors[a]+120,anchors[b]+120)]
assert len(templates)==19 and not overlap
assert all(len(t['event_ids'])==1 for t in templates)
observed=dict(event_count=len(cal['ids']),component_count=len(templates),
              pair_count=len(pairs),overlap_pair_count=len(overlap),
              interpretation='observed eligible cohort; separate from synthetic fixture',
              components=[dict(component_id=t['id'],event_ids=t['event_ids'],anchor_index=t['base'],last_index=t['base']+t['span']-1) for t in templates])
e2.atomic(e2.OUT/'observed_component_audit.json',observed)

groups=defaultdict(list);fitdiag=defaultdict(lambda:dict(cases=0,fit_failures=0,bound_contacts=0,ties=0));errors=[];count=0
for truth in e2.TRUTHS:
 for N in e2.NS:
  for design in e2.DESIGNS:
   for rep in range(200):
    p=e2.unit_path(truth,N,design,rep)
    if not p.exists():errors.append(('missing',str(p)));continue
    unit=json.loads(p.read_text());count+=1
    if unit['status']!='complete':errors.append(('error',str(p)));continue
    assert unit['unit_id']==f'{truth}/{N}/{design}/{rep}'
    assert unit['spec_hash']==e2.sha(e2.SPEC)
    assert unit['input_manifest_hash']==e2.sha(e2.OUT/'input_manifest.json')
    assert unit['code_hash']==e2.sha(e2.__file__)
    assert set(unit['starts'])=={'1','5'}
    for start in e2.STARTS:
        case=unit['starts'][str(start)]
        for family in e2.TRUTHS:
            d=fitdiag[(truth,N,design,start,family)];d['cases']+=1
            best=case['models'][family]['best']
            d['fit_failures']+=int(best is None)
            d['bound_contacts']+=int(best is not None and any(case['models'][family]['best']['active_mask']))
            d['ties']+=int(family in case['ties'] and len(case['ties'])>1)
    if design=='calendar_joint':
        assert len(unit['draws'])==unit['component_draw_count']
        assert sum(len(d['selected_ids']) for d in unit['draws'])==N
        assert len({d['source_component'] for d in unit['draws']})==unit['distinct_source_component_count']
        for d in unit['draws']:
            assert len(d['selected_ids'])==d['k']
            assert set(d['selected_ids'])<=set(d['candidate_ids'])
            assert d['inclusion_probability']==d['k']/d['m']
            template=templates[d['source_component']]
            assert all(d['offsets'][x]==template['offsets'][x] and d['masks'][x]==template['masks'][x] for x in d['selected_ids'])
    else:
        assert len(unit['draws']['source_rows'])==N
        assert len(unit['draws']['block_starts'])==N
    groups[(truth,N,design)].append((unit['component_draw_count'],unit['distinct_source_component_count']))
assert not errors,errors[:5]
assert count==8000,count
rows=[]
for (truth,N,design),v in sorted(groups.items()):
    assert len(v)==200
    draws=[x[0] for x in v];distinct=[x[1] for x in v if x[1] is not None]
    rows.append(dict(truth=truth,N=N,design=design,replications=len(v),
                     component_draw_min=min(draws),component_draw_mean=statistics.mean(draws),component_draw_max=max(draws),
                     distinct_source_min=min(distinct) if distinct else '',
                     distinct_source_mean=statistics.mean(distinct) if distinct else '',
                     distinct_source_max=max(distinct) if distinct else ''))
with (e2.OUT/'component_distribution.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
diagnostics=[dict(truth=t,N=n,design=d,start=s,family=f,**v)
             for (t,n,d,s,f),v in sorted(fitdiag.items())]
with (e2.OUT/'fit_diagnostics.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=diagnostics[0]);writer.writeheader();writer.writerows(diagnostics)

with (e2.OUT/'N80_components.csv').open(newline='') as f: components=list(csv.DictReader(f))
secondary=[]
for design in e2.DESIGNS:
 for start in e2.STARTS:
    qualifying=[]
    for N in e2.NS:
        rows=[r for r in components if r['design']==design and int(r['N'])==N and int(r['start'])==start]
        assert len(rows)==4
        good=all((int(r['count'])>=160 if r['estimand']=='PL_recognition' else int(r['count'])<=20) for r in rows)
        qualifying.append(good)
    hit=next((str(N) for N,good in zip(e2.NS,qualifying) if good),'not attained on tested grid through 320')
    secondary.append(dict(design=design,start=start,secondary_N80=hit,
                          qualifying_N=';'.join(str(N) for N,good in zip(e2.NS,qualifying) if good)))
with (e2.OUT/'N80_start_diagnostics.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=secondary[0]);writer.writeheader();writer.writerows(secondary)

with (e2.OUT/'boundary_frequencies.csv').open('w',newline='') as f:
    writer=csv.writer(f);writer.writerow(['truth','N','design','event_id','candidate_count','selected_count','selected_frequency','mean_conditional_inclusion'])
e2.atomic(e2.OUT/'audit_handoff.json',dict(units_checked=count,component_cells=len(groups),
    observed_overlap_pairs=len(overlap),boundary_overshoots=0,
    boundary_frequency_status='vacuous: all 19 observed templates contain one event',
    fit_cases_checked=count*2,all_hashes_match=True))
print(json.dumps(dict(units_checked=count,component_cells=len(groups),overlap_pairs=len(overlap),secondary=secondary),indent=2))
