"""Independent E4 audit on official CSV copies; never connects to WRDS."""
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, date
from itertools import combinations
from pathlib import Path

P = Path(__file__).parent
NAMES = ('VXN', 'RVX', 'OVX', 'GVZ')
def rows(name):
    path = P / 'vera_sources' / f'{name}_History.csv'
    raw = list(csv.DictReader(path.open()))
    out = [(datetime.strptime(r['DATE'], '%m/%d/%Y').date(), float(r.get('CLOSE', r.get(name)))) for r in raw]
    assert len(out) == len(set(d for d, _ in out)) and out == sorted(out)
    assert all(math.isfinite(v) and v > 0 for _,v in out)
    return path, out
def csvrows(name):
    return list(csv.DictReader((P/name).open()))

source = {r['index']:r for r in json.loads((P/'source_manifest.json').read_text())}
catalog = []
counts = []
for name in NAMES:
    path, data = rows(name)
    s = source[name]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == s['sha256']
    assert (len(data), str(data[0][0]), str(data[-1][0])) == (s['valid_rows'], s['first_date'], s['last_date'])
    candidates = []
    for i,(day,value) in enumerate(data):
        one = i >= 1 and value/data[i-1][1] >= 1.35
        five = i >= 5 and value/data[i-5][1] >= 1.5
        if one or five: candidates.append(i)
    onsets = []
    for i in candidates:
        if not onsets or (data[i][0]-data[onsets[-1]][0]).days > 30:
            onsets.append(i)
    eligible = 0
    for j,i in enumerate(onsets):
        day = data[i][0]
        end = data[i+119][0] if i+119 < len(data) else None
        next_before = bool(end and j+1 < len(onsets) and data[onsets[j+1]][0] <= end)
        pre = i >= 250
        baseline = pre and sum(v for _,v in data[i-250:i]) > 0
        flags = [int(pre), int(bool(end)), int(i+10 < len(data)), int(baseline), int(next_before)]
        good = int(all(flags[:4]) and not flags[4]); eligible += good
        catalog.append(dict(event_id=f'{name}-{day}', index=name, onset=str(day), end_120=str(end) if end else '',
                            censor_end=str(data[min(i+119,len(data)-1)][0]), pre_250=str(flags[0]),
                            full_120=str(flags[1]), peak_11=str(flags[2]), baseline_positive=str(flags[3]),
                            next_onset_before_end=str(flags[4]), potential_eligible=str(good)))
    counts.append(dict(index=name,candidate_triggers=str(len(candidates)),accepted_onsets=str(len(onsets)),potential_eligible=str(eligible)))
assert counts == csvrows('achievable_counts.csv')
saved_catalog = csvrows('event_catalog.csv')
assert [{k:v for k,v in r.items() if k != 'cluster_id'} for r in saved_catalog] == catalog
events = [r for r in catalog if r['potential_eligible']=='1']
edges = set(); pair_counts = Counter(); same_counts = Counter()
for i,j in combinations(range(len(events)),2):
    a,b = events[i],events[j]
    key = tuple(sorted((a['index'],b['index']),key=NAMES.index))
    if a['onset']==b['onset']: same_counts[key]+=1
    if max(a['onset'],b['onset']) <= min(a['end_120'],b['end_120']):
        pair_counts[key]+=1; edges.add((i,j))
matrix=[]
for i,a in enumerate(NAMES):
    for b in NAMES[i:]:
        matrix.append(dict(index_a=a,index_b=b,same_onset_pairs=str(same_counts[(a,b)]),overlapping_window_pairs=str(pair_counts[(a,b)])))
assert matrix == csvrows('overlap_matrix.csv')
parent=list(range(len(events)))
def root(i):
    while parent[i]!=i: i=parent[i]
    return i
for i,j in edges: parent[root(j)]=root(i)
groups={}
for i,e in enumerate(events): groups.setdefault(root(i),[]).append(e)
groups=sorted(groups.values(),key=lambda g:min(e['onset'] for e in g))
clusters=[]
for k,g in enumerate(groups,1):
    ids={e['event_id'] for e in g}; names={e['index'] for e in g}
    clusters.append(dict(cluster_id=f'C{k:03d}',start=min(e['onset'] for e in g),end=max(e['end_120'] for e in g),
                         event_count=str(len(g)),index_count=str(len(names)),indices=';'.join(sorted(names)),
                         event_ids=';'.join(e['event_id'] for e in catalog if e['event_id'] in ids)))
assert clusters == csvrows('clusters.csv')
assert (len(events),len(groups),sum(same_counts.values()),sum(pair_counts.values()))==(35,13,7,31)
summary=json.loads((P/'inventory_summary.json').read_text())
assert (summary['potential_events'],summary['date_clusters'],summary['source_indices'])==(len(events),len(groups),len({e['index'] for e in events}))
for filename, digest in json.loads((P/'wrds_handoff_freeze.json').read_text())['files'].items():
    assert hashlib.sha256((P/filename).read_bytes()).hexdigest()==digest
for line in (P/'freeze_manifest.txt').read_text().splitlines():
    words=line.split()
    if words and words[0] in ('event_inventory_spec.md','build_inventory.py'):
        assert hashlib.sha256((P/words[0]).read_bytes()).hexdigest()==words[1]
release=P.parents[1]/'stages/S05_release'
for line in (release/'manifest.sha256').read_text().splitlines():
    digest,name=line.split(maxsplit=1)
    assert hashlib.sha256((release/name).read_bytes()).hexdigest()==digest
result={'sources':[{**s,'fresh_sha256':hashlib.sha256((P/'vera_sources'/f'{s["index"]}_History.csv').read_bytes()).hexdigest()} for s in source.values()],
        'counts':counts,'eligible':len(events),'clusters':len(groups),'same_onset_pairs':sum(same_counts.values()),
        'overlap_pairs':sum(pair_counts.values()),'matrix':matrix,'cluster_sizes':[len(g) for g in groups],
        'spec_code_freeze_matches':True,'wrds_handoff_freeze_matches':True,'protected_release_manifest_matches':True}
(P/'vera_verification_evidence.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ('counts','eligible','clusters','same_onset_pairs','overlap_pairs','cluster_sizes')},indent=2))
