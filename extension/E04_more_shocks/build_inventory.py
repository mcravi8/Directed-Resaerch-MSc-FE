import csv, hashlib, json, math
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAMES = ('VXN', 'RVX', 'OVX', 'GVZ')
TH1, TH5 = math.log1p(7/20), math.log1p(10/20)

def put(name, rows, fields):
    with (ROOT/name).open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

def read(name):
    p=ROOT/'sources'/f'{name}_History.csv'
    raw=list(csv.DictReader(p.open()))
    rows=[]; bad=[]
    for k,r in enumerate(raw, start=2):
        try:
            d=date.fromisoformat(r['DATE']) if '-' in r['DATE'] else __import__('datetime').datetime.strptime(r['DATE'],'%m/%d/%Y').date()
            v=float(r['CLOSE'] if 'CLOSE' in r else r[name])
            if not math.isfinite(v) or v<=0: raise ValueError('nonpositive/nonfinite')
            rows.append((d,v))
        except (ValueError,KeyError) as e: bad.append((k,str(e)))
    rows.sort()
    if len({x[0] for x in rows}) != len(rows): raise ValueError(f'{name}: duplicate dates')
    return p,rows,bad

catalog=[]; counts=[]; files=[]
for name in NAMES:
    p,rows,bad=read(name); files.append({'index':name,'file':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'raw_rows':len(rows)+len(bad),'valid_rows':len(rows),'invalid_rows':len(bad),'first_date':str(rows[0][0]),'last_date':str(rows[-1][0])})
    last=None; candidates=0; accepted=0; eligible=0
    for i,(d,v) in enumerate(rows):
        trig=(i>=1 and math.log(v/rows[i-1][1])>=TH1) or (i>=5 and math.log(v/rows[i-5][1])>=TH5)
        if not trig: continue
        candidates+=1
        if last is not None and (d-last).days<=30: continue
        last=d; accepted+=1
        full=i+119<len(rows); pre=i>=250; peak=i+10<len(rows)
        baseline=sum(x[1] for x in rows[i-250:i])/250 if pre else None
        catalog.append({'event_id':f'{name}-{d.isoformat()}','index':name,'onset':str(d),'end_120':str(rows[i+119][0]) if full else '', 'censor_end':str(rows[min(i+119,len(rows)-1)][0]),'pre_250':int(pre),'full_120':int(full),'peak_11':int(peak),'baseline_positive':int(baseline is not None and baseline>0),'next_onset_before_end':'','potential_eligible':0})
    counts.append({'index':name,'candidate_triggers':candidates,'accepted_onsets':accepted,'potential_eligible':0})

by_index=defaultdict(list)
for e in catalog: by_index[e['index']].append(e)
for name,ev in by_index.items():
    for k,e in enumerate(ev):
        nxt=ev[k+1]['onset'] if k+1<len(ev) else ''
        e['next_onset_before_end']=int(bool(e['end_120'] and nxt and nxt<=e['end_120']))
        e['potential_eligible']=int(all(int(e[x]) for x in ('pre_250','full_120','peak_11','baseline_positive')) and not e['next_onset_before_end'])
    next(c for c in counts if c['index']==name)['potential_eligible']=sum(e['potential_eligible'] for e in ev)

eligible=[e for e in catalog if e['potential_eligible']]
parent=list(range(len(eligible)))
def find(x):
    while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
    return x
def union(a,b): parent[find(a)]=find(b)
matrix=[]
for ai,a in enumerate(NAMES):
    for bi,b in enumerate(NAMES[ai:],ai):
        same=overlap=0
        for i,x in enumerate(eligible):
            if x['index']!=a: continue
            for j,y in enumerate(eligible):
                if y['index']!=b or (a==b and j<=i): continue
                if x['onset']==y['onset']: same+=1
                if x['onset']<=y['end_120'] and y['onset']<=x['end_120']:
                    overlap+=1; union(i,j)
        matrix.append({'index_a':a,'index_b':b,'same_onset_pairs':same,'overlapping_window_pairs':overlap})
groups=defaultdict(list)
for i,e in enumerate(eligible): groups[find(i)].append(e)
clusters=[]
for j,g in enumerate(sorted(groups.values(),key=lambda g:min(e['onset'] for e in g)),1):
    cid=f'C{j:03d}'
    clusters.append({'cluster_id':cid,'start':min(e['onset'] for e in g),'end':max(e['end_120'] for e in g),'event_count':len(g),'index_count':len({e['index'] for e in g}),'indices':';'.join(sorted({e['index'] for e in g})),'event_ids':';'.join(e['event_id'] for e in g)})
    for e in g:e['cluster_id']=cid
for e in catalog:e.setdefault('cluster_id','')
put('event_catalog.csv',catalog,list(catalog[0]))
put('achievable_counts.csv',counts,list(counts[0]))
put('overlap_matrix.csv',matrix,list(matrix[0]))
put('clusters.csv',clusters,list(clusters[0]) if clusters else ['cluster_id','start','end','event_count','index_count','indices','event_ids'])
(ROOT/'source_manifest.json').write_text(json.dumps(files,indent=2)+'\n')
summary={'potential_events':len(eligible),'date_clusters':len(clusters),'source_indices':len({e['index'] for e in eligible}),'source_files':files}
(ROOT/'inventory_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary))
