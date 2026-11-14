import csv, hashlib, json, math
from collections import defaultdict
from datetime import datetime, date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAMES = ('VXN','RVX','OVX','GVZ','VXEEM','VXAPL','VXAZN','NIKKEI225VI')
URLS = {n: ('https://indexes.nikkei.co.jp/nkave/historical/nikkei_stock_average_vi_daily_en.csv' if n=='NIKKEI225VI' else f'https://cdn-api.cboe.com/api/global/us_indices/daily_prices/{n}_History.csv') for n in NAMES}

def write_csv(name, rows, fields):
    with (ROOT/name).open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

def read_source(name):
    p=ROOT/'sources'/f'{name}.csv'
    with p.open(encoding='cp932' if name=='NIKKEI225VI' else 'utf-8-sig', newline='') as f: raw=list(csv.DictReader(f))
    rows=[]; invalid=[]
    for k,r in enumerate(raw, 2):
        try:
            ds=r.get('DATE') or r.get('Date of Data')
            d=datetime.strptime(ds, '%Y/%m/%d' if '/' in ds and name=='NIKKEI225VI' else '%m/%d/%Y' if '/' in ds else '%Y-%m-%d').date()
            v=float(r.get('CLOSE') or r.get('Close') or r.get(name))
            if not math.isfinite(v) or v<=0: raise ValueError('nonfinite or nonpositive')
            rows.append((d,v))
        except (ValueError,TypeError,KeyError) as e: invalid.append({'line':k,'reason':str(e)})
    rows.sort()
    if len({d for d,v in rows})!=len(rows): raise ValueError(f'duplicate dates {name}')
    if not rows: raise ValueError(f'empty {name}')
    return rows,{'index':name,'source_url':URLS[name],'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'raw_rows':len(raw),'valid_rows':len(rows),'invalid_rows':len(invalid),'first_date':str(rows[0][0]),'last_date':str(rows[-1][0]),'invalid':invalid}

events=[]; manifest=[]; per=[]
for name in NAMES:
    rows,meta=read_source(name); manifest.append(meta)
    onset_indices=[]; candidates=0
    for i,(d,v) in enumerate(rows):
        trig=(i>=1 and v/rows[i-1][1]>=1.35) or (i>=5 and v/rows[i-5][1]>=1.5)
        if not trig: continue
        candidates+=1
        if onset_indices and (d-rows[onset_indices[-1]][0]).days<=30: continue
        onset_indices.append(i)
    eligible=0
    for j,i in enumerate(onset_indices):
        d=rows[i][0]; pre=i>=250; peak=i+10<len(rows)
        anchor=max(range(i,min(i+11,len(rows))),key=lambda z: rows[z][1]) if peak else None
        full=anchor is not None and anchor+119<len(rows)
        end=rows[anchor+119][0] if full else None
        next_onset=rows[onset_indices[j+1]][0] if j+1<len(onset_indices) else None
        bad=[]
        if not pre: bad.append('insufficient_pre_250')
        if not peak: bad.append('incomplete_peak_11')
        if not full: bad.append('incomplete_response_120')
        baseline=sum(v for _,v in rows[i-250:i])/250 if pre else None
        if pre and baseline<=0: bad.append('nonpositive_baseline')
        if full and next_onset is not None and next_onset<=end: bad.append('next_onset_before_response_end')
        ok=not bad; eligible+=ok
        events.append({'event_id':f'{name}:{d}','index':name,'onset':str(d),'anchor':str(rows[anchor][0]) if anchor is not None else '', 'footprint_first':str(rows[i-250][0]) if pre else '', 'footprint_last':str(end) if end else '', 'next_onset':str(next_onset) if next_onset else '', 'eligible':int(ok),'exclusion':'|'.join(bad),'date_id':'','component_id':''})
    per.append({'index':name,'candidate_triggers':candidates,'accepted_onsets':len(onset_indices),'eligible_events':eligible})

eligible=[e for e in events if e['eligible']]
by_date=defaultdict(list)
for e in eligible: by_date[e['onset']].append(e)
dates=[]
for k,(d,g) in enumerate(sorted(by_date.items()),1):
    did=f'D{k:03d}'
    for e in g: e['date_id']=did
    dates.append({'date_id':did,'onset':d,'index_count':len(g),'indices':'|'.join(sorted(e['index'] for e in g)),'event_ids':'|'.join(sorted(e['event_id'] for e in g)),'component_id':''})

# Inclusive interval sweep; transitivity follows from the running maximum end.
groups=[]; group=[]; max_end=''
for e in sorted(eligible,key=lambda x:(x['footprint_first'],x['footprint_last'],x['event_id'])):
    if group and e['footprint_first']>max_end:
        groups.append(group); group=[]; max_end=''
    group.append(e); max_end=max(max_end,e['footprint_last'])
if group: groups.append(group)
components=[]
for k,g in enumerate(groups,1):
    cid=f'C{k:03d}'
    for e in g: e['component_id']=cid
    components.append({'component_id':cid,'footprint_first':min(e['footprint_first'] for e in g),'footprint_last':max(e['footprint_last'] for e in g),'event_count':len(g),'date_count':len({e['onset'] for e in g}),'index_count':len({e['index'] for e in g}),'indices':'|'.join(sorted({e['index'] for e in g})),'date_ids':'|'.join(sorted({e['date_id'] for e in g})),'event_ids':'|'.join(sorted(e['event_id'] for e in g))})
for row in dates: row['component_id']=next(iter({e['component_id'] for e in by_date[row['onset']]}))
write_csv('events.csv',events,list(events[0]))
write_csv('date_merged.csv',dates,list(dates[0]))
write_csv('full_footprint_components.csv',components,list(components[0]))
write_csv('per_index_counts.csv',per,list(per[0]))
(ROOT/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
summary={'source_indices':len(NAMES),'candidate_triggers':sum(x['candidate_triggers'] for x in per),'accepted_onsets':sum(x['accepted_onsets'] for x in per),'eligible_event_rows':len(eligible),'date_merged_events':len(dates),'same_date_extra_rows':len(eligible)-len(dates),'full_footprint_components':len(components),'largest_component_events':max(c['event_count'] for c in components),'inferential_minimum_met':len(components)>=10}
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
write_csv('number_ledger.csv',[{'claim':k,'value':v,'evidence':'derived from events.csv, date_merged.csv, full_footprint_components.csv'} for k,v in summary.items()],['claim','value','evidence'])
print(json.dumps(summary,sort_keys=True))
