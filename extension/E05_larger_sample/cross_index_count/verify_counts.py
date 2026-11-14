import csv,json
from collections import Counter,defaultdict
from pathlib import Path
p=Path(__file__).resolve().parent
def read(n):
    with (p/n).open(newline='') as f:return list(csv.DictReader(f))
events=read('events.csv'); dates=read('date_merged.csv'); comps=read('full_footprint_components.csv'); per=read('per_index_counts.csv')
eligible=[r for r in events if r['eligible']=='1']
assert len(events)==sum(int(r['accepted_onsets']) for r in per)
assert len(eligible)==sum(int(r['eligible_events']) for r in per)
assert Counter(r['index'] for r in eligible)=={r['index']:int(r['eligible_events']) for r in per}
assert len(dates)==len({r['onset'] for r in eligible})
assert sum(int(r['index_count']) for r in dates)==len(eligible)
by_date=defaultdict(list)
for r in eligible:by_date[r['onset']].append(r)
for d in dates:
    g=by_date[d['onset']]
    assert len(g)==int(d['index_count'])
    assert set(d['event_ids'].split('|'))=={r['event_id'] for r in g}
    assert {r['component_id'] for r in g}=={d['component_id']}
assert sum(int(c['event_count']) for c in comps)==len(eligible)
assert sum(int(c['date_count']) for c in comps)>=len(dates) # dates may span clusters only if logic faulty; checked below
assert len({r['event_id'] for r in eligible})==len(eligible)
for c in comps:
    members=[r for r in eligible if r['component_id']==c['component_id']]
    assert len(members)==int(c['event_count'])
    assert len({r['onset'] for r in members})==int(c['date_count'])
    assert c['footprint_first']==min(r['footprint_first'] for r in members)
    assert c['footprint_last']==max(r['footprint_last'] for r in members)
    assert set(c['event_ids'].split('|'))=={r['event_id'] for r in members}
    intervals=sorted((r['footprint_first'],r['footprint_last']) for r in members)
    end=intervals[0][1]
    for a,b in intervals[1:]:
        assert a<=end,(c['component_id'],a,end)
        end=max(end,b)
for a,b in zip(comps,comps[1:]):assert a['footprint_last']<b['footprint_first']
summary=json.loads((p/'summary.json').read_text())
assert summary['eligible_event_rows']==len(eligible)
assert summary['date_merged_events']==len(dates)
assert summary['full_footprint_components']==len(comps)
print(json.dumps({'verified':True,'eligible':len(eligible),'dates':len(dates),'components':len(comps)}))
