"""Independent consistency checks on published summary tables, not full refitting."""
import csv
import json
import math
from pathlib import Path

root=Path(__file__).resolve().parents[1]
def rows(p): return list(csv.DictReader((root/p).open()))
e2='extension/E02_sample_size/'
rec=rows(e2+'recognition.csv')
assert len(rec)==80
for r in rec:
    assert int(r['completed'])==200
    assert math.isclose(float(r['rate']),int(r['correct'])/200)
pl=[r for r in rec if r['truth']=='PL' and r['design']=='calendar_joint' and r['start']=='5']
assert [int(r['correct']) for r in sorted(pl,key=lambda r:int(r['N']))]==[10,9,18,12,22]
for r in rows(e2+'N80.csv'): assert r['N80']=='not attained on tested grid through 320'
tests=rows('extension/E03_forecast/tests.csv');assert len(tests)==24
for t in tests:
    assert float(t['mean_d'])<0
    assert math.isclose(float(t['mean_d']),float(t['har_mse'])-float(t['harx_mse']),abs_tol=1e-12)
    assert t['positive_claim'].lower()=='false'
primary=tests[0];assert int(primary['K'])==15 and int(primary['paired_n'])==2512
e5=root/'extension/E05_larger_sample/fresh_public_v5'
pool=json.loads((e5/'pooled_summary.json').read_text())
assert sum(pool['index_event_counts'].values())==84
assert pool['valid_draws']==666 and pool['failed_draws']==333
assert pool['saved_draws']==pool['valid_draws']+pool['failed_draws']==999
assert pool['base_clusters']==7
cat=json.loads((root/'extension/E05_larger_sample/cross_index_count/summary.json').read_text())
assert cat['eligible_event_rows']==cat['date_merged_events']+cat['same_date_extra_rows']==64
assert cat['date_merged_events']==50 and cat['full_footprint_components']==1
print('Saved identification, forecast, pooled-draw and date-merging summaries reconcile.')
