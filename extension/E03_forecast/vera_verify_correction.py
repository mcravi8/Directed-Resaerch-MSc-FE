"""Read-only cross-check of the corrected E3 evidence and report inventory."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

root = Path(__file__).parent
read = lambda name: list(csv.DictReader((root / name).open(newline="")))
digest = lambda name: hashlib.sha256((root / name).read_bytes()).hexdigest()
evidence = read("leakage_byte_v3_evidence.csv")
old = read("leakage_audit_v2.csv")
pred = read("predictions.csv")
tests = read("tests.csv")
summary = json.loads((root / "leakage_byte_v3_summary.json").read_text())
inputs = read("leakage_byte_v3_inputs.csv")
assert len(evidence) == 1140 and len({r['date'] for r in evidence}) == 253
assert len({r['model_id'] for r in evidence}) == 797
key = lambda r: (r['date'],r['maturity'],int(r['h']))
assert len({key(r) for r in evidence}) == len(evidence)
assert {key(r) for r in evidence} == {key(r) for r in old}
assert len(inputs) == sum(1 + len({r['maturity'] for r in evidence if r['date']==date})
                          for date in {r['date'] for r in evidence})
assert {r['date'] for r in inputs} == {r['date'] for r in evidence}
assert all(r['original_sha256'] != r['perturbed_sha256'] for r in inputs)
dates = sorted({r['date'] for r in pred})
seedbytes = json.dumps([5902026,'Extension','E3','truncation_audit'],separators=(',',':')).encode()
seed = int.from_bytes(hashlib.sha256(seedbytes).digest()[:8],'big')
rng = np.random.Generator(np.random.PCG64(seed))
selected = set(rng.choice(dates,size=min(200,len(dates)),replace=False))
for year in {d[:4] for d in dates}:
    year_dates = [d for d in dates if d.startswith(year)]
    selected.update((year_dates[0],year_dates[-1]))
assert selected == {r['date'] for r in evidence}
assert summary['seed'] == seed and summary['selected_dates'] == len(selected)
assert summary['comparison_rows'] == len(evidence) and summary['status'] == 'passed'
triples = {
    'features': ('full_features_hex','prefix_features_hex','suffix_features_hex'),
    'har_coef': ('full_har_coef_hex','prefix_har_coef_hex','suffix_har_coef_hex'),
    'harx_coef': ('full_harx_coef_hex','prefix_harx_coef_hex','suffix_harx_coef_hex'),
    'predictions': ('full_predictions_hex','prefix_predictions_hex','suffix_predictions_hex','saved_predictions_hex'),
    'tau': ('full_tau','prefix_tau','suffix_tau'),
}
for r in evidence:
    for group,cols in triples.items():
        assert len({r[c] for c in cols}) == 1, (key(r),group)
    assert r['model_id'] == f"{r['maturity']}_{r['h']}_{r['date'][:7]}"
    for col in summary['equal_counts']:
        assert r[col] == '1', (key(r),col)
for col in summary['equal_counts']:
    assert summary['equal_counts'][col] == len(evidence) and summary['failed_rows'][col] == []
primary = [r for r in pred if r['maturity']=='VIX30D' and r['h']=='10']
active = [r for r in primary if r['active']=='1']
assert len(active) == 2512 and (min(r['date'] for r in active),max(r['date'] for r in active)) == ('2001-09-17','2025-09-18')
assert len(primary) == 6772 and (min(r['date'] for r in primary),max(r['date'] for r in primary)) == ('2000-01-03','2026-10-16')
assert len(pred) == 29272 and len(tests) == 24
assert digest('predictions.csv') == 'b93a568a0aa299cfd25786e9c129e99205f26c71451f7e06173b8b3be439bc33'
assert digest('tests.csv') == '7e6083f97a216b3b1d030f52bdfd2327bdaf95bcbcc230303c1e24061fb946e3'
checks = {
    'keys': len(evidence), 'dates': len(selected), 'vintages':len({r['model_id'] for r in evidence}),
    'strict_categories': len(summary['equal_counts']), 'active_count':len(active),
    'active_first':min(r['date'] for r in active),'active_last':max(r['date'] for r in active),
    'full_grid_count':len(primary),'full_grid_first':min(r['date'] for r in primary),
    'full_grid_last':max(r['date'] for r in primary),'forecast_rows':len(pred),
    'test_slots':len(tests),'predictions_sha256':digest('predictions.csv'),
    'tests_sha256':digest('tests.csv'), 'evidence_sha256':digest('leakage_byte_v3_evidence.csv'),
    'inputs_sha256':digest('leakage_byte_v3_inputs.csv'),
    'summary_sha256':digest('leakage_byte_v3_summary.json')}
(root/'vera_correction_independent.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks))
