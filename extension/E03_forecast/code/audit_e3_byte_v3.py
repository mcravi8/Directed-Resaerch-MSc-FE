"""Literal-byte E3 prefix/suffix audit; additive correction to audit_e3_v2.py.

Run from the Directed Research root. This deliberately uses the production
arithmetic for a strict reproducibility comparison; audit_e3_v2.py remains
the separate, numerically independent reconstruction.
"""
import csv
import hashlib
import json
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np
import pandas as pd

e = SourceFileLoader('e3_frozen_byte_audit', 'extension/E03_forecast/code/run_e3.py').load_module()
root = Path('extension/E03_forecast')
manifest = json.loads((root/'input_manifest.json').read_text())
assert e.sha(e.SPEC) == manifest['spec_sha256']
assert e.sha(e.__file__) == manifest['code_sha256']
for name, info in manifest['sources'].items():
    assert e.sha(root/'raw'/f'{name}.csv') == info['sha256']

with (root/'predictions.csv').open(newline='') as f:
    predictions = list(csv.DictReader(f))
panel = {name: e.parse(name)[:2] for name in e.NAMES}
vdates, vvalues = panel['VIX30D']
full_state = e.onset_features(vdates, vvalues)
full_rows = {name: e.make_rows(name, dates, vals, vdates, full_state)
             for name, (dates, vals) in panel.items()}
by_date = {}
for p in predictions:
    by_date.setdefault(p['date'], []).append(p)
dates = sorted(by_date)
audit_seed = e.seed('Extension', 'E3', 'truncation_audit')
rng = np.random.Generator(np.random.PCG64(audit_seed))
chosen = set(rng.choice(dates, size=min(200, len(dates)), replace=False))
for year in sorted({d[:4] for d in dates}):
    year_dates = [d for d in dates if d.startswith(year)]
    chosen.update((year_dates[0], year_dates[-1]))

def bits(x):
    return np.asarray(x, dtype='<f8').tobytes()

def hexbytes(x):
    return bits(x).hex()

def five(row, tau):
    c = row['core']
    return np.asarray([c[0], c[1], c[2], c[0]*np.exp(-c[2]/tau), c[0]*c[3]], dtype='<f8')

def forecast(row, model):
    har = float(row['har'] @ np.asarray(model['har']))
    harx = float(e.design([row], model['tau'])[0] @ np.asarray(model['harx']))
    return har, harx

input_rows = []
evidence = []
model_cache = {}
for ds in sorted(chosen):
    t = pd.Timestamp(str(ds))
    j = int(np.searchsorted(vdates, t))
    assert vdates[j] == t
    changed_vix = vvalues.copy()
    changed_vix[j+1:] = np.maximum(.01, changed_vix[j+1:] * rng.uniform(.5, 1.5, size=len(changed_vix)-j-1))
    prefix_state = e.onset_features(vdates[:j+1], vvalues[:j+1])
    changed_state = e.onset_features(vdates, changed_vix)
    input_rows.append(dict(date=ds, maturity='VIX30D', prefix_end=ds,
                           original_sha256=hashlib.sha256(vvalues.tobytes()).hexdigest(),
                           perturbed_sha256=hashlib.sha256(changed_vix.tobytes()).hexdigest(),
                           prefix_sha256=hashlib.sha256(vvalues[:j+1].tobytes()).hexdigest(),
                           seed=audit_seed))
    row_cache = {}
    for p in by_date[ds]:
        name = p['maturity']
        if name not in row_cache:
            md, mv = panel[name]
            i = int(np.searchsorted(md, t))
            assert md[i] == t
            changed_mv = mv.copy()
            changed_mv[i+1:] = np.maximum(.01, changed_mv[i+1:] * rng.uniform(.5, 1.5, size=len(changed_mv)-i-1))
            prefix_rows = e.make_rows(name, md[:i+1], mv[:i+1], vdates[:j+1], prefix_state)
            # All monthly labels mature by C<t. Only the rows through t can
            # enter this vintage or its decision-date prediction. Build that
            # dependency slice from the changed full-series state.
            changed_rows = e.make_rows(name, md[:i+1], changed_mv[:i+1],
                                       vdates[:j+1], {k:(v[:j+1] if isinstance(v, np.ndarray) else v)
                                                      for k,v in changed_state.items()})
            row_cache[name] = (i, prefix_rows, changed_rows)
            input_rows.append(dict(date=ds, maturity=name, prefix_end=ds,
                                   original_sha256=hashlib.sha256(mv.tobytes()).hexdigest(),
                                   perturbed_sha256=hashlib.sha256(changed_mv.tobytes()).hexdigest(),
                                   prefix_sha256=hashlib.sha256(mv[:i+1].tobytes()).hexdigest(),
                                   seed=audit_seed))
        i, prefix_rows, changed_rows = row_cache[name]
        h = int(p['h'])
        C = pd.Timestamp(p['cutoff'])
        stored = json.loads((root/'monthly_models'/f"{p['model_id']}.json").read_text())['model']
        if p['model_id'] not in model_cache:
            model_cache[p['model_id']] = (e.month_model(prefix_rows, h, C),
                                         e.month_model(changed_rows, h, C))
        (prefix_model, pe), (changed_model, ce) = model_cache[p['model_id']]
        assert prefix_model is not None and changed_model is not None, (ds, name, h, pe, ce)
        full_row, prefix_row, changed_row = full_rows[name][i], prefix_rows[i], changed_rows[i]
        values = {}
        for tag, row, model in [('full', full_row, stored), ('prefix', prefix_row, prefix_model),
                                ('suffix', changed_row, changed_model)]:
            values[tag] = (five(row, model['tau']),
                           bits(np.asarray(model['har'])), bits(np.asarray(model['harx'])),
                           forecast(row, model))
        full, pre, suf = values['full'], values['prefix'], values['suffix']
        saved = (float(p['har']), float(p['harx']))
        checks = dict(
            features_prefix=bits(full[0]) == bits(pre[0]),
            features_suffix=bits(full[0]) == bits(suf[0]),
            vintage_prefix=stored['tau'] == prefix_model['tau'] and p['model_id'].encode() == f'{name}_{h}_{t.to_period("M")}'.encode(),
            vintage_suffix=stored['tau'] == changed_model['tau'],
            coefficients_prefix=full[1:3] == pre[1:3],
            coefficients_suffix=full[1:3] == suf[1:3],
            predictions_prefix=bits(full[3]) == bits(pre[3]),
            predictions_suffix=bits(full[3]) == bits(suf[3]),
            predictions_saved=bits(full[3]) == bits(saved),
        )
        evidence.append(dict(date=ds, maturity=name, h=h, model_id=p['model_id'],
            **{k:int(v) for k,v in checks.items()},
            full_features_hex=hexbytes(full[0]), prefix_features_hex=hexbytes(pre[0]), suffix_features_hex=hexbytes(suf[0]),
            full_har_coef_hex=full[1].hex(), prefix_har_coef_hex=pre[1].hex(), suffix_har_coef_hex=suf[1].hex(),
            full_harx_coef_hex=full[2].hex(), prefix_harx_coef_hex=pre[2].hex(), suffix_harx_coef_hex=suf[2].hex(),
            full_predictions_hex=hexbytes(full[3]), prefix_predictions_hex=hexbytes(pre[3]),
            suffix_predictions_hex=hexbytes(suf[3]), saved_predictions_hex=hexbytes(saved),
            prefix_tau=prefix_model['tau'], suffix_tau=changed_model['tau'], full_tau=stored['tau']))

with (root/'leakage_byte_v3_inputs.csv').open('w', newline='') as f:
    w=csv.DictWriter(f, fieldnames=list(input_rows[0]));w.writeheader();w.writerows(input_rows)
with (root/'leakage_byte_v3_evidence.csv').open('w', newline='') as f:
    w=csv.DictWriter(f, fieldnames=list(evidence[0]));w.writeheader();w.writerows(evidence)
columns=[k for k in evidence[0] if k.startswith(('features_', 'vintage_', 'coefficients_', 'predictions_')) and k not in ('predictions_saved',) and not k.endswith('_hex')]
columns.append('predictions_saved')
summary=dict(selected_dates=len(chosen), comparison_rows=len(evidence), seed=audit_seed,
             equal_counts={k:sum(r[k] for r in evidence) for k in columns},
             failed_rows={k:[f"{r['date']}:{r['maturity']}:{r['h']}" for r in evidence if not r[k]] for k in columns})
summary['status']='passed' if all(v==len(evidence) for v in summary['equal_counts'].values()) else 'failed'
e.atom(root/'leakage_byte_v3_summary.json', summary)
print(json.dumps({k:v for k,v in summary.items() if k!='failed_rows'}))
if summary['status']!='passed':sys.exit(1)
