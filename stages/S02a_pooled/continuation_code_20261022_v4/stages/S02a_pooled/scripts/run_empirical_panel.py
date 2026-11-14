"""Run one frozen S02a empirical panel in the isolated continuation workspace."""
import json
import fcntl
import sys
import time
from pathlib import Path

import parallel_pipeline as runtime

p = runtime.p
out = p.OUT

def main():
    h = int(sys.argv[1])
    maturity, cohort, mode = sys.argv[2:5]
    assert h in p.HORIZONS and maturity in p.MATS
    assert cohort in ('full', 'common') and mode in ('event', 'group')
    key = f'empirical_{h}_{maturity}_{cohort}_{mode}'
    receipt = out / 'continuation_receipts' / f'{key}.json'
    lock_dir = out / 'panel_locks'
    lock_dir.mkdir(exist_ok=True)
    lock = (lock_dir / f'{key}.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX)
    if receipt.exists():
        print(json.dumps({'key':key,'status':'already_complete','receipt':str(receipt)}),flush=True)
        return
    ts, events = p.inputs()
    panels = {m: p.panels(ts, events, m, h) for m in p.MATS}
    common = sorted(set.intersection(*(set(panels[m][0]) for m in p.MATS)))
    panel, intervals, _ = panels[maturity]
    ids = sorted(panel) if cohort == 'full' else common
    assert ids, f'coverage failure for {key}'
    paired_key = f'paired_{h}_{mode}' if cohort == 'common' else None
    started = time.perf_counter()
    meta = runtime.fit_unit(panel, ids, intervals, key, 499, mode, paired_key)
    folder = out / 'units' / key
    expected = {
        f'r{rep}_b{baseline}_s{start}.json'
        for rep in range(-1, 499)
        for baseline in (0, 1)
        for start in p.core.STARTS
    }
    actual = {path.name for path in folder.glob('r*.json')}
    assert actual == expected, (len(actual), len(expected), sorted(expected-actual)[:5])
    record = dict(**meta, wall_seconds=time.perf_counter()-started,
                  fit_files=len(actual), panel_file=(folder/'panel.json').exists(),
                  frozen_core='stages/S01_protocol/attempt_02/protocol_core.py',
                  runtime='parallel_pipeline.py with copied pipeline time guard disabled')
    receipt.parent.mkdir(parents=True, exist_ok=True)
    p.write(receipt, record)
    print(json.dumps(record), flush=True)

if __name__ == '__main__':
    main()
