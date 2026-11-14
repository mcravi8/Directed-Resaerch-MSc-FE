"""Deterministic construction checks, separate from the observed-cohort evidence."""
import json
import numpy as np
import e2

seed=e2.core.seed('Extension','E2','noise_test_v2')
rng=np.random.default_rng(seed)
template=dict(id=0,event_ids=['a','b','c'],base=0,span=125,
              offsets={'a':0,'b':2,'c':4},
              masks={e:list(range(i,i+121)) for e,i in [('a',0),('b',2),('c',4)]})
meta=dict(templates=[template],runs=[list(range(125))],
          calendar_values={i:float(i*i) for i in range(125)})
noise,draws,boundaries=e2.calendar_panel(2,rng,np.random.default_rng(seed+1),meta)
record=draws[0]
assert len(boundaries)==1 and len(record['selected_ids'])==2 and len(record['excluded_ids'])==1
for event_index,eid in enumerate(record['selected_ids']):
    offset=record['offsets'][eid]
    expected=np.asarray(record['field'][offset:offset+121])-record['field'][offset]
    assert np.array_equal(noise[event_index],expected)
for t in range(125):
    covering=[(j,eid) for j,eid in enumerate(record['selected_ids'])
              if record['offsets'][eid]<=t<record['offsets'][eid]+121]
    if len(covering)==2:
        recovered=[noise[j,t-record['offsets'][eid]]+record['field'][record['offsets'][eid]]
                   for j,eid in covering]
        assert recovered[0]==recovered[1]==record['field'][t]

residual=np.arange(19*121,dtype=float).reshape(19,121)**2
independent,metadata=e2.independent_panel(2,np.random.default_rng(seed+2),residual)
assert independent.shape==(2,121)
assert (metadata['source_rows'][0],metadata['block_starts'][0])!=(metadata['source_rows'][1],metadata['block_starts'][1])
assert not np.array_equal(independent[0],independent[1])
report=dict(seed=seed,forced_overlap=True,boundary_selected=record['selected_ids'],
            boundary_excluded=record['excluded_ids'],shared_coordinates_checked=119,
            independent_source_rows=metadata['source_rows'],
            independent_block_starts=metadata['block_starts'],passed=True,
            interpretation='synthetic implementation test; observed cohort is audited separately')
e2.atomic(e2.OUT/'noise_design_checks_v2.json',report)
print(json.dumps(report,indent=2))
