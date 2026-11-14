"""One clean full-wall 2OU timing replication; preserves the resumed grid separately.

The inherited 2OU fit units completed across two commands, so their recorded
4608.9 seconds do not cover generation and the first batch of fits. A clean
measurement with the same frozen seed and scientific code is needed to fill
the benchmark T_2OU field. It runs after empirical computation finishes.
"""
import os
os.environ['S02A_WORKERS']='8'
from pathlib import Path
import parallel_pipeline as runtime


def main():
    p = runtime.p
    original = p.OUT
    p.OUT = original / 'timing_remeasurement_2OU'
    p.OUT.mkdir(parents=True, exist_ok=True)
    p.fit_unit = runtime.fit_unit
    p.pilot('2OU')


if __name__ == '__main__':
    main()
