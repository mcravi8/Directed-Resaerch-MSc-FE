"""Bounded offline checks; never starts empirical or bootstrap production jobs."""
import os
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parents[1]
tmp=root/'.tmp';tmp.mkdir(exist_ok=True)
env=dict(os.environ,TMPDIR=str(tmp),PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR=str(tmp/'mpl'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
jobs=[
 ['stages/S01_protocol/attempt_02/test_protocol.py'],
 ['stages/S02a_pooled/scripts/test_stage.py'],
 ['extension/E02_sample_size/code/test_noise.py'],
 ['-m','unittest','discover','-s','extension/E04_more_shocks','-p','test*offline.py','-v'],
 ['-m','unittest','discover','-s','extension/E04_more_shocks','-p','test_wrds_gate_v2.py','-v'],
 ['tools/check_saved_results.py'],
]
for job in jobs:
    print('CHECK', ' '.join(job),flush=True)
    subprocess.run([sys.executable,*job],cwd=root,env=env,check=True,timeout=180)
print('All six offline check groups passed.')
