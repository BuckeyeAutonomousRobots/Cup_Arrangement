"""One authorized seed12 continuation, independent 15-minute outer watchdog."""
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'models/policy100/runs/SinglePositionACT_seed12_continue_20261006_01'
assert not OUT.exists(),'Refusing duplicate output'
with OUT.with_suffix('.claim').open('x') as claim:
    command=['timeout','--signal=TERM','--kill-after=30s','900s','nice','-n','10',
             str(ROOT/'models/act_v1/.venv/bin/python'),'-u','models/policy100/continue_single_position_once.py']
    environment=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',
                     PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1')
    with OUT.with_suffix('.log').open('x') as log:
        process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                 start_new_session=True,env=environment)
    record=dict(watchdog_pid=process.pid,command=command,output=str(OUT),offline_only=True,
                budget='4000additional updates/600training seconds;900seconds outer watchdog for setup/evaluation/cleanup')
    json.dump(record,claim,indent=2);claim.flush();os.fsync(claim.fileno())
    print(json.dumps(record))
