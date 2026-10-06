"""Start the explicitly requested two-model offline run exactly once."""
import json
import os
from pathlib import Path
import subprocess as sp

root=Path(__file__).resolve().parents[2]
runs=root/'models/policy100/runs';runs.mkdir(exist_ok=True)
name='ACT_Diffusion_100_20261004'
out=runs/name
assert not out.exists(),'Run exists; do not relaunch'
with (runs/(name+'.claim')).open('x') as claim:
    cmd=['timeout','--signal=TERM','3600s','nice','-n','10',str(root/'models/act_v1/.venv/bin/python'),'-u',
         'models/policy100/train_compare.py','--output',str(out),'--steps','5000','--deadline','2026-10-04T23:55:00-04:00']
    with (runs/(name+'.log')).open('x') as log:
        proc=sp.Popen(cmd,cwd=root,stdin=sp.DEVNULL,stdout=log,stderr=sp.STDOUT,start_new_session=True)
    record={'launcher_pid':proc.pid,'command':cmd,'output':str(out),'offline_only':True}
    json.dump(record,claim,indent=2);claim.flush();os.fsync(claim.fileno())
    print(json.dumps(record))
