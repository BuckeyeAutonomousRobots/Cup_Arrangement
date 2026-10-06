"""Detached once-only continuation with an independent one-hour watchdog."""
import json
import os
import subprocess as sp
from continue_act100 import OUT, t

assert not OUT.exists(),'Existing output; no relaunch'
with OUT.with_suffix('.claim').open('x') as claim:
    cmd=['timeout','--signal=TERM','--kill-after=120s','3600s','nice','-n','10',
         str(t.ROOT/'models/act_v1/.venv/bin/python'),'-u','models/policy100/continue_act100.py']
    with OUT.with_suffix('.log').open('x') as log:
        p=sp.Popen(cmd,cwd=t.ROOT,stdin=sp.DEVNULL,stdout=log,stderr=sp.STDOUT,start_new_session=True)
    record={'watchdog_pid':p.pid,'command':cmd,'output':str(OUT),'offline_only':True}
    json.dump(record,claim,indent=2);claim.flush();os.fsync(claim.fileno());print(json.dumps(record))
