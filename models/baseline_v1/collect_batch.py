#!/usr/bin/env python3
"""Bounded host-side batch: fresh baseline scene per seed, no hidden retries."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

BACKEND=(Path(__file__).resolve().parents[2] / 'ros_backend1.1')
CONTAINER='cup_arrangement_backend'
def now():return datetime.datetime.now().astimezone().isoformat()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--batch',required=True)
    ap.add_argument('--count',type=int,default=10)
    ap.add_argument('--start-seed',type=int,default=0)
    a=ap.parse_args()
    if not 1<=a.count<=100 or not a.batch.replace('-','').isalnum():
        ap.error('Invalid batch name/count')
    root=BACKEND/'runtime'/a.batch
    root.mkdir(parents=True,exist_ok=True)
    lock=open('/tmp/baseline_collection_batch.lock','w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(batch=a.batch,pid=os.getpid(),started=now(),status='running',
        requested=a.count,seeds=list(range(a.start_seed,a.start_seed+a.count)),episodes=[])
    def save():
        state['updated']=now()
        tmp=root/'status.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(root/'status.json')
    def command(args,log,timeout=None):
        with log.open('w') as out:
            p=subprocess.Popen(args,cwd=BACKEND,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
            try:return p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=15)
                except subprocess.TimeoutExpired:pass
                raise RuntimeError('Stage timed out; batch stopped for inspection: '+str(log))
    save()
    try:
        for index,seed in enumerate(state['seeds']):
            if (root/'STOP_AFTER_CURRENT').exists():
                state['status']='stopped_by_request';break
            state.update(current_index=index+1,current_seed=seed,stage='reset')
            save()
            # Refuse to reset while any other collection/controller is active.
            active=subprocess.run(['docker','exec',CONTAINER,'pgrep','-af',
                r'scripts/(cup_task|record_collection_smoke|rollout_policy_ros)\.py'],capture_output=True,text=True)
            if active.returncode!=1:
                raise RuntimeError('Backend not idle/available: '+active.stdout+active.stderr)
            scene=(BACKEND/'profiles/active_scene_profile.txt').read_text().strip()
            if scene!='profiles/scenes/baseline_v1/scene.yaml':
                raise RuntimeError('Active scene changed; refusing batch reset')
            episode=f'episode-{index:03d}-seed-{seed}'
            if command(['./scripts/backend11_lifecycle.sh','bringup_dual'],root/f'{episode}-bringup.log',180):
                raise RuntimeError('Bringup failed')
            state['stage']='randomize';save()
            if command(['./scripts/randomize_cups.sh',str(seed)],root/f'{episode}-randomize.log',40):
                raise RuntimeError('Randomization failed')
            state['stage']='recording';save();started=time.monotonic()
            dest=f'/workspace/runtime/{a.batch}/{episode}'
            # Recorder internally bounds motion to 600 s and handles SIGTERM
            # cleanup. Avoid killing docker exec from the host mid-controller.
            rc=command(['docker','exec',CONTAINER,'bash','-lc',
                'source /workspace/install/setup.bash; '
                f'python3 /workspace/scripts/record_collection_smoke.py --output {dest} --max-seconds 600'],
                root/f'{episode}-record.log')
            summary_path=root/episode/'summary.json'
            if not summary_path.exists():raise RuntimeError('Recorder summary missing; inspect cleanup')
            summary=json.loads(summary_path.read_text())
            state['episodes'].append(dict(seed=seed,directory=episode,
                status=summary['status'],returncode=rc,frames=summary.get('frames'),
                wall_duration_s=round(time.monotonic()-started,2)))
            state['successful']=sum(e['status']=='controller_pass' for e in state['episodes'])
            state['failed']=len(state['episodes'])-state['successful'];save()
            # A known task failure is retained, then the next iteration verifies
            # idle ownership and performs a fresh reset. No failed data is training-ready.
            if summary['status'] not in ('controller_pass','controller_failed'):
                raise RuntimeError('Recorder interrupted/timed out; stopping for inspection')
        else:state['status']='completed'
    except Exception as error:
        state['status']='needs_attention';state['error']=str(error)
    finally:
        state['finished']=now();state['stage']='idle';save()
        print(json.dumps(state,indent=2),flush=True)
    return 0 if state['status']=='completed' else 1

if __name__=='__main__':raise SystemExit(main())
