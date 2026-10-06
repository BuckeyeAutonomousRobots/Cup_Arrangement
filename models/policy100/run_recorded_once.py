"""One recorded seed-8 attempt per requested model, no retries or expert actions."""
import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import subprocess as sp
import sys
import time
from zoneinfo import ZoneInfo

R=Path(__file__).resolve().parents[2];B=R/'ros_backend1.1'
sys.path.insert(0,str(R/'models/act_v1'))
from balanced_trial_once_20261004 import sha, runtime_hashes
sys.path.insert(0,str(R/'models/policy100'))
from train_compare import check_workloads

def window():
    assert dt.datetime.now(ZoneInfo('America/New_York'))<dt.datetime(2026,10,5,9,40,tzinfo=ZoneInfo('America/New_York')), 'Insufficient time before FEA handoff'

def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['ACT_100','Diffusion_100'],required=True);a=p.parse_args()
    kind=a.kind;name='policy-'+kind.lower()+'-20261005-01'
    d=R/'models/policy100/runs'/name;o=B/'runtime'/name
    run=R/'models/policy100/runs/ACT_Diffusion_100_20261004'/kind
    hashes={'ACT_100':'9b45a11b3c4209c26339021dd76c16efeede54c9d34eb6687a7d1cf0a605ee11',
            'Diffusion_100':'c34130789248e2288e6a78e20f1c2bd89ad27ad3ffadc6feb94a86c75d0842f4'}
    state={'pid':os.getpid(),'model':kind,'stage':'preflight','trial_launched':False}
    def save(**kw):
        state.update(kw,updated=dt.datetime.now().astimezone().isoformat());temp=d/'status.tmp'
        temp.write_text(json.dumps(state,indent=2));temp.replace(d/'status.json');print(json.dumps(state),flush=True)
    def logged(cmd,file,timeout=240):
        with (d/file).open('x') as log:return sp.run(cmd,cwd=B,stdout=log,stderr=sp.STDOUT,timeout=timeout,check=True)
    server=None
    with open('/tmp/baseline_collection_batch.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);window();check_workloads()
        assert not d.exists() and not o.exists() and not d.with_suffix('.claim').exists(),'Existing attempt; inspect only'
        assert (B/'profiles/active_scene_profile.txt').read_text().strip()=='profiles/scenes/baseline_v1/scene.yaml'
        with d.with_suffix('.claim').open('x') as claim:
            json.dump(state,claim);claim.flush();os.fsync(claim.fileno())
        d.mkdir();save()
        try:
            status=json.loads((run/'status.json').read_text())
            assert status['status']=='completed' and status['step']==5000 and status['checkpoint_reload_pass']
            assert sha(run/'last_policy/model.safetensors')==hashes[kind]==status['last_policy_sha256']
            runtime=runtime_hashes()
            expected={'rollout_policy_ros.py':'a118517bc800a8465bd6cf5fe190c9343878dd8feea92eb2db604992d56a323b',
                      'camera_pipeline.py':'14c07e038d4092879a70d6a84fe27b7ac53ecd1a169d8a56d2406dba4a2fcee9',
                      'observation_recovery.py':'4d9e58f44d6f2425e4ba2ada6ca6928b535a0169872d75668af46384b4e9a621',
                      'cup_task.py':'505447bd65d4ac04463564b3d6107a981785006eb453884272051b24fa9629fd'}
            assert runtime==expected,'Runtime differs from verified baseline'
            sock=B/'runtime'/(name+'.sock');assert not sock.exists()
            cmd=[str(R/'models/act_v1/.venv/bin/python'),'-u',str(R/'models/policy100/serve_comparison.py'),
                 '--kind',kind,'--run',str(run),'--socket',str(sock)]
            (d/'provenance.json').write_text(json.dumps({'run':str(run),'checkpoint_sha256':hashes[kind],
                'normalization_sha256':sha(run/'normalization.npz'),'runtime_hashes':runtime,
                'server_sha256':sha(R/'models/policy100/serve_comparison.py'),'runtime_mismatches':[]},indent=2))
            (d/'preflight_processes.txt').write_text(sp.check_output(['ps','-eo','pid,ppid,stat,args'],text=True))
            logged(cmd+['--self-test'],'self_test.log',180)
            check_workloads();window();save(stage='resetting')
            logged(['./scripts/backend11_lifecycle.sh','bringup_dual'],'bringup.log')
            logged(['./scripts/randomize_cups.sh','8'],'randomize.log',90)
            check_workloads();window();assert runtime_hashes()==runtime
            with (d/'server.log').open('x') as log:
                server=sp.Popen(cmd,cwd=R,stdout=log,stderr=sp.STDOUT)
                save(stage='server_starting',server_pid=server.pid)
                for _ in range(120):
                    if sock.exists():break
                    assert server.poll() is None,'Server exited';time.sleep(.5)
                assert sock.exists(),'Server startup timeout'
                save(stage='policy_running',trial_launched=True)
                with (d/'rollout.log').open('x') as log:
                    res=sp.run(['docker','exec','cup_arrangement_backend','bash','-lc',
                        'source /workspace/install/setup.bash; exec timeout --signal=TERM 530s python3 /workspace/scripts/rollout_policy_ros.py '
                        f'--socket /workspace/runtime/{name}.sock --output /workspace/runtime/{name} '
                        f'--policy-label {kind}-final-5000 --sim-seconds 90 --wall-seconds 500'],stdout=log,stderr=sp.STDOUT,timeout=570)
                save(stage='cleanup',rollout_returncode=res.returncode)
        except BaseException as exc:
            save(stage='blocked',error=repr(exc));raise
        finally:
            if server is not None:
                try:server.wait(timeout=10)
                except sp.TimeoutExpired:server.terminate();server.wait(timeout=10)
            check_workloads()
            processes=sp.check_output(['docker','exec','cup_arrangement_backend','ps','-eo','pid,stat,args'],text=True)
            (d/'cleanup_processes.txt').write_text(processes)
            states={int(r.split()[0]):r.split()[1] for r in processes.splitlines()[1:] if r.split()}
            if state['trial_launched']:
                events=[]
                for line in (d/'rollout.log').read_text().splitlines():
                    try:events.append(json.loads(line))
                    except ValueError:pass
                acquired=[e['paused_pids'] for e in events if e.get('ownership')=='acquired']
                released=[e['resumed_pids'] for e in events if e.get('ownership')=='released']
                assert acquired and acquired==released,'Ownership cleanup not confirmed'
                assert all(pid in states and not states[pid].startswith('T') for pid in acquired[0])
                assert (o/'summary.json').exists()
            save(cleanup_verified=True,stage='blocked' if state['stage']=='blocked' else 'awaiting_artifact_validation')

if __name__=='__main__':main()
