"""One bounded simulation-only expert position replay; never trains a policy."""
import datetime as dt
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess as sp
import sys
import time
from zoneinfo import ZoneInfo
from train_compare import check_workloads

R=Path(__file__).resolve().parents[2]; B=R/'ros_backend1.1'
NAME='expert-position-replay-20261005-01'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    global NAME
    parser=argparse.ArgumentParser();parser.add_argument('--raw-reference',action='store_true')
    parser.add_argument('--resume-import-failure',action='store_true');args=parser.parse_args()
    if args.raw_reference: NAME='expert-position-replay-20261005-02'
    if args.resume_import_failure:
        assert args.raw_reference
        NAME='expert-position-replay-20261005-04'
    deadline=dt.datetime(2026,10,5,9,40,tzinfo=ZoneInfo('America/New_York')).timestamp()
    d=R/'models/policy100/runs'/NAME; output=B/'runtime'/NAME
    server=None; launched=False
    with open('/tmp/baseline_collection_batch.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);check_workloads()
        assert time.time()+(450 if args.resume_import_failure else 700)<deadline,'Insufficient reset/trial/cleanup window'
        if args.resume_import_failure:
            prior=R/'models/policy100/runs/expert-position-replay-20261005-02'
            prior_log=(prior/'rollout.log').read_text()
            assert "ModuleNotFoundError: No module named 'rclpy'" in prior_log and 'policy_started' not in prior_log
            assert not (B/'runtime/expert-position-replay-20261005-02').exists()
            third=R/'models/policy100/runs/expert-position-replay-20261005-03'
            assert 'FileNotFoundError' in (third/'rollout.log').read_text()
            assert not (B/'runtime/expert-position-replay-20261005-03/policy_actions.jsonl').exists()
        assert not d.exists() and not output.exists()
        with d.with_suffix('.claim').open('x') as f:json.dump({'pid':os.getpid(),'mode':'expert_position_replay'},f)
        d.mkdir();state={'status':'preflight','learned_policy':False}
        def save(**kw):state.update(kw);(d/'status.json').write_text(json.dumps(state,indent=2))
        def run(cmd,name,timeout):
            with (d/name).open('x') as log:sp.run(cmd,cwd=B,stdout=log,stderr=sp.STDOUT,timeout=timeout,check=True)
        try:
            source=B/'scripts/rollout_policy_ros.py'
            assert sha(source)=='a118517bc800a8465bd6cf5fe190c9343878dd8feea92eb2db604992d56a323b'
            text=source.read_text();old="'reset_policy': reset_policy}, latest)"
            assert text.count(old)==1
            # Only add a timestamp to the existing IPC; all runtime guards unchanged.
            scoped=B/'runtime'/(NAME+'-runtime.py')
            assert not scoped.exists()
            text=text.replace(old,"'reset_policy': reset_policy, 'observation_sim_s': obs_sim}, latest)")
            for module in ('camera_pipeline.py','observation_recovery.py'):
                text=text.replace(f"Path(__file__).with_name('{module}')",f"(ROOT/'scripts/{module}')")
            text=text.replace("'inputs': 'wrist RGB + 8 joints only'", "'inputs': 'EXPERT recorded seed8 joint trajectory plus live joint feedback; NOT a learned policy'")
            scoped.write_text(text)
            sock=B/'runtime'/(NAME+'.sock')
            cmd=[str(R/'models/act_v1/.venv/bin/python'),str(R/'models/policy100/position_replay_server.py'),
                 '--cache',str(R/('models/policy100/runs/position_target_pilot_20261005/raw_seed8_reference' if args.raw_reference else 'models/policy100/runs/ACT_Diffusion_100_20261004/cache/seed-8')),
                 '--socket',str(sock),'--log',str(d/'tracking.jsonl')]
            run(cmd+['--self-test'],'self_test.log',30)
            run(['docker','exec','cup_arrangement_backend','bash','-lc',
                 'source /workspace/install/setup.bash; export PYTHONPATH=/workspace/scripts:$PYTHONPATH; python3 -c "import rclpy, cup_task, camera_pipeline, observation_recovery"'],
                 'import_test.log',20)
            save(status='resetting',runtime_source_sha256=sha(source),scoped_runtime_sha256=sha(scoped))
            check_workloads()
            assert (B/'profiles/active_scene_profile.txt').read_text().strip()=='profiles/scenes/baseline_v1/scene.yaml'
            if not args.resume_import_failure:
                run(['./scripts/backend11_lifecycle.sh','bringup_dual'],'bringup.log',180)
                run(['./scripts/randomize_cups.sh','8'],'randomize.log',60)
            else:save(reset_reused_from='expert-position-replay-20261005-02; import failed before control')
            check_workloads();assert time.time()+430<deadline
            with (d/'server.log').open('x') as log:
                server=sp.Popen(cmd,stdout=log,stderr=sp.STDOUT)
                for _ in range(60):
                    if sock.exists():break
                    assert server.poll() is None;time.sleep(.2)
                assert sock.exists()
                launched=True;save(status='running',server_pid=server.pid)
                run(['docker','exec','cup_arrangement_backend','bash','-lc',
                    'source /workspace/install/setup.bash; export PYTHONPATH=/workspace/scripts:$PYTHONPATH; '
                    f'exec timeout --signal=TERM 380s python3 /workspace/runtime/{scoped.name} '
                    f'--socket /workspace/runtime/{NAME}.sock --output /workspace/runtime/{NAME} '
                    '--policy-label EXPERT-POSITION-REPLAY-NOT-LEARNED --sim-seconds 60 --wall-seconds 350'],
                    'rollout.log',410)
                save(status='ended')
        except BaseException as e:save(status='blocked',error=repr(e));raise
        finally:
            if server is not None:
                try:server.wait(timeout=10)
                except sp.TimeoutExpired:server.terminate();server.wait(timeout=10)
            check_workloads()
            proc=sp.check_output(['docker','exec','cup_arrangement_backend','ps','-eo','pid,stat,args'],text=True)
            (d/'cleanup_processes.txt').write_text(proc)
            if launched:
                events=[]
                for line in (d/'rollout.log').read_text().splitlines():
                    try:events.append(json.loads(line))
                    except ValueError:pass
                got=[x['paused_pids'] for x in events if x.get('ownership')=='acquired']
                released=[x['resumed_pids'] for x in events if x.get('ownership')=='released']
                assert got and got==released,'Unverified command ownership cleanup'
                states={int(x.split()[0]):x.split()[1] for x in proc.splitlines()[1:] if x.split()}
                assert all(pid in states and not states[pid].startswith('T') for pid in got[0])
                assert not any(NAME+'-runtime.py' in x for x in proc.splitlines()),'Replay controller still active'
            save(cleanup_verified=True)
if __name__=='__main__':main()
