"""Exactly one authorized balanced ACT simulation evaluation; no retries."""
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess as sp
import time
from zoneinfo import ZoneInfo
import numpy as np

R = Path(__file__).resolve().parents[2]
B = R/'ros_backend1.1'
N = 'policy-balanced-final-20261004-01'
D = R/'models/act_v1/runs'/N
O = B/'runtime'/N
C = D.with_suffix('.claim')
P = R/'models/act_v1/runs/P3_ACT_50'
T = R/'models/act_v1/runs/ACT_rotation_comparison_20261004-02'
EXPECTED = '408d02ba6329124dd6754e9c30646c6df75de9e96471136fe293d7488d5e8f3d'
PY = str(R/'models/act_v1/.venv/bin/python')
state = {'pid': os.getpid(), 'stage': 'preflight', 'trial_launched': False}

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def save():
    state['updated'] = dt.datetime.now(ZoneInfo('America/New_York')).isoformat()
    tmp = D/'status.tmp'
    tmp.write_text(json.dumps(state, indent=2))
    os.replace(tmp, D/'status.json')
    print(json.dumps(state), flush=True)

def window():
    now = dt.datetime.now(ZoneInfo('America/New_York'))
    assert now.date() == dt.date(2026, 10, 4) and now.hour >= 18, 'Outside October 4 evening EDT gate'

def idle():
    forbidden = ['collect_resilient.py', 'record_collection_smoke.py', 'cup_task.py',
                 'rollout_policy_ros.py', 'serve_policy.py', 'train_rotation_comparison.py',
                 'run_jax', 'FEA_JAX', 'FEA_Project', 'abaqus', 'runSofa', 'fea-benchmark']
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        if int(p.parent.name) == os.getpid():
            continue
        try:
            args = p.read_bytes().decode(errors='replace').split('\0')
        except (OSError, ProcessLookupError):
            continue
        if any(token in arg for arg in args[:3] for token in forbidden):
            raise RuntimeError('Competing workload '+str(p)+': '+' '.join(args)[:300])
    gpu = sp.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'], text=True)
    assert not gpu.strip(), 'Unexpected GPU compute workload: '+gpu
    return sp.check_output(['ps', '-eo', 'pid,ppid,stat,args'], text=True)

def logged(cmd, filename, timeout=240):
    with (D/filename).open('x') as log:
        return sp.run(cmd, cwd=B, stdout=log, stderr=sp.STDOUT, timeout=timeout, check=True)

def runtime_hashes():
    names = ['rollout_policy_ros.py','camera_pipeline.py','observation_recovery.py','cup_task.py']
    result = {}
    for n in names:
        host = sha(B/'scripts'/n)
        inside = sp.check_output(['docker','exec','cup_arrangement_backend','sha256sum','/workspace/scripts/'+n], text=True).split()[0]
        assert host == inside, 'Host/container mismatch: '+n
        result[n] = host
    return result

def main():
    server = None
    with open('/tmp/baseline_collection_batch.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        window()
        assert not C.exists() and not D.exists() and not O.exists(), 'Existing claim/output; inspect only'
        batch = json.loads((B/'runtime/batch50-20261004-seeds50-99/status.json').read_text())
        assert batch['status'] == 'completed'
        assert not Path('/proc') .joinpath(str(batch['pid'])).exists(), 'Collector still exists'
        inventory = idle()
        assert (B/'profiles/active_scene_profile.txt').read_text().strip() == 'profiles/scenes/baseline_v1/scene.yaml'
        with C.open('x') as f:
            json.dump({'pid':os.getpid(), 'created':dt.datetime.now().isoformat(), 'trial':N}, f)
            f.flush(); os.fsync(f.fileno())
        D.mkdir()
        (D/'preflight_processes.txt').write_text(inventory)
        try:
            save()
            trained = json.loads((T/'status.json').read_text())
            assert trained['status']=='completed' and all(v['checkpoint_reload_pass'] for v in trained['arms'].values())
            replay = json.loads((R/'models/act_v1/runs/ACT_temporal_replay_20261004/summary.json').read_text())
            checkpoint = T/'rotation_balanced/last_policy'
            assert sha(checkpoint/'model.safetensors') == EXPECTED == replay['arms']['rotation_balanced']['checkpoint_sha256']
            assert sha(P/'normalization.npz') == replay['source_normalization_sha256']
            a,b = np.load(P/'normalization.npz'),np.load(T/'rotation_balanced/normalization.npz')
            assert set(a.files)==set(b.files) and all(np.array_equal(a[k],b[k]) for k in a.files)
            bundle = D/'bundle'; bundle.mkdir()
            (bundle/'best_policy').symlink_to(checkpoint, target_is_directory=True)
            (bundle/'normalization.npz').symlink_to(P/'normalization.npz')
            (bundle/'cache').symlink_to(P/'cache', target_is_directory=True)
            assert (bundle/'cache/seed-8/images.npy').is_file()
            sock = B/'runtime'/(N+'.sock')
            assert not sock.exists()
            hashes = runtime_hashes()
            provenance = {'checkpoint':str(checkpoint.resolve()),'checkpoint_sha256':EXPECTED,
                          'normalization':str((bundle/'normalization.npz').resolve()),
                          'normalization_sha256':sha(bundle/'normalization.npz'),
                          'normalization_numerically_identical':True,'cache':str((bundle/'cache').resolve()),
                          'runtime_hashes':hashes,'runtime_mismatches':[], 'batch':batch,
                          'server_sha256':sha(R/'models/act_v1/serve_policy.py')}
            (D/'provenance.json').write_text(json.dumps(provenance,indent=2))
            logged([PY,str(R/'models/act_v1/serve_policy.py'),'--run',str(bundle),'--socket',str(sock),'--self-test'],'self_test.log',120)
            idle(); window()
            state['stage']='resetting'; save()
            logged(['./scripts/backend11_lifecycle.sh','bringup_dual'],'bringup.log')
            logged(['./scripts/randomize_cups.sh','8'],'randomize.log',90)
            idle(); window()
            assert runtime_hashes()==hashes
            with (D/'server.log').open('x') as log:
                server=sp.Popen([PY,'-u',str(R/'models/act_v1/serve_policy.py'),'--run',str(bundle),'--socket',str(sock)],cwd=R,stdout=log,stderr=sp.STDOUT)
                state.update(stage='server_starting',server_pid=server.pid); save()
                for _ in range(120):
                    if sock.exists(): break
                    assert server.poll() is None,'Server exited'
                    time.sleep(.5)
                assert sock.exists(),'Server startup timeout'
                state.update(stage='policy_running',trial_launched=True); save()
                with (D/'rollout.log').open('x') as output:
                    result=sp.run(['docker','exec','cup_arrangement_backend','bash','-lc',
                        'source /workspace/install/setup.bash; exec timeout --signal=TERM 530s python3 /workspace/scripts/rollout_policy_ros.py '
                        f'--socket /workspace/runtime/{N}.sock --output /workspace/runtime/{N} '
                        '--policy-label ACT-rotation-balanced-final-5000 --sim-seconds 90 --wall-seconds 500'],stdout=output,stderr=sp.STDOUT,timeout=570)
                state.update(stage='cleanup',rollout_returncode=result.returncode); save()
        except BaseException as e:
            state.update(stage='blocked',error=repr(e)); save()
            raise
        finally:
            if server is not None:
                try: server.wait(timeout=10)
                except sp.TimeoutExpired:
                    server.terminate(); server.wait(timeout=10)
            # Release batch lock only after verified client exit and publisher restoration.
            idle()
            inventory = sp.check_output(['docker','exec','cup_arrangement_backend','ps','-eo','pid,stat,args'],text=True)
            (D/'cleanup_processes.txt').write_text(inventory)
            rows = {int(r.split()[0]):r.split()[1] for r in inventory.splitlines()[1:] if r.split()}
            if state['trial_launched']:
                events=[]
                for line in (D/'rollout.log').read_text().splitlines():
                    try: events.append(json.loads(line))
                    except ValueError: pass
                acquired=[e['paused_pids'] for e in events if e.get('ownership')=='acquired']
                released=[e['resumed_pids'] for e in events if e.get('ownership')=='released']
                assert acquired and acquired==released,'Ownership release missing'
                assert all(pid in rows and not rows[pid].startswith('T') for pid in acquired[0]), 'Owned publishers not restored'
                assert (O/'summary.json').exists(), 'Rollout summary missing'
            state['cleanup_verified']=True
            if state['stage']!='blocked': state['stage']='awaiting_artifact_validation'
            save()

if __name__=='__main__': main()
