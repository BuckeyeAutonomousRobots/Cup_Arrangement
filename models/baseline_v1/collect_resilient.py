#!/usr/bin/env python3
"""Bounded collection with explicit, preserved infrastructure retries only."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

BACKEND = (Path(__file__).resolve().parents[2] / 'ros_backend1.1')
CONTAINER = 'cup_arrangement_backend'
ACTIVE = r'scripts/(cup_task|record_collection_smoke|rollout_policy_ros)\.py'


def now():
    return datetime.datetime.now().astimezone().isoformat()


def retryable_startup(text):
    # Fail closed on unknown build/runtime errors even if old timeout text exists.
    if any(x in text for x in ('No space left', 'Permission denied', 'Traceback', 'SyntaxError', 'segmentation fault')):
        return False
    return any(x in text for x in ('Failed to activate controller', 'Switch controller timed out',
        'Controller manager not available', 'Could not contact service /',
        'Could not switch controller', 'waiting for service /left_controller_manager',
        'waiting for service /right_controller_manager'))


def retryable_recorder(summary, text):
    return (summary.get('status') == 'startup_failed' and
            summary.get('cheat_returncode') is None and
            'Camera/joints/calibration/static TF readiness failed' in text)


def idle():
    p = subprocess.run(['docker', 'exec', CONTAINER, 'pgrep', '-af', ACTIVE],
                       capture_output=True, text=True, timeout=15)
    if p.returncode != 1:
        raise RuntimeError('Robot not idle or Docker unavailable: '+p.stdout+p.stderr)
    if (BACKEND/'profiles/active_scene_profile.txt').read_text().strip() != 'profiles/scenes/baseline_v1/scene.yaml':
        raise RuntimeError('Active scene changed')
    if shutil.disk_usage(BACKEND).free < 50*1024**3:
        raise RuntimeError('Less than 50 GiB free; refusing further recording')


def command(args, log, timeout=None):
    with log.open('x') as stream:
        p = subprocess.Popen(args, cwd=BACKEND, stdout=stream, stderr=subprocess.STDOUT,
                             start_new_session=True)
        try:
            return p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                pass
            # Never reset after uncertain cleanup or a host-command timeout.
            raise RuntimeError('Host stage timeout; inspect cleanup: '+str(log))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', required=True)
    ap.add_argument('--start-seed', type=int, required=True)
    ap.add_argument('--count', type=int, required=True)
    a = ap.parse_args()
    if not 1 <= a.count <= 100 or a.start_seed < 0 or not a.batch.replace('-', '').isalnum():
        ap.error('Invalid batch/seed/count')
    with open('/tmp/baseline_collection_batch.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        root = BACKEND/'runtime'/a.batch
        root.mkdir(exist_ok=False)  # Never overwrite a previous batch or reset retry budgets.
        state = dict(batch=a.batch, pid=os.getpid(), status='running', started=now(),
                     seeds=list(range(a.start_seed, a.start_seed+a.count)), requested=a.count,
                     attempts=[], episodes=[], max_attempts_per_seed=3)
        def save():
            state['updated'] = now()
            temp = root/'status.tmp'; temp.write_text(json.dumps(state, indent=2)); temp.replace(root/'status.json')
        start = time.monotonic(); consecutive_failures = 0
        save()
        try:
            for index, seed in enumerate(state['seeds']):
                for attempt in range(1, 4):
                    if (root/'STOP_AFTER_CURRENT').exists():
                        state['status'] = 'stopped_by_request'; return 0
                    if time.monotonic()-start > 12*3600:
                        raise RuntimeError('12 hour batch budget exhausted')
                    idle()
                    name = f'episode-{index:03d}-seed-{seed}-attempt-{attempt}'
                    state.update(current_index=index+1, current_seed=seed, current_attempt=attempt, stage='reset')
                    entry = dict(seed=seed, attempt=attempt, directory=name, started=now(), status='resetting')
                    state['attempts'].append(entry); save()
                    bringup_log = root/(name+'-bringup.log')
                    rc = command(['./scripts/backend11_lifecycle.sh', 'bringup_dual'], bringup_log, 240)
                    retry = False
                    if rc:
                        retry = retryable_startup(bringup_log.read_text())
                        entry.update(status='bringup_failed', returncode=rc, retryable=retry)
                    else:
                        state['stage'] = 'randomize'; save()
                        if command(['./scripts/randomize_cups.sh', str(seed)], root/(name+'-randomize.log'), 40):
                            raise RuntimeError('Randomization failed; no automatic retry')
                        state['stage'] = 'recording'; entry['status'] = 'recording'; save()
                        dest = f'/workspace/runtime/{a.batch}/{name}'
                        record_log = root/(name+'-record.log')
                        # Recorder owns its 600 s watchdog and robot cleanup.
                        rc = command(['docker', 'exec', CONTAINER, 'bash', '-lc',
                            'source /workspace/install/setup.bash; '
                            f'python3 /workspace/scripts/record_collection_smoke.py --output {dest} --max-seconds 600'], record_log)
                        summary = json.loads((root/name/'summary.json').read_text())
                        entry.update(status=summary['status'], returncode=rc, frames=summary.get('frames'),
                                     wall_duration_s=summary.get('wall_duration_s'))
                        retry = retryable_recorder(summary, record_log.read_text())
                        entry['retryable'] = retry
                        if summary['status'] in ('controller_pass', 'controller_failed'):
                            idle()  # Verify cleanup before moving on to another seed.
                            state['episodes'].append(dict(entry))
                            state['successful'] = sum(x['status']=='controller_pass' for x in state['episodes'])
                            state['failed'] = len(state['episodes'])-state['successful']
                            consecutive_failures = consecutive_failures+1 if summary['status']=='controller_failed' else 0
                            entry['finished'] = now(); save()
                            if consecutive_failures >= 3:
                                raise RuntimeError('Three consecutive motion failures; stop for inspection')
                            break
                    entry['finished'] = now(); save()
                    if not retry:
                        raise RuntimeError('Unrecognized/unsafe failure; inspect '+name)
                    if attempt == 3:
                        raise RuntimeError('Infrastructure retry budget exhausted for seed '+str(seed))
                    idle()
                    state['stage'] = 'recovery_wait'; save()
                    time.sleep(10)
                    # Next attempt starts a fresh simulation, retaining every failed attempt.
            state['status'] = 'completed'
        except Exception as exc:
            state.update(status='needs_attention', error=str(exc))
        finally:
            state.update(finished=now(), stage='idle'); save()
            print(json.dumps(state, indent=2), flush=True)
        return 0 if state['status']=='completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
