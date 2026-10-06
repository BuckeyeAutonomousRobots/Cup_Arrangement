"""Offline-only position-target pilot. No ROS, GPU, or controller changes.

Targets: measured arm q(t+0.1s), commanded common jaw at t. Future states
are labels only. Invalid next ticks are masked, never bridged or interpolated.
This prepares evidence, not a deployment-ready ACT dataset/controller.
"""
import argparse
import bisect
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ARM = '/right_joint_group_velocity_controller/commands'

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def targets(q, a, ticks):
    valid = np.r_[np.diff(ticks) == 1, False]
    y = np.full((len(q), 7), np.nan, dtype=np.float32)
    idx = np.flatnonzero(valid)
    y[idx, :6] = q[idx+1, :6]
    y[idx, 6] = a[idx, 6]
    return y, valid

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--deadline', type=float, required=True)
    args = ap.parse_args()
    lock = open('/tmp/baseline_collection_batch.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Existing workload checker, invoked before this CPU-only scan.
    from train_compare import check_workloads
    check_workloads()
    q = np.arange(32).reshape(4, 8); a = np.zeros((4, 7))
    y, valid = targets(q, a, np.array([0, 1, 3, 4]))
    assert valid.tolist() == [True, False, True, False]
    assert np.array_equal(y[0, :6], q[1, :6]) and np.isnan(y[1]).all()
    src = ROOT/'models/policy100/runs/ACT_Diffusion_100_20261004/dataset_manifest.json'
    out = ROOT/'models/policy100/runs/position_target_pilot_20261005'
    out.mkdir(exist_ok=False)
    report = {'status':'running', 'source_manifest_sha256':sha(src),
              'contract':'input q/image at or before t; target measured arm q at next contiguous 10Hz tick; jaw COMMAND at t',
              'unit_test':'passed gap and terminal masking', 'episodes':[], 'onset_audit':[],
              'deployment_ready':False, 'simulation_run':False}
    def save(): (out/'summary.json').write_text(json.dumps(report, indent=2))
    save()
    try:
        for e in json.loads(src.read_text())['episodes']:
            if time.time() >= args.deadline: raise TimeoutError('Resource stop deadline')
            p = Path(e['cache']); d = {}
            for k in ('states', 'actions', 'ticks'):
                assert sha(p/(k+'.npy')) == e['cache_sha256'][k], (e['seed'], k)
                d[k] = np.load(p/(k+'.npy'), mmap_mode='r')
            y, valid = targets(d['states'], d['actions'], d['ticks'])
            assert np.isfinite(y[valid]).all()
            speeds = np.abs(y[valid, :6]-d['states'][valid, :6])*10
            dest = out/('seed-'+str(e['seed'])+'.npz')
            np.savez_compressed(dest, targets=y, valid=valid, ticks=d['ticks'])
            report['episodes'].append({'seed':e['seed'], 'split':e['split'],
                'source_cache':str(p), 'targets_sha256':sha(dest), 'valid':int(valid.sum()),
                'invalid':int((~valid).sum()), 'max_measured_speed_rad_s':speeds.max(axis=0).tolist(),
                'intervals_above_existing_0_245_limit':int((speeds>.245).any(axis=1).sum())})
            # Fit-only seeds; do not tune labels against held-out data.
            if e['seed'] not in (0,3,4,5,6,7,10,12): continue
            ep = Path(e['episode']); frames = [json.loads(l) for l in (ep/'frames.jsonl').open()]
            ft = [f['stamp_sim_s'] for f in frames]
            rot = np.flatnonzero(np.abs(d['actions'][:,5])>.02)
            if not len(rot): continue
            t = float(d['ticks'][rot[0]])/10
            cmds = []
            for line in (ep/'robot_data.jsonl').open():
                r = json.loads(line)
                if r['topic'] == ARM and r['receipt_sim_s'] is not None:
                    s = r['receipt_sim_s']; v = r['message']['data'][5]
                    if t-.3 <= s <= t and abs(v)>.02: cmds.append((s,v))
            onset = min(x[0] for x in cmds) if cmds else None
            i = bisect.bisect_right(ft, t)-1
            j = bisect.bisect_right(ft, (onset if onset is not None else t)-.001)-1
            item = {'seed':e['seed'], 'first_rotation_grid_s':t,
                'first_recent_command_receipt_sim_s':onset, 'grid_image_s':ft[i],
                'image_minus_command_s':None if onset is None else ft[i]-onset,
                'pre_command_image_s':ft[j] if j>=0 else None}
            if j>=0:
                name='seed-'+str(e['seed'])+'-pre-onset.png'
                shutil.copyfile(ep/frames[j]['file'], out/name); item['pre_image']=name
            report['onset_audit'].append(item)
        report['status']='completed'
        report['note']='Measured future states are not desired expert setpoints. Controller feasibility, tracking and live replay remain unvalidated. Timestamp audit uses receipt-clock command estimates, not actuation times.'
    except BaseException as exc:
        report['status']='failed'; report['error']=repr(exc); raise
    finally: save()
    print(json.dumps({k:v for k,v in report.items() if k!='episodes'}, indent=2))

if __name__ == '__main__': main()
