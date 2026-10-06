#!/usr/bin/env python3
"""Small offline RGB+proprioception ridge behavior-cloning baseline. No ROS writes.

Commands are previous receipt-clock samples, not exact actuation timestamps.
This is a pipeline smoke baseline, not an autonomous manipulation claim.
"""
import argparse
import bisect
import json
import hashlib
import platform
from pathlib import Path
import cv2
import numpy as np

JOINTS = ['right_'+n+'_joint' for n in
          ('shoulder_pan','shoulder_lift','elbow','wrist_1','wrist_2','wrist_3')]
JOINTS += ['right_robotiq_hande_left_finger_joint','right_robotiq_hande_right_finger_joint']
ARM = '/right_joint_group_velocity_controller/commands'
GRIP = '/right_hande_position_controller/commands'

def load_episode(path):
    summary = json.loads((path/'summary.json').read_text())
    if summary['status'] != 'controller_pass':
        raise ValueError(f'Exclude failed/incomplete episode: {path}')
    events = [json.loads(s) for s in (path/'events.jsonl').read_text().splitlines()]
    start = next(e['receipt_sim_s'] for e in events if e['event']=='cheat_started')
    end = next(e['receipt_sim_s'] for e in events if e['event']=='cheat_finished')
    joints = {n: [] for n in JOINTS}
    commands = {ARM: [], GRIP: []}
    for line in (path/'robot_data.jsonl').open():
        row = json.loads(line); msg = row['message']; topic = row['topic']
        if topic == '/joint_states':
            stamp = msg['header']['stamp']; t = stamp['sec']+stamp['nanosec']*1e-9
            for name, pos in zip(msg['name'], msg['position']):
                if name in joints: joints[name].append((t,pos))
        elif topic in commands and row['receipt_sim_s'] is not None:
            commands[topic].append((row['receipt_sim_s'],msg['data']))
    series = {**joints,**commands}
    for v in series.values(): v.sort(key=lambda x:x[0])
    times = {k:[v[0] for v in values] for k,values in series.items()}
    X,Y,stamps = [],[],[]
    for line in (path/'frames.jsonl').open():
        frame=json.loads(line); t=frame['stamp_sim_s']
        if not start<=t<=end: continue
        prior={}
        for k,values in series.items():
            i=bisect.bisect_right(times[k],t)-1
            if i<0: break
            age=t-values[i][0]
            # Joint state and velocity stream must be fresh; gripper target is held.
            if k != GRIP and age>0.10: break
            prior[k]=values[i][1]
        if len(prior)!=len(series): continue
        if len(prior[ARM])!=6 or len(prior[GRIP])!=2:
            raise ValueError('Unexpected action dimensions')
        rgb=cv2.cvtColor(cv2.imread(str(path/frame['file'])),cv2.COLOR_BGR2RGB)
        pixels=cv2.resize(rgb,(16,12),interpolation=cv2.INTER_AREA).reshape(-1)/255.0
        X.append(np.r_[pixels,[prior[n] for n in JOINTS]])
        Y.append(np.r_[prior[ARM],np.mean(prior[GRIP])])
        stamps.append(t)
    if len(X)<20: raise ValueError(f'Too few aligned samples ({len(X)}): {path}')
    X,Y=np.array(X),np.array(Y)
    if not np.isfinite(X).all() or not np.isfinite(Y).all(): raise ValueError('Nonfinite data')
    return X,Y,dict(episode=str(path),samples=len(X),first_sim_s=stamps[0],last_sim_s=stamps[-1],
        summary_sha256=hashlib.sha256((path/'summary.json').read_bytes()).hexdigest(),
        frames_index_sha256=hashlib.sha256((path/'frames.jsonl').read_bytes()).hexdigest(),
        robot_data_sha256=hashlib.sha256((path/'robot_data.jsonl').read_bytes()).hexdigest())

def scores(y,pred):
    return dict(mae_per_action=np.abs(y-pred).mean(0).tolist(),
                rmse_per_action=np.sqrt(np.mean((y-pred)**2,axis=0)).tolist())

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train',nargs='+',type=Path,required=True)
    ap.add_argument('--validation',nargs='+',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--ridge',type=float,default=100.)
    a=ap.parse_args()
    if set(p.resolve() for p in a.train)&set(p.resolve() for p in a.validation):
        ap.error('Training and validation episodes must be disjoint')
    if a.ridge<=0: ap.error('ridge must be positive')
    train=[load_episode(p) for p in a.train]; val=[load_episode(p) for p in a.validation]
    x,y=np.concatenate([v[0] for v in train]),np.concatenate([v[1] for v in train])
    vx,vy=np.concatenate([v[0] for v in val]),np.concatenate([v[1] for v in val])
    xm,xs=x.mean(0),x.std(0); xs[xs<1e-6]=1
    ym,ys=y.mean(0),y.std(0); ys[ys<1e-6]=1
    z=(x-xm)/xs; target=(y-ym)/ys
    weights=np.linalg.solve(z.T@z+a.ridge*np.eye(z.shape[1]),z.T@target)
    pred=lambda f: (((f-xm)/xs)@weights)*ys+ym
    a.output.mkdir(parents=True,exist_ok=False)
    checkpoint=a.output/'model.npz'
    np.savez_compressed(checkpoint,weights=weights,input_mean=xm,input_scale=xs,
                        action_mean=ym,action_scale=ys)
    # Verify the saved file, not merely the in-memory arrays.
    with np.load(checkpoint) as saved:
        reloaded=((vx[:1]-saved['input_mean'])/saved['input_scale']@saved['weights'])*saved['action_scale']+saved['action_mean']
    np.testing.assert_allclose(reloaded,pred(vx[:1]),rtol=1e-10,atol=1e-10)
    report=dict(model='ridge_rgb_joint_behavior_cloning',ridge=a.ridge,
        environment=dict(python=platform.python_version(),numpy=np.__version__,opencv=cv2.__version__),
        trainer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        feature_order={'rgb':'12x16x3 RGB INTER_AREA, row-major, divided by 255','joints':JOINTS},
        actions=JOINTS[:6]+['common_finger_position_m'],
        action_units=['rad/s']*6+['m'],train_episodes=[v[2] for v in train],
        validation_episodes=[v[2] for v in val],train=scores(y,pred(x)),
        validation=scores(vy,pred(vx)),validation_mean_action=scores(vy,np.broadcast_to(ym,vy.shape)),
        checkpoint_reload_pass=True,closed_loop_evaluation=False,
        limitations=['Tiny dataset; offline imitation error is not task success',
                    'Unstamped command alignment uses previous receipt simulation clock',
                    'Gripper target is sample-and-held; only joints and arm commands gated at 100 ms',
                    'No image augmentation or hyperparameter selection; validation is exploratory'])
    (a.output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
