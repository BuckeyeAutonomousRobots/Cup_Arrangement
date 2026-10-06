"""Audit fixed full validation first-action sequences; not closed-loop physics."""
import argparse
import json
from pathlib import Path
import numpy as np
from train_act import sha, atomic_json

p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--comparison',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(exist_ok=False)
m=json.loads((a.source/'dataset_manifest.json').read_text())
arms=['uniform_control','rotation_balanced']
pred={k:np.load(a.comparison/k/'validation_first_actions.npy') for k in arms}
report=dict(method='All recorded validation samples, saved first actions at step 5000. No temporal ensembling or model-driven state evolution.',
    dataset_sha256=sha(a.source/'dataset_manifest.json'),normalization_sha256=sha(a.source/'normalization.npz'),
    checkpoints={k:sha(a.comparison/k/'last_policy/model.safetensors') for k in arms},episodes=[])
offset=0
for e in [e for e in m['episodes'] if e['split']=='validation']:
    root=Path(e['cache']);target=np.load(root/'actions.npy');q=np.load(root/'states.npy');ticks=np.load(root/'ticks.npy');n=len(ticks)
    rot=np.abs(target[:,5])>.02;hold=~rot
    # Exclude a one-second neighborhood of any rotating label for steady holds.
    steady=np.array([not np.any(rot[np.abs(ticks-t)<=10]) for t in ticks])
    dt=np.r_[np.diff(ticks)/10.,0.];dt[dt>.100001]=0
    close=np.flatnonzero(target[:,6]<-.004)[0];approach=np.arange(n)<close
    item=dict(seed=e['seed'],samples=n,rotation_samples=int(rot.sum()),steady_hold_samples=int(steady.sum()),arms={})
    arrays=dict(expert=target,states=q,ticks=ticks,rotation=rot,steady_hold=steady)
    for arm in arms:
        raw=pred[arm][offset:offset+n];assert raw.shape==target.shape
        applied=raw.copy();applied[:,:6]=np.clip(applied[:,:6],-.245,.245);applied[:,6]=np.clip(applied[:,6],-.01,0)
        err=np.abs(raw-target);r=raw[:,5]
        # Integrated command discrepancy along EXPERT observations, never actual angle error.
        cum=np.cumsum((applied[:,5]-target[:,5])*dt)
        item['arms'][arm]=dict(mae_per_action=err.mean(0).tolist(),
            approach_mae_per_action=err[approach].mean(0).tolist(),
            rotation_wrong_direction_fraction=float(np.mean(r[rot]*target[rot,5]<0)),
            rotation_correct_nontrivial_fraction=float(np.mean((r[rot]*target[rot,5]>0)&(np.abs(r[rot])>.02))),
            rotation_mae_rad_s=float(err[rot,5].mean()),
            holding_false_rotation_fraction=float(np.mean(np.abs(r[hold])>.02)),
            steady_hold_false_rotation_fraction=float(np.mean(np.abs(r[steady])>.02)) if steady.any() else None,
            steady_hold_peak_abs_rad_s=float(np.abs(r[steady]).max()) if steady.any() else None,
            wrist_peak_abs_rad_s=float(np.abs(r).max()),
            arm_clamp_samples=int(np.any(np.abs(raw[:,:6])>.245,axis=1).sum()),
            expert_approach_wrist_command_integral_deg=float(np.rad2deg(np.sum(target[approach,5]*dt[approach]))),
            predicted_approach_wrist_command_integral_deg=float(np.rad2deg(np.sum(applied[approach,5]*dt[approach]))),
            peak_cumulative_wrist_command_discrepancy_deg=float(np.rad2deg(np.abs(cum).max())),
            gripper_strong_close_recall=float(np.mean(raw[target[:,6]<-.004,6]<-.004)))
        arrays[arm]=raw
    np.savez(a.output/f'seed-{e["seed"]}.npz',**arrays)
    report['episodes'].append(item);offset+=n
assert all(len(v)==offset for v in pred.values())
atomic_json(a.output/'summary.json',report)
for e in report['episodes']:
    print(json.dumps({'seed':e['seed'],**{k:{metric:v for metric,v in e['arms'][k].items() if metric in ['rotation_mae_rad_s','steady_hold_false_rotation_fraction','wrist_peak_abs_rad_s','arm_clamp_samples','predicted_approach_wrist_command_integral_deg','expert_approach_wrist_command_integral_deg','gripper_strong_close_recall']} for k in arms}}))
