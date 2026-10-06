"""CPU-only stock ACT ensemble replay on fixed expert observations, no ROS."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from lerobot.policies.act.modeling_act import ACTPolicy, ACTTemporalEnsembler

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def metrics(raw,target,ticks):
    applied=raw.copy();applied[:,:6]=np.clip(applied[:,:6],-.245,.245);applied[:,6]=np.clip(applied[:,6],-.01,0)
    rot=np.abs(target[:,5])>.02;hold=~rot
    steady=np.array([not np.any(rot[np.abs(ticks-t)<=10]) for t in ticks])
    err=np.abs(applied-target); close=np.flatnonzero(target[:,6]<-.004)[0];dt=np.r_[np.diff(ticks)/10.,0.];dt[dt>.100001]=0
    def avg(v,mask):return float(np.mean(v[mask])) if mask.any() else None
    return dict(samples=len(raw),rotation_samples=int(rot.sum()),holding_samples=int(hold.sum()),steady_samples=int(steady.sum()),
        arm_clamp_samples=int(np.any(np.abs(raw[:,:6])>.245,axis=1).sum()),wrist_clamp_samples=int((np.abs(raw[:,5])>.245).sum()),
        raw_wrist_peak_abs_rad_s=float(np.abs(raw[:,5]).max()),applied_mae_per_action=err.mean(0).tolist(),
        rotation_mae_rad_s=avg(err[:,5],rot),correct_rotation_fraction=avg((applied[:,5]*target[:,5]>0)&(np.abs(applied[:,5])>.02),rot),
        holding_false_rotation_fraction=avg(np.abs(applied[:,5])>.02,hold),steady_false_rotation_fraction=avg(np.abs(applied[:,5])>.02,steady),
        predicted_approach_command_deg=float(np.rad2deg(np.sum(applied[:close,5]*dt[:close]))),
        expert_approach_command_deg=float(np.rad2deg(np.sum(target[:close,5]*dt[:close]))),
        gripper_strong_close_recall=avg(applied[:,6]<-.004,target[:,6]<-.004))

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--comparison',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(exist_ok=False);torch.set_num_threads(2);torch.set_num_interop_threads(1)
    if hasattr(os,'sched_getaffinity'):os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[-2:]))
    manifest=json.loads((a.source/'dataset_manifest.json').read_text()); stats=dict(np.load(a.source/'normalization.npz'))
    mean=np.array([.485,.456,.406],np.float32)[None,:,None,None];std=np.array([.229,.224,.225],np.float32)[None,:,None,None]
    report=dict(status='running',device='cpu',dataset_sha256=sha(a.source/'dataset_manifest.json'),source_normalization_sha256=sha(a.source/'normalization.npz'),
        script_sha256=sha(Path(__file__)),checkpoint_selection='last_policy at step 5000, not server default best_policy',
        primary_gap_handling='Continuous per received observation; stock server does not infer resets from timestamp gaps',
        secondary_gap_handling='Reset at each missing cached 10Hz tick; sensitivity analysis, not reconstructed recovery events',
        limitations='Teacher-forced expert observations; no actual RPC scheduling, repeated observation selection, recovery triggers, pose guards or robot motion; CPU batched chunks validated against select_action prefix',arms={})
    start=time.monotonic()
    def save(): (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    for arm in ['uniform_control','rotation_balanced']:
        ckpt=a.comparison/arm/'last_policy';digest=sha(ckpt/'model.safetensors')
        other=dict(np.load(a.comparison/arm/'normalization.npz'));assert all(np.array_equal(stats[k],other[k]) for k in stats)
        policy=ACTPolicy.from_pretrained(ckpt).cpu().eval()
        assert policy.config.chunk_size==20 and policy.config.n_action_steps==1 and policy.config.temporal_ensemble_coeff==.01
        entries=[];report['arms'][arm]=dict(checkpoint_sha256=digest,normalization_sha256=sha(a.comparison/arm/'normalization.npz'),episodes=entries)
        for e in [e for e in manifest['episodes'] if e['split']=='validation']:
            c=Path(e['cache']);data={k:np.load(c/(k+'.npy'),mmap_mode='r') for k in ['images','states','actions','ticks']};ticks=np.asarray(data['ticks']);n=len(ticks)
            def batch(lo,hi):
                rgb=np.asarray(data['images'][lo:hi]).transpose(0,3,1,2).astype(np.float32)/255.
                q=(np.asarray(data['states'][lo:hi])-stats['state_mean'])/stats['state_std']
                return {'observation.images.wrist':torch.from_numpy((rgb-mean)/std),'observation.state':torch.from_numpy(q.copy())}
            chunks=[]
            with torch.inference_mode():
                for lo in range(0,n,8):
                    chunks.append(policy.predict_action_chunk(batch(lo,min(n,lo+8))).cpu())
                    time.sleep(.01)
                chunks=torch.cat(chunks)
                # Direct deployed API equivalence over more than one full chunk history.
                policy.reset(); reference=[]
                for i in range(min(25,n)):reference.append(policy.select_action(batch(i,i+1))[0].clone())
                ens=ACTTemporalEnsembler(.01,20);continuous=[];reset=[];sens=ACTTemporalEnsembler(.01,20)
                for i in range(n):
                    if i and ticks[i]!=ticks[i-1]+1:sens.reset()
                    continuous.append(ens.update(chunks[i:i+1])[0].clone());reset.append(sens.update(chunks[i:i+1])[0].clone())
                continuous=torch.stack(continuous);reset=torch.stack(reset);reference=torch.stack(reference)
                delta=float((continuous[:len(reference)]-reference).abs().max())
                torch.testing.assert_close(continuous[:len(reference)],reference,rtol=2e-4,atol=2e-5)
            raw=continuous.numpy()*stats['action_std']+stats['action_mean']; alt=reset.numpy()*stats['action_std']+stats['action_mean'];target=np.asarray(data['actions'])
            assert np.isfinite(raw).all()
            entry=dict(seed=e['seed'],missing_tick_boundaries=int((np.diff(ticks)!=1).sum()),prefix_normalized_max_difference=delta,
                       continuous=metrics(raw,target,ticks),reset_at_gaps=metrics(alt,target,ticks),
                       input_hashes={k:sha(c/(k+'.npy')) for k in ['states','actions','ticks']})
            entries.append(entry)
            np.savez(a.output/f'{arm}-seed-{e["seed"]}.npz',ticks=ticks,expert=target,continuous=raw,reset_at_gaps=alt)
            save();print(json.dumps(dict(arm=arm,seed=e['seed'],metrics=entry['continuous'])),flush=True)
        assert sha(ckpt/'model.safetensors')==digest
        report['arms'][arm]['checkpoint_unchanged']=True
        del policy;gc.collect()
    report.update(status='completed',wall_seconds=time.monotonic()-start);save()

if __name__=='__main__':main()
