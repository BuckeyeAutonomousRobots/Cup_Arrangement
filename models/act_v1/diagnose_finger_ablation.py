#!/usr/bin/env python3
"""Bounded offline finger-input intervention. No ROS or command interfaces."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from lerobot.policies.act.modeling_act import ACTPolicy


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def sustained(mask, ticks, n=3):
    for i in range(len(mask)-n+1):
        if mask[i:i+n].all() and np.all(np.diff(ticks[i:i+n]) == 1):
            return i
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    files = [a.run/'best_policy/model.safetensors', a.run/'normalization.npz']
    hashes = {str(f):sha(f) for f in files}
    manifest = json.loads((a.run/'dataset_manifest.json').read_text())
    stats = dict(np.load(files[1]))
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.cuda.set_per_process_memory_fraction(.20)
    policy = ACTPolicy.from_pretrained(a.run/'best_policy').cuda().eval()
    mean = np.array([.485,.456,.406],np.float32)[:,None,None]
    std = np.array([.229,.224,.225],np.float32)[:,None,None]
    report = dict(status='running', hashes=hashes, episodes=[],
        method='t0-4s to t0+2s; intervention from t0-2s; reset at start/gaps; 2s original warmup',
        onset='first sustained expert departure >0.01mm from pregrasp plateau; three contiguous samples',
        thresholds_mm=[.1,.25,.5], delay_sim_s=.5)
    started = time.monotonic()
    for seed in manifest['validation_seeds']:
        cache = a.run/'cache'/f'seed-{seed}'
        d = {k:np.load(cache/(k+'.npy'),mmap_mode='r') for k in ['images','states','actions','ticks']}
        ticks = np.asarray(d['ticks']); times = ticks/10.
        actions = np.asarray(d['actions']); states = np.asarray(d['states'])
        strong = np.flatnonzero(actions[:,6]<-.004)[0]
        plateau = (times>=times[strong]-3)&(times<=times[strong]-2)
        baseline = float(np.median(actions[plateau,6]))
        search = (times>=times[strong]-2)&(times<=times[strong])
        onset = sustained(search & (actions[:,6]<baseline-.00001), ticks)
        if onset is None: raise ValueError(f'No onset seed {seed}')
        t0 = times[onset]
        held = np.median(states[(times>=t0-.8)&(times<=t0-.3),6:],axis=0)
        ix = np.flatnonzero((times>=t0-4)&(times<=t0+2+1e-6))
        rel = times[ix]-t0
        measured = states[ix].copy()
        valid = rel>=-2-1e-6
        pre_motion = valid & (np.max(np.abs(measured[:,6:]-held),axis=1)<.00005) & (rel<=1)
        result = dict(seed=seed, expert_onset_elapsed_s=float(t0-times[0]),
                      pregrasp_target_m=baseline, held_fingers_m=held.tolist(),
                      pre_motion_samples=int(pre_motion.sum()), conditions={})
        arrays = dict(relative_sim_s=rel,ticks=ticks[ix],expert_actions=actions[ix],states=measured,
                      pre_motion_mask=pre_motion,analysis_mask=valid)
        for mode in ['sequential','reset_each']:
            for condition in ['original','held_open','delayed']:
                supplied = measured.copy()
                for j,i in enumerate(ix):
                    if not valid[j]: continue
                    if condition=='held_open': supplied[j,6:] = held
                    if condition=='delayed':
                        prev=max(0,int(np.searchsorted(ticks,ticks[i]-5,side='right')-1))
                        supplied[j,6:]=states[prev,6:]
                assert np.array_equal(supplied[:,:6],measured[:,:6])
                policy.reset(); out=[]
                with torch.inference_mode():
                    for j,i in enumerate(ix):
                        if mode=='reset_each' or (j and ticks[i]!=ticks[ix[j-1]]+1): policy.reset()
                        rgb=d['images'][i].transpose(2,0,1).astype(np.float32)/255.
                        state=(supplied[j]-stats['state_mean'])/stats['state_std']
                        batch={'observation.images.wrist':torch.from_numpy(((rgb-mean)/std)[None]).cuda(),
                               'observation.state':torch.from_numpy(state[None].copy()).cuda()}
                        pred=policy.select_action(batch)[0].cpu().numpy()*stats['action_std']+stats['action_mean']
                        if not np.isfinite(pred).all(): raise ValueError('Nonfinite prediction')
                        out.append(pred)
                out=np.asarray(out); key=mode+'_'+condition
                arrays[key]=out; arrays[key+'_supplied_fingers']=supplied[:,6:]
                close_onset={}; extra_onset={}
                for threshold in [.0001,.00025,.0005]:
                    k=str(round(threshold*1000,2))
                    idx=sustained(valid&(out[:,6]<baseline-threshold),ticks[ix])
                    close_onset[k]=None if idx is None else float(rel[idx])
                    idx=sustained(valid&(out[:,6]<supplied[:,6:].mean(1)-threshold),ticks[ix])
                    extra_onset[k]=None if idx is None else float(rel[idx])
                firstsec=(rel>=-1e-6)&(rel<1-1e-6)
                samples={str(t):float(out[np.argmin(np.abs(rel-t)),6]*1000) for t in [0,.5,1,2]}
                base=arrays[mode+'_original']
                result['conditions'][key]=dict(onset_relative_s=close_onset,
                    additional_closure_onset_relative_s=extra_onset, target_mm_at_seconds=samples,
                    first_second_mean_target_mm=float(out[firstsec,6].mean()*1000),
                    pre_motion_min_target_mm=float(out[pre_motion,6].min()*1000),
                    arm_change_mae_rad_s=np.mean(np.abs(out[valid,:6]-base[valid,:6]),axis=0).tolist())
        np.savez(a.output/f'seed-{seed}.npz',**arrays)
        report['episodes'].append(result)
        (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'seed':seed,'completed':len(report['episodes']),
                         'onsets_025mm':{k:v['onset_relative_s']['0.25'] for k,v in result['conditions'].items()}}),flush=True)
    assert all(sha(f)==hashes[str(f)] for f in files)
    report.update(status='completed',weights_and_normalization_unchanged=True,wall_seconds=time.monotonic()-started)
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__': main()
