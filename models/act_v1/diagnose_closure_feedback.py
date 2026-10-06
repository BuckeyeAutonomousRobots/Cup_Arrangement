#!/usr/bin/env python3
"""CPU-only hypothetical jaw feedback, not a physics or robot rollout."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from lerobot.policies.act.modeling_act import ACTPolicy


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--ablation',type=Path,required=True)
    args=ap.parse_args()
    args.output.mkdir(exist_ok=False)
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    # Use one allowed CPU core; no CUDA allocation or simulation interfaces.
    if hasattr(os,'sched_getaffinity'):
        os.sched_setaffinity(0,{max(os.sched_getaffinity(0))})
    policy=ACTPolicy.from_pretrained(args.run/'best_policy').to('cpu').eval()
    stats=dict(np.load(args.run/'normalization.npz'))
    hashes={str(p):sha(p) for p in [args.run/'best_policy/model.safetensors',args.run/'normalization.npz']}
    mean=np.array([.485,.456,.406],np.float32)[:,None,None]
    std=np.array([.229,.224,.225],np.float32)[:,None,None]
    def infer(rgb,q):
        image=(rgb.transpose(2,0,1).astype(np.float32)/255.-mean)/std
        state=(q-stats['state_mean'])/stats['state_std']
        with torch.inference_mode():
            pred=policy.select_action({'observation.images.wrist':torch.from_numpy(image[None]),
                'observation.state':torch.from_numpy(state[None].copy())})[0].numpy()
        pred=pred*stats['action_std']+stats['action_mean']
        if not np.isfinite(pred).all():raise ValueError('Nonfinite action')
        return pred
    previous=json.loads((args.ablation/'summary.json').read_text())
    e=next(e for e in previous['episodes'] if e['seed']==8)
    cache=args.run/'cache/seed-8'
    data={k:np.load(cache/(k+'.npy'),mmap_mode='r') for k in ['images','states','actions','ticks']}
    ticks=np.asarray(data['ticks']); rel=(ticks-ticks[0])/10.
    onset=int(np.argmin(np.abs(rel-e['expert_onset_elapsed_s'])))
    t0=ticks[onset]
    warm=np.flatnonzero((ticks>=t0-20)&(ticks<t0))
    held=np.asarray(e['held_fingers_m'],np.float32)
    report=dict(status='running',seed=8,device='cpu',cpu_threads=1,hashes=hashes,
        time_step_s=.1,horizon_s=5,rate_limit_m_per_s=.0028,
        assumption='Hypothetical feedback, no cup contact, actual finger physics or rendered feedback',cases=[])
    start=time.monotonic()
    for visual in ['frozen_at_expert_close_onset','recorded_expert_sequence']:
        for feedback in ['held_open_control','ideal_tracking','rate_limited_tracking']:
            policy.reset()
            for j,i in enumerate(warm):
                if j and ticks[i]!=ticks[warm[j-1]]+1:policy.reset()
                infer(data['images'][i],data['states'][i].copy())
            qjaw=held.copy(); rows=[]
            for step in range(51):
                # Source images/arm joints never depend on predicted commands.
                idx=onset if visual.startswith('frozen') else max(0,int(np.searchsorted(ticks,t0+step,side='right')-1))
                q=data['states'][idx].copy(); q[6:]=qjaw
                action=infer(data['images'][idx],q)
                target=float(np.clip(action[6],-.01,0))
                rows.append(dict(t=step*.1,source_tick=int(ticks[idx]),input_fingers=qjaw.tolist(),
                                 raw_action=action.tolist(),applied_jaw_target=target))
                if feedback=='ideal_tracking':qjaw[:]=target
                elif feedback=='rate_limited_tracking':qjaw+=np.clip(target-qjaw,-.0028*.1,.0028*.1)
            target_times=[r['t'] for r in rows if r['applied_jaw_target']<-.004]
            state_times=[r['t'] for r in rows if max(r['input_fingers'])<-.004]
            item=dict(visual=visual,feedback=feedback,
                first_target_below_minus4mm_s=target_times[0] if target_times else None,
                first_both_fingers_below_minus4mm_s=state_times[0] if state_times else None,
                initial_target_mm=rows[0]['applied_jaw_target']*1000,
                final_target_mm=rows[-1]['applied_jaw_target']*1000,
                final_fingers_mm=(np.asarray(rows[-1]['input_fingers'])*1000).tolist())
            (args.output/(visual+'__'+feedback+'.json')).write_text(json.dumps(rows,indent=2)+'\n')
            report['cases'].append(item)
            (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(item),flush=True)
    assert all(sha(Path(p))==h for p,h in hashes.items())
    report.update(status='completed',wall_seconds=time.monotonic()-start,weights_and_normalization_unchanged=True)
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
