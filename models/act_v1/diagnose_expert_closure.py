#!/usr/bin/env python3
"""Offline teacher-forced ACT replay. No ROS, sockets, or actuator interfaces."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from lerobot.policies.act.modeling_act import ACTPolicy


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[8, 18, 2])
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.cuda.set_per_process_memory_fraction(.20)
    policy = ACTPolicy.from_pretrained(a.run/'best_policy').cuda().eval()
    stats = dict(np.load(a.run/'normalization.npz'))
    manifest = json.loads((a.run/'dataset_manifest.json').read_text())
    mean = np.array([.485, .456, .406], np.float32)[:, None, None]
    std = np.array([.229, .224, .225], np.float32)[:, None, None]
    checkpoint = a.run/'best_policy/model.safetensors'
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    report = dict(checkpoint_sha256=digest, method='Teacher-forced expert observations; not closed-loop evaluation',
                  strong_close_threshold_m=-.004, episodes=[])
    start = time.monotonic()
    for seed in a.seeds:
        entry = next(e for e in manifest['episodes'] if e['seed'] == seed)
        cache = a.run/'cache'/f'seed-{seed}'
        data = {k: np.load(cache/(k+'.npy'), mmap_mode='r') for k in ['images', 'states', 'actions', 'ticks']}
        target = np.asarray(data['actions'])
        ticks = np.asarray(data['ticks'])
        elapsed = (ticks-ticks[0])/10.
        predictions = {}
        for mode in ['sequential_ensemble', 'reset_each_observation']:
            policy.reset()
            outputs = []
            with torch.inference_mode():
                for i in range(len(ticks)):
                    if mode == 'reset_each_observation' or (i and ticks[i] != ticks[i-1]+1):
                        policy.reset()
                    rgb = np.asarray(data['images'][i]).transpose(2, 0, 1).astype(np.float32)/255.
                    state = (np.asarray(data['states'][i])-stats['state_mean'])/stats['state_std']
                    batch = {'observation.images.wrist': torch.from_numpy(((rgb-mean)/std)[None]).cuda(),
                             'observation.state': torch.from_numpy(state[None].copy()).cuda()}
                    pred = policy.select_action(batch)[0].cpu().numpy()*stats['action_std']+stats['action_mean']
                    if not np.isfinite(pred).all():
                        raise ValueError('Nonfinite prediction')
                    outputs.append(pred)
            predictions[mode] = np.asarray(outputs)
        strong = target[:, 6] < -.004
        indices = np.flatnonzero(strong)
        first = int(indices[0]) if len(indices) else None
        groups = {'all': np.ones(len(ticks), bool), 'expert_strong_close': strong,
                  'before_first_close': np.arange(len(ticks)) < (first if first is not None else len(ticks))}
        if first is not None:
            groups['first_2s_after_close'] = (elapsed >= elapsed[first]) & (elapsed < elapsed[first]+2)
        item = dict(seed=seed, split=entry['split'], samples=len(ticks),
                    expert_first_strong_close_s=float(elapsed[first]) if first is not None else None,
                    modes={})
        for mode, pred in predictions.items():
            crossed = np.flatnonzero(pred[:, 6] < -.004)
            metrics = dict(first_strong_close_s=float(elapsed[crossed[0]]) if len(crossed) else None,
                           gripper_min_m=float(pred[:, 6].min()), groups={})
            for name, mask in groups.items():
                if not mask.any():
                    continue
                metrics['groups'][name] = dict(n=int(mask.sum()),
                    action_mae=np.mean(np.abs(pred[mask]-target[mask]), axis=0).tolist(),
                    predicted_gripper_mean_m=float(pred[mask, 6].mean()),
                    expert_gripper_mean_m=float(target[mask, 6].mean()),
                    strong_close_fraction=float(np.mean(pred[mask, 6] < -.004)))
            item['modes'][mode] = metrics
        np.savez(a.output/f'seed-{seed}.npz', elapsed_sim_s=elapsed, ticks=ticks,
                 expert_actions=target, states=data['states'], **predictions)
        report['episodes'].append(item)
        (a.output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(item), flush=True)
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != digest:
        raise RuntimeError('Checkpoint changed during diagnostic')
    report.update(status='completed', wall_seconds=time.monotonic()-start,
                  checkpoint_unchanged=True)
    (a.output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
