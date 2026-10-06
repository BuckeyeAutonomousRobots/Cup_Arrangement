#!/usr/bin/env python3
"""One local Unix-socket ACT inference session; no ROS or robot commands."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import socket
import time

import numpy as np
import torch
from lerobot.policies.act.modeling_act import ACTPolicy


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--socket', type=Path, required=True)
    p.add_argument('--self-test', action='store_true')
    a = p.parse_args()
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.25)
    policy = ACTPolicy.from_pretrained(a.run/'best_policy').to('cuda').eval()
    stats = dict(np.load(a.run/'normalization.npz'))
    mean = np.array([.485, .456, .406], np.float32)[:, None, None]
    std = np.array([.229, .224, .225], np.float32)[:, None, None]

    def infer(rgb, state):
        image = (rgb.transpose(2, 0, 1).astype(np.float32)/255.-mean)/std
        state = (np.asarray(state, np.float32)-stats['state_mean'])/stats['state_std']
        batch = {'observation.images.wrist': torch.from_numpy(image[None]).to('cuda'),
                 'observation.state': torch.from_numpy(state[None]).to('cuda')}
        with torch.inference_mode():
            action = policy.select_action(batch)[0].cpu().numpy()
        action = action*stats['action_std']+stats['action_mean']
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError('Invalid model output')
        return action.tolist()

    policy.reset()
    # Warm up and validate the actual checkpoint on a recorded observation.
    rgb = np.load(a.run/'cache/seed-8/images.npy', mmap_mode='r')[0].copy()
    state = np.load(a.run/'cache/seed-8/states.npy', mmap_mode='r')[0].copy()
    start = time.monotonic()
    result = infer(rgb, state)
    print(json.dumps({'self_test_action': result, 'warmup_s': time.monotonic()-start,
                     'checkpoint': str(a.run/'best_policy'),
                     'checkpoint_sha256': hashlib.sha256((a.run/'best_policy/model.safetensors').read_bytes()).hexdigest()}), flush=True)
    policy.reset()
    if a.self_test:
        return
    # Refuse to replace an existing socket: another rollout may own it.
    with socket.socket(socket.AF_UNIX) as server:
        server.bind(str(a.socket))
        a.socket.chmod(0o600)
        try:
            server.listen(1)
            server.settimeout(180)
            print('READY', flush=True)
            conn, _ = server.accept()
            with conn, conn.makefile('rwb') as f:
                conn.settimeout(600)
                for line in f:
                    row = json.loads(line)
                    if row.get('reset_policy', False):
                        policy.reset()
                    rgb = np.frombuffer(base64.b64decode(row['rgb']), np.uint8).reshape(240, 320, 3)
                    start = time.monotonic()
                    action = infer(rgb, row['state'])
                    f.write((json.dumps({'id': row['id'], 'action': action,
                                        'inference_wall_s': time.monotonic()-start})+'\n').encode())
                    f.flush()
        finally:
            a.socket.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
