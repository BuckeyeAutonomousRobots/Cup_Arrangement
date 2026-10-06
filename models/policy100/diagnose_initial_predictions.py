"""Offline teacher-forced start comparison; no simulator commands."""
import fcntl,json,time
from pathlib import Path
import numpy as np
import torch
import cv2
from train_compare import ROOT,check_workloads,sha,atomic_json
from serve_comparison import Adapter

def main():
    with open('/tmp/baseline_collection_batch.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);check_workloads()
        torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.4)
        root=ROOT/'models/policy100/runs';source=root/'ACT_Diffusion_100_20261004'
        out=root/'initial_prediction_audit_20261005.json'
        assert not out.exists()
        images=np.load(source/'cache/seed-8/images.npy',mmap_mode='r')
        states=np.load(source/'cache/seed-8/states.npy');actions=np.load(source/'cache/seed-8/actions.npy');ticks=np.load(source/'cache/seed-8/ticks.npy')
        ep=ROOT/'ros_backend1.1/runtime/policy-act_100-20261005-01'
        first=json.loads((ep/'policy_actions.jsonl').open().readline())
        frames=[json.loads(x) for x in (ep/'frames.jsonl').read_text().splitlines()]
        frame=frames[first['image_index']]
        live=cv2.cvtColor(cv2.resize(cv2.imread(str(ep/frame['file'])),(320,240),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
        n=int(np.searchsorted(ticks,ticks[0]+50))
        results={'expert_first5_sim_seconds_samples':n,'ticks':ticks[:n].tolist(),
                 'initial_state_max_difference':float(np.max(np.abs(states[0]-first['state']))),
                 'initial_rgb_mean_absolute_difference_255':float(np.abs(images[0].astype(float)-live).mean()),
                 'expert_actions':actions[:n].tolist(),'models':{},'interpretation':'Teacher-forced predictions, not model-driven trajectories.'}
        for name,run in [('ACT_5k',source/'ACT_100'),('ACT_10k',root/'ACT_100_10k_20261005/ACT_100')]:
            check_workloads();adapter=Adapter(run,'ACT_100');pred=[]
            for i in range(n):pred.append(adapter.infer(images[i].copy(),states[i]))
            adapter.reset();initial_live=adapter.infer(live,first['state'])
            results['models'][name]={'sha256':sha(run/'last_policy/model.safetensors'),
                'expert_observation_predictions':pred,'live_initial_prediction':initial_live}
            del adapter;torch.cuda.empty_cache()
        atomic_json(out,results)
        print(json.dumps({k:v for k,v in results.items() if k not in ('models','expert_actions','ticks')}))
        for name,m in results['models'].items():
            pred=np.array(m['expert_observation_predictions']);mask=np.abs(actions[:n,5])>.02
            print(json.dumps({'model':name,'initial_live_prediction':m['live_initial_prediction'],
                'initial_expert_prediction':pred[0].tolist(),'rotation_samples':int(mask.sum()),
                'expert_rotation_mean':float(actions[:n,5][mask].mean()),'predicted_rotation_mean':float(pred[mask,5].mean())}))

if __name__=='__main__':main()
