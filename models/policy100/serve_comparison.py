"""Model-specific ACT / diffusion Unix socket adapter; no ROS/control interfaces."""
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
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

class Adapter:
    def __init__(self,run,kind):
        self.kind=kind
        cls=ACTPolicy if kind=='ACT_100' else DiffusionPolicy
        self.policy=cls.from_pretrained(run/'last_policy').cuda().eval()
        self.stats=dict(np.load(run/'normalization.npz'))
        self.mean=np.array([.485,.456,.406],np.float32)[:,None,None]
        self.std=np.array([.229,.224,.225],np.float32)[:,None,None]
        if kind=='ACT_100':
            assert self.policy.config.chunk_size==20 and self.policy.config.temporal_ensemble_coeff==.01
        else:
            assert self.policy.config.n_obs_steps==2 and self.policy.config.n_action_steps==8
            assert self.policy.config.num_inference_steps==100 and self.policy.config.noise_scheduler_type=='DDPM'
    def reset(self): self.policy.reset()
    def infer(self,rgb,state):
        image=(rgb.transpose(2,0,1).astype(np.float32)/255-self.mean)/self.std
        s=self.stats
        center,scale=('mean','std') if self.kind=='ACT_100' else ('center','scale')
        state=(np.asarray(state,np.float32)-s['state_'+center])/s['state_'+scale]
        batch={'observation.images.wrist':torch.from_numpy(image[None]).cuda(),
               'observation.state':torch.from_numpy(state[None]).cuda()}
        with torch.inference_mode(): action=self.policy.select_action(batch)[0].cpu().numpy()
        action=action*s['action_'+scale]+s['action_'+center]
        assert action.shape==(7,) and np.isfinite(action).all()
        return action.tolist()

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--kind',choices=['ACT_100','Diffusion_100'],required=True)
    p.add_argument('--socket',type=Path,required=True);p.add_argument('--self-test',action='store_true');a=p.parse_args()
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.4);torch.manual_seed(42)
    adapter=Adapter(a.run,a.kind)
    rgb=np.load(a.run/'cache/seed-8/images.npy',mmap_mode='r')
    states=np.load(a.run/'cache/seed-8/states.npy',mmap_mode='r')
    adapter.infer(rgb[0].copy(),states[0].copy());adapter.reset()
    if a.self_test:
        times=[];outputs=[]
        torch.manual_seed(42)
        for i in range(24):
            start=time.monotonic();outputs.append(adapter.infer(rgb[i].copy(),states[i].copy()));times.append(time.monotonic()-start)
        # Reset clears ACT temporal history / both diffusion history and action queue.
        adapter.reset();torch.manual_seed(42)
        repeated=adapter.infer(rgb[0].copy(),states[0].copy())
        np.testing.assert_allclose(repeated,outputs[0],rtol=1e-4,atol=1e-5)
        report={'kind':a.kind,'checkpoint_sha256':sha(a.run/'last_policy/model.safetensors'),
                'normalization_sha256':sha(a.run/'normalization.npz'),'calls':24,
                'inference_wall_s':times,'max_wall_s':max(times),'median_wall_s':float(np.median(times)),
                'reset_reproducible':True,'fresh_chunk_calls':times[::8] if a.kind=='Diffusion_100' else times,
                'latency_gate_pass':max(times)<.6,'guard_wall_s':.75,'guard_sim_s':.2}
        print(json.dumps(report),flush=True)
        if not report['latency_gate_pass']: raise RuntimeError('Inference preflight exceeds 0.6s headroom gate; no robot trial')
        return
    adapter.reset();torch.manual_seed(42)
    with socket.socket(socket.AF_UNIX) as server:
        server.bind(str(a.socket));a.socket.chmod(0o600)
        try:
            server.listen(1);server.settimeout(180);print('READY',flush=True)
            conn,_=server.accept()
            with conn,conn.makefile('rwb') as stream:
                conn.settimeout(600)
                for line in stream:
                    row=json.loads(line)
                    if row.get('reset_policy',False): adapter.reset()
                    image=np.frombuffer(base64.b64decode(row['rgb']),np.uint8).reshape(240,320,3)
                    start=time.monotonic();action=adapter.infer(image,row['state'])
                    stream.write((json.dumps({'id':row['id'],'action':action,'inference_wall_s':time.monotonic()-start})+'\n').encode());stream.flush()
        finally: a.socket.unlink(missing_ok=True)

if __name__=='__main__':main()
