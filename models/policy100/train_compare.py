#!/usr/bin/env python3
"""Offline, uniform-sampling ACT / Diffusion baseline on 100 expert episodes.

No ROS imports, simulator commands, collection, uploads, or automatic live tests.
"""
import argparse
import gc
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess as sp
import sys
import time
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'models/act_v1'))
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from train_act import prepare_episode, Chunks, JOINTS, IMAGE, atomic_json, sha
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler

STOP=False
def stopped(*_):
    global STOP
    STOP=True

def now(): return datetime.now(ZoneInfo('America/New_York')).isoformat()

def check_deadline(deadline):
    if STOP or time.time()>=deadline:
        raise InterruptedError('Training stop requested or resource deadline reached')

def check_workloads():
    ancestors=set(); pid=os.getpid()
    while pid>1:
        ancestors.add(pid)
        try: pid=int(Path('/proc',str(pid),'stat').read_text().split(') ',1)[1].split()[1])
        except OSError: break
    forbidden=('collect_resilient.py','cup_task.py','record_collection_smoke.py','rollout_policy_ros.py',
               'serve_policy.py','train_act.py','train_rotation_comparison.py','train_compare.py',
               'fea_job','fea_project','run_jax','fea-benchmark','abaqus','runSofa')
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        if int(p.parent.name) in ancestors: continue
        try: args=p.read_bytes().decode(errors='replace').split('\0')
        except OSError: continue
        if any(token in arg for arg in args[:3] for token in forbidden):
            raise RuntimeError('Conflicting workload: '+str(p)+' '+' '.join(args)[:240])
    gpu=sp.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True)
    other=[line for line in gpu.splitlines() if line.strip() and line.split(',')[0].strip()!=str(os.getpid())]
    if other: raise RuntimeError('Conflicting GPU compute: '+str(other))

def bounds(ticks,i):
    start=i
    while start>0 and ticks[start-1]==ticks[start]-1: start-=1
    end=i
    while end+1<len(ticks) and ticks[end+1]==ticks[end]+1: end+=1
    return start,end

class DiffusionChunks(Dataset):
    """Past/current observations, actions [t-1,t,...,t+14]; gap-safe padding."""
    def __init__(self,episodes,stats):
        self.stats=stats; self.episodes=[]; self.index=[]
        for entry in episodes:
            e={k:np.load(Path(entry['cache'])/(k+'.npy'),mmap_mode='r') for k in ('images','states','actions','ticks')}
            self.index.extend((len(self.episodes),i) for i in range(len(e['ticks'])))
            self.episodes.append(e)
    def __len__(self): return len(self.index)
    def __getitem__(self,idx):
        ep,i=self.index[idx]; e=self.episodes[ep]; lo,hi=bounds(e['ticks'],i)
        obs=[max(lo,i-1),i]
        targets=np.arange(i-1,i+15); pad=(targets<lo)|(targets>hi)
        actions=e['actions'][np.clip(targets,lo,hi)]
        states=e['states'][obs]
        rgb=e['images'][obs].astype(np.float32).transpose(0,3,1,2)/255
        rgb=(rgb-np.array([.485,.456,.406],np.float32)[None,:,None,None])/np.array([.229,.224,.225],np.float32)[None,:,None,None]
        return {IMAGE:torch.from_numpy(rgb),'observation.state':torch.from_numpy(((states-self.stats['state_center'])/self.stats['state_scale']).copy()),
                'action':torch.from_numpy(((actions-self.stats['action_center'])/self.stats['action_scale']).copy()),
                'action_is_pad':torch.from_numpy(pad)}

def prepare(out,update,deadline):
    old=ROOT/'models/act_v1/runs/P3_ACT_50'
    previous=json.loads((old/'dataset_manifest.json').read_text())['episodes']
    assert len(previous)==50 and {e['seed'] for e in previous}==set(range(50))
    batch=ROOT/'ros_backend1.1/runtime/batch50-20261004-seeds50-99'
    bs=json.loads((batch/'status.json').read_text())
    assert bs['status']=='completed' and bs['successful']==50 and bs['failed']==0
    assert not Path('/proc',str(bs['pid'])).exists()
    (out/'cache').mkdir()
    entries=[]
    for seed in range(100):
        check_deadline(deadline); update(preparing_seed=seed)
        if seed<50:
            entry=dict(next(e for e in previous if e['seed']==seed))
            ep=Path(entry['episode'])
            assert sha(ep/'summary.json')==entry['summary_sha256']
            assert sha(ep/'robot_data.jsonl')==entry['robot_data_sha256']
            target=Path(entry['cache']); assert target.is_dir()
            (out/'cache'/f'seed-{seed}').symlink_to(target,target_is_directory=True)
            entry['cache']=str(out/'cache'/f'seed-{seed}')
        else:
            candidates=[]
            for ep in batch.glob(f'episode-*-seed-{seed}-attempt-*'):
                if (ep/'summary.json').exists() and json.loads((ep/'summary.json').read_text()).get('status')=='controller_pass': candidates.append(ep)
            assert len(candidates)==1,(seed,candidates)
            entry=prepare_episode(candidates[0],out/'cache'/f'seed-{seed}')
        entry.update(seed=seed,split='validation' if seed%10 in (8,9) else 'train')
        cache=Path(entry['cache'])
        entry['cache_sha256']={k:sha(cache/(k+'.npy')) for k in ('images','states','actions','ticks')}
        entries.append(entry)
        atomic_json(out/'preparation_progress.json',{'episodes':entries})
    assert len({e['controller_sha256'] for e in entries})==1
    train=[e for e in entries if e['split']=='train']; val=[e for e in entries if e['split']=='validation']
    assert len(train)==80 and len(val)==20
    states=np.concatenate([np.load(Path(e['cache'])/'states.npy') for e in train])
    actions=np.concatenate([np.load(Path(e['cache'])/'actions.npy') for e in train])
    stats=dict(state_mean=states.mean(0),state_std=np.maximum(states.std(0),np.array([.001]*6+[.0001]*2,np.float32)),
               action_mean=actions.mean(0),action_std=np.maximum(actions.std(0),np.array([.001]*6+[.0001],np.float32)))
    dp={}
    for name,arr,floor in [('state',states,np.array([.001]*6+[.0001]*2,np.float32)),('action',actions,np.array([.001]*6+[.0001],np.float32))]:
        dp[name+'_center']=(arr.max(0)+arr.min(0))/2
        dp[name+'_scale']=np.maximum((arr.max(0)-arr.min(0))/2,floor)
    np.savez(out/'act_normalization.npz',**stats)
    np.savez(out/'diffusion_normalization.npz',**dp)
    manifest={'episodes':entries,'train_seeds':[e['seed'] for e in train],'validation_seeds':[e['seed'] for e in val],
              'total_samples':sum(e['samples'] for e in entries),'frequency_hz':10,'image_wh':[320,240],
              'state_names':JOINTS,'action_names':JOINTS[:6]+['common_gripper_position'],'action_units':['rad/s']*6+['m'],
              'sampling':'uniform shuffled observation windows; no balancing or loss weighting',
              'normalization':'training split only; ACT mean/std, diffusion min/max to [-1,1]',
              'command_timing':'previous receipt simulation clock; not exact actuation time'}
    atomic_json(out/'dataset_manifest.json',manifest)
    return train,val,stats,dp

def train_one(kind,out,train,val,stats,dp,steps,deadline,update,resume=None):
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    d=out/kind; d.mkdir()
    ds=Chunks(train,stats) if kind=='ACT_100' else DiffusionChunks(train,dp)
    vs=Chunks(val,stats) if kind=='ACT_100' else DiffusionChunks(val,dp)
    common=dict(input_features={IMAGE:PolicyFeature(type=FeatureType.VISUAL,shape=(3,240,320)),
                               'observation.state':PolicyFeature(type=FeatureType.STATE,shape=(8,))},
                output_features={'action':PolicyFeature(type=FeatureType.ACTION,shape=(7,))},device='cuda',push_to_hub=False)
    if kind=='ACT_100':
        config=ACTConfig(**common,chunk_size=20,n_action_steps=1,temporal_ensemble_coeff=.01,optimizer_lr=1e-4,optimizer_lr_backbone=1e-5)
        policy=ACTPolicy(config).cuda()
        optimizer=torch.optim.AdamW(policy.get_optim_params(),lr=1e-4,weight_decay=1e-4)
        scheduler=None; ema=None; norm=stats
    else:
        config=DiffusionConfig(**common,n_obs_steps=2,horizon=16,n_action_steps=8,
            num_inference_steps=100,do_mask_loss_for_padding=True,crop_shape=None,
            pretrained_backbone_weights=None,use_group_norm=True)
        policy=DiffusionPolicy(config).cuda()
        optimizer=torch.optim.Adam(policy.parameters(),lr=1e-4,betas=(.95,.999),eps=1e-8,weight_decay=1e-6)
        scheduler=get_scheduler('cosine',optimizer,num_warmup_steps=500,num_training_steps=steps)
        ema=EMAModel(policy.parameters(),power=.75,use_ema_warmup=True); norm=dp
    start_step=0
    if resume is not None:
        assert kind=='ACT_100', 'Continuation currently validated for ACT only'
        saved=torch.load(resume/'optimizer.pt',map_location='cuda',weights_only=False)
        assert saved['step']==5000 and saved['scheduler'] is None and saved['ema'] is None
        loaded=ACTPolicy.from_pretrained(resume/'last_raw_policy').cuda()
        policy.load_state_dict(loaded.state_dict());del loaded
        optimizer.load_state_dict(saved['optimizer']);start_step=saved['step']
        assert all(int(s['step'])==start_step for s in optimizer.state.values())
        for key in norm:
            np.testing.assert_array_equal(norm[key],np.load(resume/'normalization.npz')[key])
    config.save_pretrained(d/'policy_config'); np.savez(d/'normalization.npz',**norm)
    (d/'cache').symlink_to(out/'cache',target_is_directory=True)
    loader=DataLoader(ds,batch_size=8,shuffle=True,num_workers=0,drop_last=True)
    # 8 evenly spaced observations per validation episode: same anchors for both models.
    indices=[]; offset=0
    for e in val:
        indices.extend(offset+int(i) for i in np.linspace(0,e['samples']-1,8)); offset+=e['samples']
    validation=DataLoader(torch.utils.data.Subset(vs,indices),batch_size=8,num_workers=0)
    gpu=lambda batch:{k:v.cuda() for k,v in batch.items()}
    def predict(batch):
        if kind=='ACT_100': return policy.predict_action_chunk({k:v for k,v in batch.items() if k not in ('action','action_is_pad')})[:,0]
        inputs={'observation.state':batch['observation.state'],'observation.images':batch[IMAGE].unsqueeze(2)}
        return policy.diffusion.generate_actions(inputs)[:,0]
    def evaluate():
        policy.eval(); errors=[]; preds=[]; labels=[]
        with torch.random.fork_rng(devices=[0]), torch.no_grad():
            torch.manual_seed(1234)
            for batch in validation:
                check_deadline(deadline)
                batch=gpu(batch); pred=predict(batch)
                target=batch['action'][:,0 if kind=='ACT_100' else 1]
                scale=stats['action_std'] if kind=='ACT_100' else dp['action_scale']
                center=stats['action_mean'] if kind=='ACT_100' else dp['action_center']
                pred=pred.cpu().numpy()*scale+center; target=target.cpu().numpy()*scale+center
                assert np.isfinite(pred).all()
                preds.append(pred);labels.append(target);errors.append(np.abs(pred-target))
        policy.train(); err=np.concatenate(errors); pr=np.concatenate(preds); gt=np.concatenate(labels)
        rot=np.abs(gt[:,5])>.02
        return {'mae_per_action':err.mean(0).tolist(),'shared_scale_normalized_mae':float((err/stats['action_std']).mean()),
                'rotation_wrist_mae':float(err[rot,5].mean()) if rot.any() else None,
                'samples':len(err),'offline_only':True},pr,gt
    def save_checkpoint(step,name='last_policy'):
        policy.save_pretrained(d/name)
        torch.save({'step':step,'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict() if scheduler else None,
                    'ema':ema.state_dict() if ema else None},d/'optimizer.pt')
    info={'status':'training','step':0,'steps':steps,'parameters':sum(p.numel() for p in policy.parameters()),'started':now(),
          'train_samples':len(ds),'validation_samples':len(vs),'dataset_sha256':sha(out/'dataset_manifest.json'),
          'ema':ema is not None,'normalization_applied_externally':True}
    def report(**kw):
        info.update(kw,updated=now());atomic_json(d/'status.json',info);update(active_model=kind,model_progress=info)
    best=float('inf'); iterator=iter(loader); start=time.monotonic(); policy.train(); step=start_step
    if resume is not None:
        metric,pr,gt=evaluate();metric['step']=start_step
        baseline=json.loads((resume/'status.json').read_text())['validation']
        np.testing.assert_allclose(metric['mae_per_action'],baseline['mae_per_action'],rtol=1e-4,atol=1e-7)
        best=metric['shared_scale_normalized_mae']
        policy.save_pretrained(d/'best_policy');atomic_json(d/'best_metrics.json',metric)
        with (d/'metrics.jsonl').open('a') as f:f.write(json.dumps(metric)+'\n')
        info.update(step=start_step,resume_from=str(resume),resume_optimizer_verified=True,baseline_replay_pass=True,
                    rng_note='Fresh seed-42 shuffled stream; original RNG/sampler state was not saved. Not bitwise continuation.')
    report()
    try:
        for step in range(start_step+1,steps+1):
            check_deadline(deadline)
            if step%100==1: check_workloads()
            try: batch=next(iterator)
            except StopIteration: iterator=iter(loader);batch=next(iterator)
            loss,_=policy(gpu(batch)); assert torch.isfinite(loss),'Nonfinite loss'
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(policy.parameters(),1);optimizer.step()
            if scheduler: scheduler.step()
            if ema: ema.step(policy.parameters())
            time.sleep(.02)
            if step==1 or step%25==0:
                elapsed=time.monotonic()-start
                report(step=step,loss=float(loss),elapsed_s=elapsed,estimated_remaining_s=(steps-step)*elapsed/(step-start_step))
            if step%1000==0 or step==steps:
                # Preserve raw model+optimizer for reproducible continuation; evaluate EMA for diffusion.
                save_checkpoint(step,'last_raw_policy')
                if ema: ema.store(policy.parameters());ema.copy_to(policy.parameters())
                try:
                    metric,pr,gt=evaluate();metric['step']=step
                    with (d/'metrics.jsonl').open('a') as f:f.write(json.dumps(metric)+'\n')
                    policy.save_pretrained(d/'last_policy')
                    if metric['shared_scale_normalized_mae']<best:
                        best=metric['shared_scale_normalized_mae'];policy.save_pretrained(d/'best_policy');atomic_json(d/'best_metrics.json',metric)
                    np.savez(d/'validation_predictions.npz',prediction=pr,target=gt,indices=indices)
                    report(validation=metric)
                    if step==steps:
                        probe=gpu(next(iter(validation)))
                        with torch.no_grad(),torch.random.fork_rng(devices=[0]):
                            torch.manual_seed(1234);policy.eval();expected=predict(probe)
                            loaded=type(policy).from_pretrained(d/'last_policy').cuda().eval()
                            original=policy;policy=loaded;torch.manual_seed(1234);actual=predict(probe)
                            torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-5)
                            policy=original;del loaded
                        report(checkpoint_reload_pass=True,last_policy_sha256=sha(d/'last_policy/model.safetensors'))
                finally:
                    if ema: ema.restore(policy.parameters())
        report(status='completed',finished=now(),closed_loop_evaluated=False)
    except InterruptedError:
        save_checkpoint(max(0,step-1),'interrupted_raw_policy');report(status='stopped_at_deadline',finished=now());raise
    except Exception:
        report(status='failed',error=traceback.format_exc());raise
    del policy,optimizer,ema,loader,iterator
    gc.collect();torch.cuda.empty_cache()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--steps',type=int,default=5000)
    parser.add_argument('--deadline',required=True);a=parser.parse_args()
    deadline=datetime.fromisoformat(a.deadline).timestamp()
    signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
    torch.set_num_threads(2)
    import cv2
    cv2.setNumThreads(1)
    with open('/tmp/baseline_collection_batch.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);check_workloads();check_deadline(deadline)
        a.output.mkdir(parents=True,exist_ok=False)
        status={'status':'preparing','pid':os.getpid(),'started':now(),'steps_per_model':a.steps,'deadline':a.deadline,'trainer_sha256':sha(Path(__file__))}
        def update(**kw):
            status.update(kw,updated=now());atomic_json(a.output/'status.json',status);print(json.dumps(status),flush=True)
        update()
        try:
            assert torch.cuda.is_available(),'No CUDA; refusing CPU fallback'
            torch.cuda.set_per_process_memory_fraction(.55)
            train,val,stats,dp=prepare(a.output,update,deadline)
            update(status='training')
            for kind in ('ACT_100','Diffusion_100'):
                check_workloads();check_deadline(deadline)
                train_one(kind,a.output,train,val,stats,dp,a.steps,deadline,update)
            update(status='completed',finished=now(),closed_loop_evaluated=False)
        except InterruptedError as e: update(status='stopped_at_deadline',error=str(e),finished=now())
        except Exception: update(status='failed',error=traceback.format_exc(),finished=now());raise

if __name__=='__main__': main()
