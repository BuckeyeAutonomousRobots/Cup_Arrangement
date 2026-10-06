#!/usr/bin/env python3
"""Paired offline ACT training; read-only source cache, no ROS/control interfaces."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import time
import traceback
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from train_act import Chunks, atomic_json, now, sha


def rotating_windows(dataset):
    flags=[]
    for ep,i in dataset.index:
        e=dataset.episodes[ep]; n=1
        while n<dataset.chunk and i+n<len(e['ticks']) and e['ticks'][i+n]==e['ticks'][i]+n:n+=1
        flags.append(bool(np.any(np.abs(e['actions'][i:i+n,5])>.02)))
    return np.asarray(flags)


def state_hash(policy):
    h=hashlib.sha256()
    for k,v in policy.state_dict().items():
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--collection',type=Path,required=True)
    p.add_argument('--steps',type=int,default=5000)
    args=p.parse_args(); args.output.mkdir(exist_ok=False)
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(.20)
    status=dict(status='preparing',pid=os.getpid(),started=now(),steps_per_arm=args.steps,
                source=str(args.source),arms={},script_sha256=sha(Path(__file__)))
    def update(**kw):
        status.update(kw,updated=now()); atomic_json(args.output/'status.json',status)
    def collection_guard():
        c=json.loads((args.collection/'status.json').read_text())
        if c['status'] not in ['running','completed']:
            raise RuntimeError('Collection requires attention; stopping training only: '+c['status'])
        if c['status']=='running' and c.get('stage')=='recording':
            episode=c['attempts'][-1]['directory']; frames=args.collection/episode/'frames.jsonl'
            if frames.exists() and time.time()-frames.stat().st_mtime>45:
                raise RuntimeError('No frame manifest update in 45s; stopping training only')
    update()
    try:
        collection_guard()
        manifest=json.loads((args.source/'dataset_manifest.json').read_text())
        stats=dict(np.load(args.source/'normalization.npz'))
        train=Chunks([e for e in manifest['episodes'] if e['split']=='train'],stats)
        val=Chunks([e for e in manifest['episodes'] if e['split']=='validation'],stats)
        flags=rotating_windows(train); positive=int(flags.sum()); negative=len(flags)-positive
        assert positive and negative
        atomic_json(args.output/'design.json',dict(source_manifest_sha256=sha(args.source/'dataset_manifest.json'),
            source_normalization_sha256=sha(args.source/'normalization.npz'),source_checkpoint_sha256=sha(args.source/'best_policy/model.safetensors'),
            train_seeds=manifest['train_seeds'],validation_seeds=manifest['validation_seeds'],train_windows=len(train),
            rotation_windows=positive,nonrotation_windows=negative,natural_rotation_window_fraction=float(flags.mean()),
            rotation_definition='Any VALID action in a 20-tick contiguous window has abs(wrist3 velocity)>0.02 rad/s',
            sampling='Both arms use replacement sampling. Control uniform; balanced assigns 50% total probability to each window class.',
            seed=42,batch_size=8,steps=args.steps,interstep_sleep_s=.1,
            primary_comparison='Final checkpoint at fixed step budget; same 256 validation windows every 500 steps; all validation first actions at final',
            unchanged='Architecture, loss, optimizer, normalization, dataset split; no new collection episodes included',
            resources='2 CPU threads; no loader workers; 20% CUDA memory cap; 100ms interstep yield; training-only stop on collection terminal problem or stale recording'))
        def to_gpu(batch):return {k:v.cuda() for k,v in batch.items()}
        indices=np.linspace(0,len(val)-1,min(256,len(val)),dtype=int).tolist()
        validation=DataLoader(torch.utils.data.Subset(val,indices),batch_size=8,num_workers=0)
        first_initial_hash=None
        for arm in ['uniform_control','rotation_balanced']:
            collection_guard(); dest=args.output/arm;dest.mkdir()
            random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
            config=PreTrainedConfig.from_pretrained(args.source/'policy_config')
            assert isinstance(config, ACTConfig)
            config.device='cuda';config.push_to_hub=False
            policy=ACTPolicy(config).cuda(); initial_hash=state_hash(policy)
            if first_initial_hash is None:first_initial_hash=initial_hash
            assert initial_hash==first_initial_hash, 'Initialization differs'
            optimizer=torch.optim.AdamW(policy.get_optim_params(),lr=1e-4,weight_decay=1e-4)
            weights=np.ones(len(train)) if arm=='uniform_control' else np.where(flags,.5/positive,.5/negative)
            generator=torch.Generator().manual_seed(42)
            sampler=WeightedRandomSampler(torch.as_tensor(weights,dtype=torch.double),args.steps*8,replacement=True,generator=generator)
            loader=DataLoader(train,batch_size=8,sampler=sampler,num_workers=0)
            np.savez(dest/'normalization.npz',**stats)
            armstate=dict(status='training',initial_state_sha256=initial_hash,step=0)
            status['arms'][arm]=armstate;update(active_arm=arm,status='training')
            def evaluate(loader,first_only=False):
                policy.eval(); total=np.zeros(7);counts={'rotation':0,'holding':0}; errors={k:0. for k in counts};sign_ok=0;false_rotation=0;norm=0.;n=0; predictions=[]
                with torch.inference_mode():
                    for batch in loader:
                        collection_guard(); batch=to_gpu(batch)
                        pred=policy.predict_action_chunk({k:v for k,v in batch.items() if k not in ['action','action_is_pad']})
                        if not torch.isfinite(pred).all():raise ValueError('Nonfinite validation output')
                        target=batch['action']; mask=~batch['action_is_pad']
                        if first_only:pred=pred[:,:1];target=target[:,:1];mask=mask[:,:1]
                        err=(pred-target).abs(); valid=mask.unsqueeze(-1)
                        norm+=float((err*valid).sum()); n+=int(mask.sum())
                        physical=err.cpu().numpy()*stats['action_std'];total+=physical[mask.cpu().numpy()].sum(0)
                        pvel=pred[:,:,5].cpu().numpy()*stats['action_std'][5]+stats['action_mean'][5]
                        tvel=target[:,:,5].cpu().numpy()*stats['action_std'][5]+stats['action_mean'][5]
                        m=mask.cpu().numpy();rot=m&(np.abs(tvel)>.02);hold=m&~rot
                        for name,which in [('rotation',rot),('holding',hold)]:
                            counts[name]+=int(which.sum());errors[name]+=float(np.abs(pvel-tvel)[which].sum())
                        sign_ok+=int(((pvel*tvel>0)&(np.abs(pvel)>.02)&rot).sum())
                        false_rotation+=int(((np.abs(pvel)>.02)&hold).sum())
                        if first_only:predictions.append(pred[:,0].cpu().numpy()*stats['action_std']+stats['action_mean'])
                        time.sleep(.05)
                policy.train()
                metrics=dict(normalized_l1=norm/(n*7),mae_per_action=(total/n).tolist(),counts=counts,
                    wrist_mae_by_phase={k:errors[k]/max(1,counts[k]) for k in counts},
                    correct_direction_nontrivial_rotation_fraction=sign_ok/max(1,counts['rotation']),
                    false_rotation_during_holding_fraction=false_rotation/max(1,counts['holding']))
                return metrics,predictions
            best=float('inf'); start=time.monotonic()
            with (dest/'metrics.jsonl').open('x') as log:
                for step,batch in enumerate(loader,1):
                    if step%25==1:collection_guard()
                    loss,_=policy(to_gpu(batch))
                    if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                    optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(policy.parameters(),1);optimizer.step()
                    time.sleep(.1)
                    if step==1 or step%25==0:
                        armstate.update(step=step,loss=float(loss),elapsed_s=time.monotonic()-start,
                            estimated_remaining_s=(args.steps-step)*(time.monotonic()-start)/step)
                        update();print(json.dumps(dict(arm=arm,**armstate)),flush=True)
                    if step%500==0 or step==args.steps:
                        metrics,_=evaluate(validation);log.write(json.dumps(dict(step=step,**metrics))+'\n');log.flush()
                        if metrics['normalized_l1']<best:
                            best=metrics['normalized_l1'];policy.save_pretrained(dest/'best_policy')
                        armstate['validation']=metrics;update()
            policy.eval();policy.save_pretrained(dest/'last_policy')
            final,preds=evaluate(DataLoader(val,batch_size=8,num_workers=0),first_only=True)
            np.save(dest/'validation_first_actions.npy',np.concatenate(preds))
            atomic_json(dest/'final_metrics.json',final)
            probe=to_gpu(next(iter(validation)));probe={k:v for k,v in probe.items() if k not in ['action','action_is_pad']}
            policy.eval()
            with torch.inference_mode():expected=policy.predict_action_chunk(probe).cpu()
            del policy,optimizer,loader;gc.collect();torch.cuda.empty_cache()
            restored=ACTPolicy.from_pretrained(dest/'last_policy').cuda().eval()
            with torch.inference_mode():actual=restored.predict_action_chunk(probe).cpu()
            torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-5)
            del restored,probe;gc.collect();torch.cuda.empty_cache()
            armstate.update(status='completed',checkpoint_reload_pass=True,final_metrics=final);update()
        update(status='completed',finished=now(),closed_loop_evaluated=False)
    except Exception:
        update(status='failed',error=traceback.format_exc(),finished=now());raise


if __name__=='__main__':main()
