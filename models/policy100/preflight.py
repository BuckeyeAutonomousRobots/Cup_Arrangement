"""Bounded GPU forward/backward and diffusion action-queue checks; no robot I/O."""
import fcntl
import gc
import json
import torch
from train_compare import check_workloads, IMAGE, ACTConfig, ACTPolicy, DiffusionConfig, DiffusionPolicy, FeatureType, PolicyFeature

with open('/tmp/baseline_collection_batch.lock','a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);check_workloads()
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.55)
    common=dict(input_features={IMAGE:PolicyFeature(type=FeatureType.VISUAL,shape=(3,240,320)),
            'observation.state':PolicyFeature(type=FeatureType.STATE,shape=(8,))},
            output_features={'action':PolicyFeature(type=FeatureType.ACTION,shape=(7,))},device='cuda',push_to_hub=False)
    for kind in ('act','diffusion'):
        torch.manual_seed(42)
        if kind=='act':
            policy=ACTPolicy(ACTConfig(**common,chunk_size=20,n_action_steps=1,temporal_ensemble_coeff=.01)).cuda()
            batch={IMAGE:torch.zeros(2,3,240,320,device='cuda'),'observation.state':torch.zeros(2,8,device='cuda'),
                   'action':torch.zeros(2,20,7,device='cuda'),'action_is_pad':torch.zeros(2,20,dtype=torch.bool,device='cuda')}
        else:
            policy=DiffusionPolicy(DiffusionConfig(**common,n_obs_steps=2,horizon=16,n_action_steps=8,
                num_inference_steps=100,do_mask_loss_for_padding=True,crop_shape=None)).cuda()
            batch={IMAGE:torch.zeros(2,2,3,240,320,device='cuda'),'observation.state':torch.zeros(2,2,8,device='cuda'),
                   'action':torch.zeros(2,16,7,device='cuda'),'action_is_pad':torch.zeros(2,16,dtype=torch.bool,device='cuda')}
        loss,_=policy(batch);assert torch.isfinite(loss);loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None)
        policy.eval();policy.reset()
        obs={IMAGE:torch.zeros(2,3,240,320,device='cuda'),'observation.state':torch.zeros(2,8,device='cuda')}
        with torch.no_grad(): action=policy.select_action(obs)
        assert action.shape==(2,7) and torch.isfinite(action).all()
        print(json.dumps({'model':kind,'forward_backward_finite':True,'select_action_shape':list(action.shape),'parameters':sum(p.numel() for p in policy.parameters())}),flush=True)
        del policy,batch,loss,action;gc.collect();torch.cuda.empty_cache()
