"""Bounded offline sampler/loss audit and 200-update memorization diagnostic."""
import fcntl,json,os,random,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from train_compare import ROOT,Chunks,IMAGE,ACTPolicy,check_workloads,atomic_json,sha

def main():
    with open('/tmp/baseline_collection_batch.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);check_workloads()
        torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.4)
        random.seed(42);np.random.seed(42);torch.manual_seed(42)
        source=ROOT/'models/policy100/runs/ACT_Diffusion_100_20261004';run=source/'ACT_100'
        out=ROOT/'models/policy100/runs/wrist_learning_audit_20261005'
        assert not out.exists();out.mkdir()
        manifest=json.loads((source/'dataset_manifest.json').read_text());train=[e for e in manifest['episodes'] if e['split']=='train']
        stats=dict(np.load(run/'normalization.npz'));ds=Chunks(train,stats)
        targets=[];lengths=[];offset=0;picks={-1:[],1:[]}
        for e,entry in zip(ds.episodes,train):
            a=e['actions'];ticks=e['ticks'];n=len(ticks);ln=np.ones(n,dtype=int)
            for i in range(n-2,-1,-1):
                if ticks[i+1]==ticks[i]+1:ln[i]=min(20,ln[i+1]+1)
            lengths.extend(ln);targets.append(a)
            for sign in (-1,1):
                valid=np.flatnonzero((a[:,5]*sign>.1)&(ln==20))
                if len(valid) and len(picks[sign])<4:picks[sign].append((offset+int(valid[len(valid)//2]),entry['seed'],int(valid[len(valid)//2])))
            offset+=n
        targets=np.concatenate(targets);lengths=np.array(lengths);rot=np.abs(targets[:,5])>.02
        chosen=picks[-1]+picks[1];assert len(chosen)==8
        result={'source_sha256':sha(run/'last_policy/model.safetensors'),'dataset_sha256':sha(source/'dataset_manifest.json'),
            'normalization':{k:v.tolist() for k,v in stats.items()},'train_windows':len(ds),
            'rotation_first_action_fraction':float(rot.mean()),'mean_valid_chunk_length':float(lengths.mean()),
            'rotation_mean_valid_length':float(lengths[rot].mean()),'nonrotation_mean_valid_length':float(lengths[~rot].mean()),
            'batch_indices_seed_local':chosen,'diagnostic_only':True}
        # Reproducible uniform-shuffle probe, not reconstruction of the historical RNG stream.
        gen=torch.Generator().manual_seed(42);order=torch.randperm(len(ds),generator=gen).numpy()[:5000]
        result['sampled_rotation_fraction_first5000_windows']=float(rot[order].mean())
        batch=next(iter(DataLoader(Subset(ds,[x[0] for x in chosen]),batch_size=8)))
        batch={k:v.cuda() for k,v in batch.items()}
        policy=ACTPolicy.from_pretrained(run/'last_policy').cuda()
        sums=np.zeros(7);rot_sums=np.zeros(7);hold_sums=np.zeros(7);counts=np.zeros(3)
        policy.eval()
        with torch.no_grad():
            for probe in DataLoader(Subset(ds,order[:64].tolist()),batch_size=8):
                probe={k:v.cuda() for k,v in probe.items()}
                pred=policy.predict_action_chunk({k:v for k,v in probe.items() if k not in ('action','action_is_pad')})
                err=(pred-probe['action']).abs();valid=~probe['action_is_pad']
                isrot=(probe['action'][:,:,5]*float(stats['action_std'][5])+float(stats['action_mean'][5])).abs()>.02
                sums+=err[valid].sum(0).cpu().numpy();rot_sums+=err[valid&isrot].sum(0).cpu().numpy();hold_sums+=err[valid&~isrot].sum(0).cpu().numpy()
                counts+=np.array([int(valid.sum()),int((valid&isrot).sum()),int((valid&~isrot).sum())])
        result['uniform64_prior_loss_probe']={'valid_slots_total_rotation_nonrotation':counts.tolist(),
            'normalized_mae_per_channel':(sums/counts[0]).tolist(),'rotation_phase_mae_per_channel':(rot_sums/max(counts[1],1)).tolist(),
            'nonrotation_phase_mae_per_channel':(hold_sums/max(counts[2],1)).tolist(),
            'absolute_loss_share_per_channel':(sums/sums.sum()).tolist()}
        mask=~batch['action_is_pad'].unsqueeze(-1)
        scale=torch.as_tensor(stats['action_std'],device='cuda');center=torch.as_tensor(stats['action_mean'],device='cuda')
        def evaluate():
            policy.eval()
            with torch.no_grad():
                pred=policy.predict_action_chunk({k:v for k,v in batch.items() if k not in ('action','action_is_pad')})
                err=(pred-batch['action']).abs()*mask
                return {'prior_normalized_mae':err.mean((0,1)).cpu().tolist(),
                    'prior_first_action':(pred[:,0]*scale+center).cpu().tolist(),
                    'target_first_action':(batch['action'][:,0]*scale+center).cpu().tolist()}
        result['before']=evaluate();policy.train()
        with torch.no_grad():
            posterior=dict(batch);posterior['observation.images']=[batch[IMAGE]]
            p,(mu,lv)=policy.model(posterior)
            result['posterior_before_normalized_mae']=((p-batch['action']).abs()*mask).mean((0,1)).cpu().tolist()
            result['posterior_before_kl']=float((-.5*(1+lv-mu.pow(2)-lv.exp())).sum(-1).mean())
        optimizer=torch.optim.AdamW(policy.get_optim_params(),lr=1e-4,weight_decay=1e-4)
        start=time.monotonic();history=[]
        for step in range(1,201):
            assert time.monotonic()-start<180,'Diagnostic exceeded three minutes'
            if step%50==1:check_workloads()
            policy.train();loss,parts=policy(batch);assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(policy.parameters(),1);optimizer.step()
            if step%50==0:
                row={'step':step,'loss':float(loss),**parts,**evaluate()};history.append(row);print(json.dumps(row),flush=True)
        result['history']=history;result['elapsed_s']=time.monotonic()-start
        assert sha(run/'last_policy/model.safetensors')==result['source_sha256']
        atomic_json(out/'summary.json',result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('normalization','history','before')}),flush=True)

if __name__=='__main__':main()
