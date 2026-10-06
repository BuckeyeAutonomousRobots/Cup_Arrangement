"""One frozen-CNN + state MLP behavior-cloning candidate; no privileged inputs."""
import fcntl,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
from train_compare import ROOT,check_workloads,atomic_json,sha
from probe_observability import Probe
sys.path.insert(0,str(ROOT/'models/act_v1'))
from replay_temporal_comparison import metrics

OUTPUT_NAME='FeatureBC_20261005'
STEPS=3000
RESUME_NAME=None
HOLD_WEIGHT=1.
LEARNING_RATE=.001

def main():
    start=time.monotonic()
    with open('/tmp/baseline_collection_batch.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);check_workloads()
        torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.4);torch.manual_seed(42)
        root=ROOT/'models/policy100/runs';features=root/'observability_20261005';source=root/'ACT_Diffusion_100_20261004'
        out=root/OUTPUT_NAME;assert not out.exists();out.mkdir()
        with out.with_suffix('.claim').open('x') as f:json.dump({'pid':os.getpid(),'steps':STEPS,'hold_weight':HOLD_WEIGHT,'resume':RESUME_NAME,'script_sha256':sha(Path(__file__)),'offline_only':True},f)
        manifest=json.loads((source/'dataset_manifest.json').read_text());stats=dict(np.load(features/'feature_normalization.npz'))
        np.savez(out/'normalization.npz',**stats)
        rows={};splits={'train':[],'validation':[]}
        for e in manifest['episodes']:
            d=dict(np.load(features/'features'/f'seed-{e["seed"]}.npz'));rows[e['seed']]=d
            f=(d['features']-stats['feature_mean'])/stats['feature_std'];q=(d['states']-stats['state_mean'])/stats['state_std'];x=np.concatenate([f,q],1)
            y=(d['actions']-stats['action_mean'])/stats['action_std'];cls=np.where(d['actions'][:,5]<-.02,1,np.where(d['actions'][:,5]>.02,2,0))
            rot=cls!=0;ticks=d['ticks'];steady=np.array([not np.any(rot[np.abs(ticks-t)<=10]) for t in ticks])
            splits[e['split']].append((x,y,cls,e['seed'],steady))
        x=torch.from_numpy(np.concatenate([r[0] for r in splits['train']])).cuda();y=torch.from_numpy(np.concatenate([r[1] for r in splits['train']])).cuda();labels=torch.from_numpy(np.concatenate([r[2] for r in splits['train']])).cuda()
        steady=torch.from_numpy(np.concatenate([r[4] for r in splits['train']])).cuda()
        model=Probe(x.shape[1],10).cuda()
        if RESUME_NAME:model.load_state_dict(torch.load(root/RESUME_NAME/'model.pt',weights_only=True))
        opt=torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE,weight_decay=.0001)
        report={'status':'training','pid':os.getpid(),'dataset_sha256':sha(source/'dataset_manifest.json'),
                'input':'Frozen ACT5k CNN 2x3 spatial features + measured eight joints; no poses',
                'action':'Same six joint velocities and common jaw target, externally normalized',
                'architecture':'3072+8 ->128 ReLU ->128 ReLU ->10; first7 action regression, last3 auxiliary hold/negative/positive classification',
                'loss':'normalized action MSE +0.1 phase cross entropy; steady-training-hold factor applied only to wrist error and phase CE; uniform replacement sampling batch256',
                'steady_hold_weight':HOLD_WEIGHT,'learning_rate':LEARNING_RATE,'resume':RESUME_NAME,
                'resume_sha256':sha(root/RESUME_NAME/'model.pt') if RESUME_NAME else None,
                'backbone_checkpoint_sha256':sha(source/'ACT_100/last_policy/model.safetensors'),
                'steps':STEPS,'seed':42,'validation_used_for_training':False,'history':[]}
        for step in range(1,STEPS+1):
            assert time.monotonic()-start<300
            if step%250==1:check_workloads()
            idx=torch.randint(len(x),(256,),device='cuda');p=model(x[idx]);weight=1+(HOLD_WEIGHT-1)*steady[idx].float()
            errors=(p[:,:7]-y[idx]).square();errors[:,5]=errors[:,5]*weight;reg=errors.mean()
            ce=(torch.nn.functional.cross_entropy(p[:,7:],labels[idx],reduction='none')*weight).mean();loss=reg+.1*ce
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
            if step%500==0:
                row={'step':step,'loss':float(loss),'regression_mse':float(reg),'phase_ce':float(ce)};report['history'].append(row);atomic_json(out/'summary.json',report);print(json.dumps(row),flush=True)
        torch.save(model.state_dict(),out/'model.pt');loaded=Probe(x.shape[1],10).cuda();loaded.load_state_dict(torch.load(out/'model.pt',weights_only=True))
        model.eval();loaded.eval()
        with torch.no_grad():torch.testing.assert_close(model(x[:8]),loaded(x[:8]));del loaded
        entries=[]
        with torch.no_grad():
            for xx,yy,cls,seed,unused_steady in splits['validation']:
                v=torch.from_numpy(xx).cuda();pred=torch.cat([model(v[i:i+512])[:,:7].cpu() for i in range(0,len(v),512)]).numpy()
                raw=pred*stats['action_std']+stats['action_mean'];d=rows[seed];target=d['actions'];ticks=d['ticks'];m=metrics(raw,target,ticks)
                idx=np.flatnonzero((np.abs(target[:,5])>.02)&(np.abs(raw[:,5]-target[:,5])>.05))
                m.update(seed=seed,first_rotation_error_over_005_elapsed_s=float((ticks[idx[0]]-ticks[0])/10) if len(idx) else None);entries.append(m)
                np.savez(out/f'validation-seed-{seed}.npz',raw=raw,expert=target,ticks=ticks)
        n=sum(e['samples'] for e in entries);nr=sum(e['rotation_samples'] for e in entries);nh=sum(e['steady_samples'] for e in entries)
        agg={'samples':n,'mae_per_action':(sum(np.array(e['applied_mae_per_action'])*e['samples'] for e in entries)/n).tolist(),
            'rotation_mae':sum(e['rotation_mae_rad_s']*e['rotation_samples'] for e in entries)/nr,
            'steady_false_rotation':sum((e['steady_false_rotation_fraction'] or 0)*e['steady_samples'] for e in entries)/nh,
            'clamp_fraction':sum(e['arm_clamp_samples'] for e in entries)/n}
        base=json.loads((root/'wrist_candidate_replay_20261005/summary.json').read_text())['arms']['baseline']['aggregate']
        seed8=next(e for e in entries if e['seed']==8);ratio=seed8['predicted_approach_command_deg']/seed8['expert_approach_command_deg']
        gates={'rotation_improved':agg['rotation_mae']<=.8*base['rotation_mae'],'steady_hold':agg['steady_false_rotation']<=.01,
            'other_arm_channels':all(agg['mae_per_action'][i]<=1.15*base['mae_per_action'][i]+.001 for i in range(5)),
            'jaw':agg['mae_per_action'][6]<=1.15*base['mae_per_action'][6]+.00005,'clamps':agg['clamp_fraction']<=.05,'seed8_approach_rotation':.5<=ratio<=1.5}
        report.update(status='completed',model_sha256=sha(out/'model.pt'),checkpoint_reload_pass=True,episodes=entries,aggregate=agg,
            live_preflight_gates=gates,eligible_for_live_preflight=all(gates.values()),wall_seconds=time.monotonic()-start,
            limitations='Pointwise offline teacher-forced predictions; no ACT ensemble, robot dynamics or live timing. Not closed-loop success.')
        atomic_json(out/'summary.json',report);print(json.dumps({'aggregate':agg,'gates':gates,'seed8':seed8}),flush=True)

        if RESUME_NAME:assert sha(root/RESUME_NAME/'model.pt')==report['resume_sha256']

if __name__=='__main__':main()
