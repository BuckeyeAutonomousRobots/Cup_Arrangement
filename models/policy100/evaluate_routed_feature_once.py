"""One learned-head coupling test, fixed confidence .95; no runtime phase labels."""
import fcntl,json,os,sys
from pathlib import Path
import numpy as np
import torch
from train_compare import ROOT,check_workloads,atomic_json,sha
from probe_observability import Probe
sys.path.insert(0,str(ROOT/'models/act_v1'))
from replay_temporal_comparison import metrics

def route(raw,logits):
    prob=torch.softmax(logits,dim=-1);confidence,kind=prob.max(-1)
    direction=torch.where(kind==1,-1.,torch.where(kind==2,1.,0.))
    active=(confidence>=.95)&(kind!=0)&(raw[:,5]*direction>0)
    result=raw.clone();result[:,5]=torch.where(active,raw[:,5],0.)
    return result

def main():
    with open('/tmp/baseline_collection_batch.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);check_workloads();torch.set_num_threads(2)
        root=ROOT/'models/policy100/runs';src=root/'FeatureBC_hold_20261005';out=root/'FeatureBC_routed_20261005'
        if out.exists():
            assert not (out/'routing.json').exists() and not (out/'summary.json').exists(),'Already evaluated'
            assert json.loads(out.with_suffix('.claim').read_text())['source_sha256']==sha(src/'model.pt')
        else:
            out.mkdir()
            with out.with_suffix('.claim').open('x') as f:json.dump({'pid':os.getpid(),'threshold':.95,'source_sha256':sha(src/'model.pt'),'offline_only':True},f)
            (out/'model.pt').symlink_to(src/'model.pt');(out/'normalization.npz').symlink_to(src/'normalization.npz')
        metadata={'confidence_threshold':.95,'source':str(src),'head_sha256':sha(src/'model.pt'),'normalization_sha256':sha(src/'normalization.npz'),
            'routing':'wrist velocity is zero unless learned rotation class confidence >=.95 and velocity sign agrees; other six outputs unchanged',
            'script_sha256':sha(Path(__file__))}
        atomic_json(out/'routing.json',metadata)
        s=dict(np.load(src/'normalization.npz'));model=Probe(3080,10).cuda();model.load_state_dict(torch.load(src/'model.pt',weights_only=True));model.eval()
        manifest=json.loads((root/'ACT_Diffusion_100_20261004/dataset_manifest.json').read_text());entries=[]
        for e in manifest['episodes']:
            if e['split']!='validation':continue
            d=np.load(root/'observability_20261005/features'/f'seed-{e["seed"]}.npz');x=np.concatenate([(d['features']-s['feature_mean'])/s['feature_std'],(d['states']-s['state_mean'])/s['state_std']],1)
            with torch.no_grad():
                pred=model(torch.from_numpy(x).cuda());raw=pred[:,:7]*torch.as_tensor(s['action_std'],device='cuda')+torch.as_tensor(s['action_mean'],device='cuda');raw=route(raw,pred[:,7:]).cpu().numpy()
            target=d['actions'];ticks=d['ticks'];m=metrics(raw,target,ticks);idx=np.flatnonzero((np.abs(target[:,5])>.02)&(np.abs(raw[:,5]-target[:,5])>.05))
            m.update(seed=e['seed'],first_rotation_error_over_005_elapsed_s=float((ticks[idx[0]]-ticks[0])/10) if len(idx) else None);entries.append(m)
            np.savez(out/f'validation-seed-{e["seed"]}.npz',raw=raw,expert=target,ticks=ticks)
        n=sum(e['samples'] for e in entries);nr=sum(e['rotation_samples'] for e in entries);nh=sum(e['steady_samples'] for e in entries)
        agg={'samples':n,'mae_per_action':(sum(np.array(e['applied_mae_per_action'])*e['samples'] for e in entries)/n).tolist(),
            'rotation_mae':sum(e['rotation_mae_rad_s']*e['rotation_samples'] for e in entries)/nr,
            'steady_false_rotation':sum((e['steady_false_rotation_fraction'] or 0)*e['steady_samples'] for e in entries)/nh,
            'clamp_fraction':sum(e['arm_clamp_samples'] for e in entries)/n}
        b=json.loads((root/'wrist_candidate_replay_20261005/summary.json').read_text())['arms']['baseline']['aggregate'];e=next(e for e in entries if e['seed']==8);ratio=e['predicted_approach_command_deg']/e['expert_approach_command_deg']
        gates={'rotation_improved':agg['rotation_mae']<=.8*b['rotation_mae'],'steady_hold':agg['steady_false_rotation']<=.01,
            'other_arm_channels':all(agg['mae_per_action'][i]<=1.15*b['mae_per_action'][i]+.001 for i in range(5)),
            'jaw':agg['mae_per_action'][6]<=1.15*b['mae_per_action'][6]+.00005,'clamps':agg['clamp_fraction']<=.05,'seed8_approach_rotation':.5<=ratio<=1.5}
        report={'status':'completed','metadata':metadata,'aggregate':agg,'episodes':entries,'live_preflight_gates':gates,'eligible_for_live_preflight':all(gates.values()),'offline_only':True}
        atomic_json(out/'summary.json',report);print(json.dumps({'aggregate':agg,'gates':gates,'seed8':e}),flush=True)

if __name__=='__main__':main()
