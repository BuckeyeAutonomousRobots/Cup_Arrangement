"""Fixed full validation-sequence comparison; no simulator access."""
import fcntl,gc,json,sys,time
from pathlib import Path
import numpy as np
import torch
from train_compare import ROOT,check_workloads,sha,atomic_json
from lerobot.policies.act.modeling_act import ACTPolicy,ACTTemporalEnsembler
sys.path.insert(0,str(ROOT/'models/act_v1'))
from replay_temporal_comparison import metrics

def main():
    with open('/tmp/baseline_collection_batch.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);check_workloads()
        torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.4)
        root=ROOT/'models/policy100/runs';source=root/'ACT_Diffusion_100_20261004';candidate=root/'ACT_wrist_quadratic_20261005/ACT_100'
        assert json.loads((candidate/'status.json').read_text())['status']=='completed'
        out=root/'wrist_candidate_replay_20261005';out.mkdir(exist_ok=True)
        manifest=json.loads((source/'dataset_manifest.json').read_text());stats=dict(np.load(source/'act_normalization.npz'))
        mean=np.array([.485,.456,.406],np.float32)[None,:,None,None];std=np.array([.229,.224,.225],np.float32)[None,:,None,None]
        report={'status':'running','dataset_sha256':sha(source/'dataset_manifest.json'),'arms':{},
                'offline_only':True,'checkpoint_selection':'final checkpoints, 5000 baseline versus 7000 candidate',
                'gate_rule':'rotation MAE <=80% baseline; steady false rotation <=1%; other arm MAE <=115% baseline +0.001 rad/s; jaw <=115% baseline +0.00005m; raw arm clamp <=5%; seed8 approach wrist integral correct sign and 50%-150% expert'}
        if (out/'summary.json').exists():
            report=json.loads((out/'summary.json').read_text());assert report['status']=='running'
            report['numerical_replay_note']='Offline batching produced normalized prefix differences up to 1.12e-4. Episodes exceeding atol5e-5/rtol2e-4 or physical tolerance 1e-5rad/s,1e-7m use exact deployed sequential select_action for every observation. No runtime guard changed.'
        start=time.monotonic()
        for name,run in [('baseline',source/'ACT_100'),('candidate',candidate)]:
            check_workloads();policy=ACTPolicy.from_pretrained(run/'last_policy').cuda().eval()
            other=dict(np.load(run/'normalization.npz'));assert all(np.array_equal(stats[k],other[k]) for k in stats)
            digest=sha(run/'last_policy/model.safetensors')
            if name in report['arms']:
                assert report['arms'][name]['checkpoint_sha256']==digest
                entries=report['arms'][name]['episodes']
            else:
                entries=[];report['arms'][name]={'checkpoint_sha256':digest,'episodes':entries}
            for e in [e for e in manifest['episodes'] if e['split']=='validation']:
                if e['seed'] in [x['seed'] for x in entries]:continue
                assert time.monotonic()-start<600
                check_workloads();cache=Path(e['cache']);data={k:np.load(cache/(k+'.npy'),mmap_mode='r') for k in ('images','states','actions','ticks')}
                n=len(data['ticks']);ticks=np.array(data['ticks'])
                def batch(lo,hi):
                    rgb=np.asarray(data['images'][lo:hi],np.float32).transpose(0,3,1,2)/255.
                    q=(data['states'][lo:hi]-stats['state_mean'])/stats['state_std']
                    return {'observation.images.wrist':torch.from_numpy((rgb-mean)/std).cuda(),'observation.state':torch.from_numpy(q.copy()).cuda()}
                with torch.inference_mode():
                    chunks=torch.cat([policy.predict_action_chunk(batch(i,min(i+8,n))).cpu() for i in range(0,n,8)])
                    policy.reset();direct=torch.stack([policy.select_action(batch(i,i+1))[0].cpu() for i in range(min(n,25))])
                    ens=ACTTemporalEnsembler(.01,20);continuous=torch.stack([ens.update(chunks[i:i+1])[0].clone() for i in range(n)])
                    delta=continuous[:len(direct)]-direct
                    fallback=not torch.allclose(continuous[:len(direct)],direct,rtol=2e-4,atol=5e-5) or not np.all(np.max(np.abs(delta.numpy())*stats['action_std'],axis=0)<np.array([1e-5]*6+[1e-7]))
                    if fallback:
                        continuous=torch.cat([direct,torch.stack([policy.select_action(batch(i,i+1))[0].cpu() for i in range(len(direct),n)])])
                raw=continuous.numpy()*stats['action_std']+stats['action_mean'];target=np.array(data['actions'])
                m=metrics(raw,target,ticks);err=np.abs(raw[:,5]-target[:,5]);idx=np.flatnonzero((np.abs(target[:,5])>.02)&(err>.05))
                m.update(seed=e['seed'],exact_sequential_fallback=fallback,prefix_normalized_max_difference=float(delta.abs().max()),first_rotation_error_over_005_elapsed_s=float((ticks[idx[0]]-ticks[0])/10) if len(idx) else None)
                entries.append(m);np.savez(out/f'{name}-seed-{e["seed"]}.npz',raw=raw,expert=target,ticks=ticks)
                atomic_json(out/'summary.json',report);print(json.dumps({'arm':name,'seed':e['seed'],'rotation_mae':m['rotation_mae_rad_s'],'steady_false_rotation':m['steady_false_rotation_fraction']}),flush=True)
            total=sum(x['samples'] for x in entries);nr=sum(x['rotation_samples'] for x in entries);ns=sum(x['steady_samples'] for x in entries)
            report['arms'][name]['aggregate']={'mae_per_action':(sum(np.array(x['applied_mae_per_action'])*x['samples'] for x in entries)/total).tolist(),
                'rotation_mae':sum(x['rotation_mae_rad_s']*x['rotation_samples'] for x in entries)/nr,
                'steady_false_rotation':sum((x['steady_false_rotation_fraction'] or 0)*x['steady_samples'] for x in entries)/ns,
                'clamp_fraction':sum(x['arm_clamp_samples'] for x in entries)/total,'samples':total}
            del policy;gc.collect();torch.cuda.empty_cache()
        b=report['arms']['baseline']['aggregate'];c=report['arms']['candidate']['aggregate'];s=next(e for e in report['arms']['candidate']['episodes'] if e['seed']==8)
        ratio=s['predicted_approach_command_deg']/s['expert_approach_command_deg']
        gates={'rotation_improved':c['rotation_mae']<=.8*b['rotation_mae'],'steady_hold':c['steady_false_rotation']<=.01,
            'other_arm_channels':all(c['mae_per_action'][i]<=1.15*b['mae_per_action'][i]+.001 for i in range(5)),
            'jaw':c['mae_per_action'][6]<=1.15*b['mae_per_action'][6]+.00005,'clamps':c['clamp_fraction']<=.05,'seed8_approach_rotation':.5<=ratio<=1.5}
        report.update(status='completed',wall_seconds=time.monotonic()-start,live_preflight_gates=gates,eligible_for_live_preflight=all(gates.values()))
        atomic_json(out/'summary.json',report);print(json.dumps({'gates':gates,'baseline':b,'candidate':c}),flush=True)

if __name__=='__main__':main()
