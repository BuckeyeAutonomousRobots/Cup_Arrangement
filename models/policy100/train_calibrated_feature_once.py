"""One internally calibrated regularized policy; outer validation evaluated once."""
import fcntl,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
from torchvision.models import resnet18
from train_compare import ROOT,check_workloads,atomic_json,sha
sys.path.insert(0,str(ROOT/'models/act_v1'))
from replay_temporal_comparison import metrics

class Head(torch.nn.Module):
    def __init__(self):
        super().__init__();self.net=torch.nn.Sequential(torch.nn.Linear(3080,64),torch.nn.ReLU(),torch.nn.Dropout(.15),torch.nn.Linear(64,32),torch.nn.ReLU(),torch.nn.Dropout(.15),torch.nn.Linear(32,10))
    def forward(self,x):return self.net(x)

def route(raw,logits,temperature,threshold):
    probs=torch.softmax(logits/temperature,-1);confidence,kind=probs.max(-1)
    sign=torch.where(kind==1,-1.,torch.where(kind==2,1.,0.))
    result=raw.clone();result[:,5]=torch.where((kind!=0)&(confidence>=threshold)&(raw[:,5]*sign>0),raw[:,5],0.)
    return result

def aggregate(entries):
    n=sum(e['samples'] for e in entries);nr=sum(e['rotation_samples'] for e in entries);nh=sum(e['steady_samples'] for e in entries)
    return {'samples':n,'mae_per_action':(sum(np.array(e['applied_mae_per_action'])*e['samples'] for e in entries)/n).tolist(),
        'rotation_mae':sum(e['rotation_mae_rad_s']*e['rotation_samples'] for e in entries)/nr,
        'rotation_recall':sum(e['correct_rotation_fraction']*e['rotation_samples'] for e in entries)/nr,
        'steady_false_rotation':sum((e['steady_false_rotation_fraction'] or 0)*e['steady_samples'] for e in entries)/nh,
        'clamp_fraction':sum(e['arm_clamp_samples'] for e in entries)/n}

def main():
    start=time.monotonic()
    with open('/tmp/baseline_collection_batch.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);check_workloads();torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.4);torch.manual_seed(42)
        root=ROOT/'models/policy100/runs';source=root/'ACT_Diffusion_100_20261004';out=root/'FeatureBC_calibrated_20261005'
        assert not out.exists();out.mkdir();(out/'features').mkdir()
        manifest=json.loads((source/'dataset_manifest.json').read_text());cal=sorted(np.random.default_rng(20261005).permutation(manifest['train_seeds'])[:16].tolist());fit=[s for s in manifest['train_seeds'] if s not in cal]
        weights=(Path.home() / '.cache/torch/hub/checkpoints/resnet18-f37072fd.pth')
        report={'status':'extracting','pid':os.getpid(),'dataset_sha256':sha(source/'dataset_manifest.json'),'fit_seeds':fit,'calibration_seeds':cal,'outer_seeds':manifest['validation_seeds'],
            'backbone':'frozen cached ImageNet ResNet18, never trained on this dataset','backbone_path':str(weights),'backbone_sha256':sha(weights),
            'architecture':'spatial2x3=3072 CNN features +8 joints ->64 ReLU Dropout.15 ->32 ReLU Dropout.15 ->10',
            'training':'3000 updates, uniform batch256, AdamW lr.001 decay.001, normalized action MSE +.1 unweighted phase CE',
            'selection_predeclared':'Fit final checkpoint only. Calibrate temperature on internal16 episodes. Threshold grid .5,.6,.7,.8,.9,.95,.975,.99,.995. Feasible: internal steady false<=.005, rotation recall>=.65, rotation MAE<=.12. Minimize mean seven normalized physical MAEs + rotationMAE/wrist_std +.1*(1-recall). If none feasible, minimize objective +100*hold_excess +5*recall_shortfall +5*rotation_error_excess; mark internal failure.',
            'outer_evaluation':'Exactly once after calibration is frozen; same original offline gates; no retraining on calibration or outer episodes.'}
        atomic_json(out/'summary.json',report)
        with out.with_suffix('.claim').open('x') as f:json.dump({'pid':os.getpid(),'script_sha256':sha(Path(__file__)),'fit_seeds':fit,'calibration_seeds':cal},f)
        cnn=resnet18(weights=None);cnn.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True));cnn=torch.nn.Sequential(*list(cnn.children())[:-2]).cuda().eval();cnn.requires_grad_(False)
        mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None];std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None];rows={}
        for e in manifest['episodes']:
            assert time.monotonic()-start<600;check_workloads();cache=Path(e['cache']);images=np.load(cache/'images.npy',mmap_mode='r');parts=[]
            with torch.inference_mode():
                for i in range(0,len(images),32):
                    rgb=torch.from_numpy(np.asarray(images[i:i+32]).copy()).cuda().permute(0,3,1,2).float()/255
                    parts.append(torch.nn.functional.adaptive_avg_pool2d(cnn((rgb-mean)/std),(2,3)).flatten(1).cpu().numpy())
            rows[e['seed']]={'features':np.concatenate(parts),**{k:np.load(cache/(k+'.npy')) for k in ('states','actions','ticks')}}
            np.savez(out/'features'/f'seed-{e["seed"]}.npz',**rows[e['seed']])
            if e['seed']%20==19:print(json.dumps({'features_through_seed':e['seed']}),flush=True)
        del cnn;torch.cuda.empty_cache()
        f=np.concatenate([rows[s]['features'] for s in fit]);q=np.concatenate([rows[s]['states'] for s in fit]);a=np.concatenate([rows[s]['actions'] for s in fit])
        stats={'feature_mean':f.mean(0),'feature_std':np.maximum(f.std(0),.001),'state_mean':q.mean(0),'state_std':np.maximum(q.std(0),np.array([.001]*6+[.0001]*2,np.float32)),
            'action_mean':a.mean(0),'action_std':np.maximum(a.std(0),np.array([.001]*6+[.0001],np.float32))};np.savez(out/'normalization.npz',**stats)
        def inputs(seed):
            r=rows[seed];return np.concatenate([(r['features']-stats['feature_mean'])/stats['feature_std'],(r['states']-stats['state_mean'])/stats['state_std']],1)
        x=torch.from_numpy(np.concatenate([inputs(s) for s in fit])).cuda();y=torch.from_numpy((a-stats['action_mean'])/stats['action_std']).cuda();cls=torch.from_numpy(np.where(a[:,5]<-.02,1,np.where(a[:,5]>.02,2,0))).cuda()
        head=Head().cuda();opt=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.001);report['status']='training'
        for step in range(1,3001):
            assert time.monotonic()-start<700
            if step%250==1:check_workloads()
            idx=torch.randint(len(x),(256,),device='cuda');p=head(x[idx]);loss=torch.nn.functional.mse_loss(p[:,:7],y[idx])+.1*torch.nn.functional.cross_entropy(p[:,7:],cls[idx]);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1);opt.step()
            if step%500==0:report.update(step=step,loss=float(loss));atomic_json(out/'summary.json',report);print(json.dumps({'step':step,'loss':float(loss)}),flush=True)
        head.eval();torch.save(head.state_dict(),out/'model.pt');loaded=Head().cuda();loaded.load_state_dict(torch.load(out/'model.pt',weights_only=True));loaded.eval()
        with torch.no_grad():torch.testing.assert_close(head(x[:8]),loaded(x[:8]));del loaded
        predictions={}
        def predict(seeds):
            with torch.no_grad():
                for seed in seeds:
                    v=torch.from_numpy(inputs(seed)).cuda();predictions[seed]=torch.cat([head(v[i:i+512]).cpu() for i in range(0,len(v),512)])
        predict(cal)
        logits=torch.cat([predictions[s][:,7:] for s in cal]);labels=torch.from_numpy(np.concatenate([np.where(rows[s]['actions'][:,5]<-.02,1,np.where(rows[s]['actions'][:,5]>.02,2,0)) for s in cal]))
        logtemp=torch.nn.Parameter(torch.zeros(()));calopt=torch.optim.LBFGS([logtemp],lr=.1,max_iter=50)
        def closure():
            calopt.zero_grad();nll=torch.nn.functional.cross_entropy(logits/logtemp.exp().clamp(.1,10),labels);nll.backward();return nll
        calopt.step(closure);temperature=float(logtemp.exp().clamp(.1,10).detach())
        def evaluate(seeds,threshold,save=False):
            entries=[]
            for seed in seeds:
                p=predictions[seed];raw=p[:,:7]*torch.from_numpy(stats['action_std'])+torch.from_numpy(stats['action_mean']);raw=route(raw,p[:,7:],temperature,threshold).numpy();r=rows[seed];m=metrics(raw,r['actions'],r['ticks']);m['seed']=seed;entries.append(m)
                if save:np.savez(out/f'validation-seed-{seed}.npz',raw=raw,expert=r['actions'],ticks=r['ticks'])
            return aggregate(entries),entries
        choices=[]
        for threshold in (.5,.6,.7,.8,.9,.95,.975,.99,.995):
            m,_=evaluate(cal,threshold);objective=float(np.mean(np.array(m['mae_per_action'])/stats['action_std'])+m['rotation_mae']/stats['action_std'][5]+.1*(1-m['rotation_recall']))
            feasible=m['steady_false_rotation']<=.005 and m['rotation_recall']>=.65 and m['rotation_mae']<=.12
            penalized=objective+100*max(0,m['steady_false_rotation']-.005)+5*max(0,.65-m['rotation_recall'])+5*max(0,m['rotation_mae']-.12)
            choices.append({'threshold':threshold,'metrics':m,'objective':objective,'penalized':penalized,'feasible':feasible})
        feasible=[c for c in choices if c['feasible']];chosen=min(feasible,key=lambda c:c['objective']) if feasible else min(choices,key=lambda c:c['penalized'])
        calibration={'temperature':temperature,'chosen':chosen,'choices':choices,'calibration_seeds':cal,'head_sha256':sha(out/'model.pt'),'normalization_sha256':sha(out/'normalization.npz')};atomic_json(out/'calibration.json',calibration)
        # Outer predictions and labels are evaluated only after immutable calibration choice is saved.
        predict(manifest['validation_seeds']);agg,entries=evaluate(manifest['validation_seeds'],chosen['threshold'],True)
        b=json.loads((root/'wrist_candidate_replay_20261005/summary.json').read_text())['arms']['baseline']['aggregate'];s8=next(e for e in entries if e['seed']==8);ratio=s8['predicted_approach_command_deg']/s8['expert_approach_command_deg']
        gates={'internal_calibration':bool(feasible),'rotation_improved':agg['rotation_mae']<=.8*b['rotation_mae'],'steady_hold':agg['steady_false_rotation']<=.01,
            'other_arm_channels':all(agg['mae_per_action'][i]<=1.15*b['mae_per_action'][i]+.001 for i in range(5)),
            'jaw':agg['mae_per_action'][6]<=1.15*b['mae_per_action'][6]+.00005,'clamps':agg['clamp_fraction']<=.05,'seed8_approach_rotation':.5<=ratio<=1.5}
        report.update(status='completed',checkpoint_reload_pass=True,model_sha256=sha(out/'model.pt'),calibration=calibration,aggregate=agg,episodes=entries,live_preflight_gates=gates,eligible_for_live_preflight=all(gates.values()),wall_seconds=time.monotonic()-start)
        atomic_json(out/'summary.json',report);print(json.dumps({'calibration':chosen,'temperature':temperature,'outer':agg,'gates':gates,'seed8':s8}),flush=True)

if __name__=='__main__':main()
