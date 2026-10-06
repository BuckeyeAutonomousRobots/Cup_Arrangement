"""One bounded two-stream causal-velocity candidate, internal-only selection."""
import fcntl,json,os,shutil,time
from pathlib import Path
import numpy as np
import torch
from train_compare import ROOT,check_workloads,atomic_json,sha
from train_calibrated_feature_once import route,aggregate,metrics

class TwoStream(torch.nn.Module):
    def __init__(self):
        super().__init__();self.vision=torch.nn.Sequential(torch.nn.Linear(3072,64),torch.nn.ReLU(),torch.nn.Dropout(.15));self.kinematics=torch.nn.Sequential(torch.nn.Linear(17,64),torch.nn.ReLU(),torch.nn.Dropout(.15));self.decoder=torch.nn.Sequential(torch.nn.Linear(128,64),torch.nn.ReLU(),torch.nn.Dropout(.15),torch.nn.Linear(64,10))
    def forward(self,x):return self.decoder(torch.cat([self.vision(x[:,:3072]),self.kinematics(x[:,3072:])],1))

def velocities(q,ticks):
    valid=np.r_[False,np.diff(ticks)==1];v=np.zeros_like(q);v[1:]=np.diff(q,axis=0)/np.maximum(np.diff(ticks)[:,None]/10,1e-6);v[~valid]=0
    return v,valid

def main():
    start=time.monotonic()
    with open('/tmp/baseline_collection_batch.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);check_workloads();torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.4);torch.manual_seed(42)
        root=ROOT/'models/policy100/runs';src=root/'FeatureBC_calibrated_20261005';out=root/'TwoStreamBC_20261005';assert not out.exists();out.mkdir()
        old=json.loads((src/'summary.json').read_text());fit=old['fit_seeds'];cal=old['calibration_seeds'];outer=old['outer_seeds'];stats=dict(np.load(src/'normalization.npz'));rows={}
        for seed in fit+cal+outer:
            d=dict(np.load(src/'features'/f'seed-{seed}.npz'));d['velocity'],d['valid']=velocities(d['states'],d['ticks']);rows[seed]=d
        vv=np.concatenate([rows[s]['velocity'][rows[s]['valid']] for s in fit]);stats.update(velocity_mean=vv.mean(0),velocity_std=np.maximum(vv.std(0),np.array([.01]*6+[.0001]*2,np.float32)));np.savez(out/'normalization.npz',**stats)
        xs={};ys={};labels={}
        for seed,d in rows.items():
            x=np.concatenate([(d['features']-stats['feature_mean'])/stats['feature_std'],(d['states']-stats['state_mean'])/stats['state_std'],(d['velocity']-stats['velocity_mean'])/stats['velocity_std'],d['valid'][:,None]],1).astype(np.float32)
            xs[seed]=torch.from_numpy(x).cuda();ys[seed]=(d['actions']-stats['action_mean'])/stats['action_std'];labels[seed]=np.where(d['actions'][:,5]<-.02,1,np.where(d['actions'][:,5]>.02,2,0))
        report={'status':'training','pid':os.getpid(),'fit_seeds':fit,'calibration_seeds':cal,'outer_seeds':outer,'dataset_sha256':old['dataset_sha256'],
            'backbone_path':old['backbone_path'],'backbone_sha256':old['backbone_sha256'],'architecture':'3072 visual features->64; 8positions+8causal velocities+validity->64; fused128->64->10; dropout.15',
            'alignment':'Original same-tick commands unchanged. Velocity uses preceding contiguous10Hz measured joint positions only; reset to0 and invalid at episode/gap. No future input or expert action input.',
            'training':'Uniform batch256 AdamW lr.001 decay.001; original normalized action MSE+.1phaseCE; 50% physical-zero velocity dropout, validity unchanged, to discourage pure velocity echo.',
            'convergence_predeclared':'Maximum8000 updates. Fixed2048 fit probe with full and zero-velocity inputs every1000. Plateau if last3 probe values vary<2% of their mean after4000. Internal checkpoint/temperature/threshold selection at2000,4000,6000,8000; no outer evaluation until choice frozen.',
            'selection_predeclared':old['selection_predeclared'],'initiation_predeclared':'Report actual stationary rotation and already-moving rotation separately, plus zero-velocity ablation. No live launch if internal stationary rotation recall<50%; tiny counts are not a success-rate claim.',
            'history':[],'candidates':[]}
        atomic_json(out/'summary.json',report)
        with out.with_suffix('.claim').open('x') as f:json.dump({'pid':os.getpid(),'script_sha256':sha(Path(__file__))},f)
        x=torch.cat([xs[s] for s in fit]);y=torch.from_numpy(np.concatenate([ys[s] for s in fit])).cuda();cls=torch.from_numpy(np.concatenate([labels[s] for s in fit])).cuda();zero=torch.from_numpy(-stats['velocity_mean']/stats['velocity_std']).cuda()
        model=TwoStream().cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001);probe=torch.linspace(0,len(x)-1,2048,device='cuda').long()
        def no_velocity(xx):
            z=xx.clone();z[:,3080:3088]=zero;return z
        def lossfn(p,target,phase):return torch.nn.functional.mse_loss(p[:,:7],target)+.1*torch.nn.functional.cross_entropy(p[:,7:],phase)
        def predict(seeds,zero_velocity=False):
            result={};model.eval()
            with torch.no_grad():
                for seed in seeds:
                    v=no_velocity(xs[seed]) if zero_velocity else xs[seed];result[seed]=torch.cat([model(v[i:i+256]).cpu() for i in range(0,len(v),256)])
            return result
        def evaluate(pred,seeds,temp,threshold,save=False):
            entries=[];stationary=[0,0];moving=[0,0]
            for seed in seeds:
                p=pred[seed];raw=route(p[:,:7]*torch.from_numpy(stats['action_std'])+torch.from_numpy(stats['action_mean']),p[:,7:],temp,threshold).numpy();d=rows[seed];target=d['actions'];rot=np.abs(target[:,5])>.02;correct=(raw[:,5]*target[:,5]>0)&(np.abs(raw[:,5])>.02)
                st=rot&d['valid']&(np.abs(d['velocity'][:,5])<=.005);mv=rot&d['valid']&(np.abs(d['velocity'][:,5])>.02)
                stationary[0]+=int(correct[st].sum());stationary[1]+=int(st.sum());moving[0]+=int(correct[mv].sum());moving[1]+=int(mv.sum())
                m=metrics(raw,target,d['ticks']);m['seed']=seed;ids=np.flatnonzero(rot&(np.abs(raw[:,5]-target[:,5])>.05));m['first_wrist_error_over005_s']=float((d['ticks'][ids[0]]-d['ticks'][0])/10) if len(ids) else None
                er=np.flatnonzero(rot);pr=np.flatnonzero(np.abs(raw[:,5])>.02);m['rotation_onset_delta_s']=float((d['ticks'][pr[0]]-d['ticks'][er[0]])/10) if len(er) and len(pr) else None;entries.append(m)
                if save:np.savez(out/f'validation-seed-{seed}.npz',raw=raw,expert=target,ticks=d['ticks'])
            agg=aggregate(entries);agg.update(stationary_rotation_correct=stationary[0],stationary_rotation_samples=stationary[1],stationary_rotation_recall=stationary[0]/max(stationary[1],1),moving_rotation_recall=moving[0]/max(moving[1],1));return agg,entries
        def calibrate(step):
            pred=predict(cal);logits=torch.cat([pred[s][:,7:] for s in cal]);lab=torch.from_numpy(np.concatenate([labels[s] for s in cal]));lt=torch.nn.Parameter(torch.zeros(()));opt=torch.optim.LBFGS([lt],lr=.1,max_iter=50)
            def closure():
                opt.zero_grad();loss=torch.nn.functional.cross_entropy(logits/lt.exp().clamp(.1,10),lab);loss.backward();return loss
            opt.step(closure);temp=float(lt.exp().clamp(.1,10).detach());choices=[]
            for threshold in (.5,.6,.7,.8,.9,.95,.975,.99,.995):
                m,_=evaluate(pred,cal,temp,threshold);obj=float(np.mean(np.array(m['mae_per_action'])/stats['action_std'])+m['rotation_mae']/stats['action_std'][5]+.1*(1-m['rotation_recall']));feasible=m['steady_false_rotation']<=.005 and m['rotation_recall']>=.65 and m['rotation_mae']<=.12
                choices.append({'threshold':threshold,'metrics':m,'objective':obj,'penalized':obj+100*max(0,m['steady_false_rotation']-.005)+5*max(0,.65-m['rotation_recall'])+5*max(0,m['rotation_mae']-.12),'feasible':feasible})
            valid=[c for c in choices if c['feasible']];chosen=min(valid,key=lambda c:c['objective']) if valid else min(choices,key=lambda c:c['penalized'])
            torch.save(model.state_dict(),out/f'checkpoint-{step}.pt');result={'step':step,'temperature':temp,'chosen':chosen,'choices':choices};report['candidates'].append(result);atomic_json(out/'summary.json',report);print(json.dumps({'calibration_step':step,'chosen':chosen}),flush=True)
        for step in range(1,8001):
            assert time.monotonic()-start<600
            if step%250==1:check_workloads()
            model.train();idx=torch.randint(len(x),(256,),device='cuda');batch=x[idx].clone();drop=torch.rand(256,device='cuda')<.5;batch[drop,3080:3088]=zero
            loss=lossfn(model(batch),y[idx],cls[idx]);optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
            if step%1000==0:
                model.eval()
                with torch.no_grad():full=float(lossfn(model(x[probe]),y[probe],cls[probe]));ablated=float(lossfn(model(no_velocity(x[probe])),y[probe],cls[probe]))
                report['history'].append({'step':step,'fit_loss_full':full,'fit_loss_zero_velocity':ablated,'fit_loss_mean':(full+ablated)/2});report['step']=step;atomic_json(out/'summary.json',report)
                print(json.dumps(report['history'][-1]),flush=True)
            if step%2000==0:
                calibrate(step)
                recent=[r['fit_loss_mean'] for r in report['history'][-3:]]
                if step>=4000 and len(recent)==3 and (max(recent)-min(recent))/max(np.mean(recent),1e-9)<.02:report['convergence']='fit probe plateau';break
        report.setdefault('convergence','budget reached; no verified plateau, do not claim convergence')
        valid=[c for c in report['candidates'] if c['chosen']['feasible']];chosen=min(valid,key=lambda c:c['chosen']['objective']) if valid else min(report['candidates'],key=lambda c:c['chosen']['penalized'])
        shutil.copy2(out/f'checkpoint-{chosen["step"]}.pt',out/'model.pt');model.load_state_dict(torch.load(out/'model.pt',weights_only=True));model.eval();digest=sha(out/'model.pt');atomic_json(out/'calibration.json',chosen)
        selected_temp=chosen['temperature'];threshold=chosen['chosen']['threshold'];outerpred=predict(outer);agg,entries=evaluate(outerpred,outer,selected_temp,threshold,True);ablated,_=evaluate(predict(outer,True),outer,selected_temp,threshold)
        b=json.loads((root/'wrist_candidate_replay_20261005/summary.json').read_text())['arms']['baseline']['aggregate'];s8=next(e for e in entries if e['seed']==8);ratio=s8['predicted_approach_command_deg']/s8['expert_approach_command_deg']
        gates={'internal_calibration':chosen['chosen']['feasible'],'stationary_initiation':chosen['chosen']['metrics']['stationary_rotation_recall']>=.5,
            'rotation_improved':agg['rotation_mae']<=.8*b['rotation_mae'],'steady_hold':agg['steady_false_rotation']<=.01,'other_arm_channels':all(agg['mae_per_action'][i]<=1.15*b['mae_per_action'][i]+.001 for i in range(5)),
            'jaw':agg['mae_per_action'][6]<=1.15*b['mae_per_action'][6]+.00005,'clamps':agg['clamp_fraction']<=.05,'seed8_approach_rotation':.5<=ratio<=1.5}
        reload=TwoStream().cuda();reload.load_state_dict(torch.load(out/'model.pt',weights_only=True));reload.eval()
        with torch.no_grad():torch.testing.assert_close(model(x[:8]),reload(x[:8]));del reload
        report.update(status='completed',selected_step=chosen['step'],model_sha256=digest,checkpoint_reload_pass=True,aggregate=agg,velocity_zero_ablation=ablated,episodes=entries,live_preflight_gates=gates,eligible_for_live_preflight=all(gates.values()),wall_seconds=time.monotonic()-start)
        atomic_json(out/'summary.json',report);print(json.dumps({'outer':agg,'velocity_zero_ablation':ablated,'gates':gates,'seed8':s8}),flush=True)

if __name__=='__main__':main()
