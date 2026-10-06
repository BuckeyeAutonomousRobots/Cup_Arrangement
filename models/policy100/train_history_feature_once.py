"""One finite-history GRU candidate, fixed internal split and calibration protocol."""
import fcntl,json,os,time
from pathlib import Path
import numpy as np
import torch
from train_compare import ROOT,check_workloads,atomic_json,sha
from train_calibrated_feature_once import route,aggregate,metrics

class HistoryHead(torch.nn.Module):
    def __init__(self):
        super().__init__();self.project=torch.nn.Sequential(torch.nn.Linear(3080,64),torch.nn.ReLU(),torch.nn.Dropout(.15));self.gru=torch.nn.GRU(64,64,batch_first=True);self.output=torch.nn.Sequential(torch.nn.Linear(64,32),torch.nn.ReLU(),torch.nn.Dropout(.15),torch.nn.Linear(32,10))
    def forward(self,x):
        seq,_=self.gru(self.project(x));return self.output(seq[:,-1])

def history_indices(ticks,length=10):
    rows=[];start=0
    for i,t in enumerate(ticks):
        if i and t!=ticks[i-1]+1:start=i
        rows.append(np.maximum(np.arange(i-length+1,i+1),start))
    return np.asarray(rows,dtype=np.int64)

def main():
    began=time.monotonic()
    with open('/tmp/baseline_collection_batch.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);check_workloads();torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.4);torch.manual_seed(42)
        np.testing.assert_array_equal(history_indices([0,1,2,5,6],3),[[0,0,0],[0,0,1],[0,1,2],[3,3,3],[3,3,4]])
        root=ROOT/'models/policy100/runs';source=root/'FeatureBC_calibrated_20261005';out=root/'HistoryBC_20261005';assert not out.exists();out.mkdir()
        previous=json.loads((source/'summary.json').read_text());fit=previous['fit_seeds'];cal=previous['calibration_seeds'];outer=previous['outer_seeds'];s=dict(np.load(source/'normalization.npz'));np.savez(out/'normalization.npz',**s)
        report={'status':'training','pid':os.getpid(),'fit_seeds':fit,'calibration_seeds':cal,'outer_seeds':outer,'dataset_sha256':previous['dataset_sha256'],
            'backbone_path':previous['backbone_path'],'backbone_sha256':previous['backbone_sha256'],'architecture':'3080->64 ReLU Dropout.15 ->GRU64 ->32 ReLU Dropout.15 ->10',
            'history':'Current plus nine preceding 10Hz observations. Reset at episode/gap; left-pad by repeating oldest available frame. Recompute GRU from zero on each rolling window. No future frames or ground-truth phase.',
            'runtime_requirement':'Clear deque on explicit recovery reset or observation_sim_s gap>0.15s; a timestamp must be added to the scoped request before live deployment. Warmup must clear history afterward.',
            'training':'3000 updates, uniform endpoint sampling batch256, AdamW lr.001 decay.001; normalized action MSE +.1 phase CE; no oversampling',
            'selection_predeclared':previous['selection_predeclared'],'history_index_unit_test_pass':True}
        atomic_json(out/'summary.json',report)
        with out.with_suffix('.claim').open('x') as f:json.dump({'pid':os.getpid(),'script_sha256':sha(Path(__file__)),'offline_only':True},f)
        rows={};xs={};hs={};labels={};targets={}
        for seed in fit+cal+outer:
            d=dict(np.load(source/'features'/f'seed-{seed}.npz'));rows[seed]=d
            xx=np.concatenate([(d['features']-s['feature_mean'])/s['feature_std'],(d['states']-s['state_mean'])/s['state_std']],1)
            xs[seed]=torch.from_numpy(xx).cuda();hs[seed]=torch.from_numpy(history_indices(d['ticks'])).cuda()
            labels[seed]=np.where(d['actions'][:,5]<-.02,1,np.where(d['actions'][:,5]>.02,2,0));targets[seed]=(d['actions']-s['action_mean'])/s['action_std']
        x=torch.cat([xs[k] for k in fit]);offset=0;his=[]
        for k in fit:his.append(hs[k]+offset);offset+=len(xs[k])
        hist=torch.cat(his);y=torch.from_numpy(np.concatenate([targets[k] for k in fit])).cuda();cls=torch.from_numpy(np.concatenate([labels[k] for k in fit])).cuda()
        model=HistoryHead().cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
        for step in range(1,3001):
            assert time.monotonic()-began<600
            if step%250==1:check_workloads()
            idx=torch.randint(len(x),(256,),device='cuda');p=model(x[hist[idx]]);loss=torch.nn.functional.mse_loss(p[:,:7],y[idx])+.1*torch.nn.functional.cross_entropy(p[:,7:],cls[idx]);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
            if step%500==0:report.update(step=step,loss=float(loss));atomic_json(out/'summary.json',report);print(json.dumps({'step':step,'loss':float(loss)}),flush=True)
        model.eval();torch.save(model.state_dict(),out/'model.pt');loaded=HistoryHead().cuda();loaded.load_state_dict(torch.load(out/'model.pt',weights_only=True));loaded.eval()
        with torch.no_grad():torch.testing.assert_close(model(x[hist[:8]]),loaded(x[hist[:8]]));del loaded
        predictions={}
        def predict(seeds):
            with torch.no_grad():
                for seed in seeds:
                    v=xs[seed];h=hs[seed];p=torch.cat([model(v[h[i:i+128]]).cpu() for i in range(0,len(v),128)])
                    # Streaming prefix equality with reset-at-gap / repeated oldest padding.
                    buffer=[];expected=[];ticks=rows[seed]['ticks']
                    for i in range(min(25,len(v))):
                        if i and ticks[i]!=ticks[i-1]+1:buffer=[]
                        buffer=(buffer+[i])[-10:];indices=[buffer[0]]*(10-len(buffer))+buffer
                        expected.append(model(v[indices][None])[0].cpu())
                    torch.testing.assert_close(p[:len(expected)],torch.stack(expected),rtol=2e-4,atol=5e-5)
                    predictions[seed]=p
        predict(cal);logits=torch.cat([predictions[k][:,7:] for k in cal]);lab=torch.from_numpy(np.concatenate([labels[k] for k in cal]));lt=torch.nn.Parameter(torch.zeros(()));optimizer=torch.optim.LBFGS([lt],lr=.1,max_iter=50)
        def closure():
            optimizer.zero_grad();loss=torch.nn.functional.cross_entropy(logits/lt.exp().clamp(.1,10),lab);loss.backward();return loss
        optimizer.step(closure);temperature=float(lt.exp().clamp(.1,10).detach())
        def evaluate(seeds,threshold,save=False):
            entries=[]
            for seed in seeds:
                p=predictions[seed];raw=route(p[:,:7]*torch.from_numpy(s['action_std'])+torch.from_numpy(s['action_mean']),p[:,7:],temperature,threshold).numpy();d=rows[seed];m=metrics(raw,d['actions'],d['ticks']);m['seed']=seed
                actual_rot=np.flatnonzero(np.abs(d['actions'][:,5])>.02);pred_rot=np.flatnonzero(np.abs(raw[:,5])>.02)
                m['rotation_onset_delta_s']=float((d['ticks'][pred_rot[0]]-d['ticks'][actual_rot[0]])/10) if len(pred_rot) and len(actual_rot) else None
                bad=np.flatnonzero((np.abs(d['actions'][:,5])>.02)&(np.abs(raw[:,5]-d['actions'][:,5])>.05));m['first_wrist_error_over005_s']=float((d['ticks'][bad[0]]-d['ticks'][0])/10) if len(bad) else None;entries.append(m)
                if save:np.savez(out/f'validation-seed-{seed}.npz',raw=raw,expert=d['actions'],ticks=d['ticks'])
            return aggregate(entries),entries
        choices=[]
        for threshold in (.5,.6,.7,.8,.9,.95,.975,.99,.995):
            m,_=evaluate(cal,threshold);obj=float(np.mean(np.array(m['mae_per_action'])/s['action_std'])+m['rotation_mae']/s['action_std'][5]+.1*(1-m['rotation_recall']));feasible=m['steady_false_rotation']<=.005 and m['rotation_recall']>=.65 and m['rotation_mae']<=.12
            choices.append({'threshold':threshold,'metrics':m,'objective':obj,'penalized':obj+100*max(0,m['steady_false_rotation']-.005)+5*max(0,.65-m['rotation_recall'])+5*max(0,m['rotation_mae']-.12),'feasible':feasible})
        valid=[c for c in choices if c['feasible']];chosen=min(valid,key=lambda c:c['objective']) if valid else min(choices,key=lambda c:c['penalized'])
        calibration={'temperature':temperature,'chosen':chosen,'choices':choices,'head_sha256':sha(out/'model.pt'),'normalization_sha256':sha(out/'normalization.npz')};atomic_json(out/'calibration.json',calibration)
        predict(outer);agg,entries=evaluate(outer,chosen['threshold'],True)
        b=json.loads((root/'wrist_candidate_replay_20261005/summary.json').read_text())['arms']['baseline']['aggregate'];s8=next(e for e in entries if e['seed']==8);ratio=s8['predicted_approach_command_deg']/s8['expert_approach_command_deg']
        gates={'internal_calibration':bool(valid),'rotation_improved':agg['rotation_mae']<=.8*b['rotation_mae'],'steady_hold':agg['steady_false_rotation']<=.01,
            'other_arm_channels':all(agg['mae_per_action'][i]<=1.15*b['mae_per_action'][i]+.001 for i in range(5)),
            'jaw':agg['mae_per_action'][6]<=1.15*b['mae_per_action'][6]+.00005,'clamps':agg['clamp_fraction']<=.05,'seed8_approach_rotation':.5<=ratio<=1.5}
        report.update(status='completed',checkpoint_reload_pass=True,streaming_prefix_equivalence_pass=True,model_sha256=sha(out/'model.pt'),calibration=calibration,aggregate=agg,episodes=entries,live_preflight_gates=gates,eligible_for_live_preflight=all(gates.values()),wall_seconds=time.monotonic()-began)
        atomic_json(out/'summary.json',report);print(json.dumps({'calibration':chosen,'temperature':temperature,'outer':agg,'gates':gates,'seed8':s8}),flush=True)

if __name__=='__main__':main()
