"""Finish the saved candidate after JSON serialization failure; never retrain."""
import fcntl,json,os
import numpy as np
import torch
from train_compare import ROOT,check_workloads,atomic_json,sha
from train_calibrated_feature_once import Head,route,aggregate,metrics

def main(run_name='FeatureBC_calibrated_20261005',history=False):
    with open('/tmp/baseline_collection_batch.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);check_workloads();torch.set_num_threads(2)
        root=ROOT/'models/policy100/runs';out=root/run_name;report=json.loads((out/'summary.json').read_text())
        assert report['step']==3000 and not (out/'calibration.json').exists() and not list(out.glob('validation-seed-*.npz'))
        note='Checkpoint training/reload finished; calibration JSON serialization failed before outer evaluation. Resume calibration only, same fixed criterion.'
        if history:note='Checkpoint training/reload finished; batch-vs-single GRU logits differed by up to .0051. Calibrate/evaluate exact single-window inference; same weights, fixed split and selection criteria; no outer evaluation previously performed.'
        digest=sha(out/'model.pt');report.update(status='calibrating',training_pid=report['pid'],pid=os.getpid(),recovery_note=note)
        atomic_json(out/'summary.json',report)
        model_class=Head
        if history:
            from train_history_feature_once import HistoryHead,history_indices
            model_class=HistoryHead
        stats=dict(np.load(out/'normalization.npz'));head=model_class().cuda();head.load_state_dict(torch.load(out/'model.pt',weights_only=True));head.eval();rows={};predictions={}
        feature_root=root/'FeatureBC_calibrated_20261005' if history else out
        def predict(seeds):
            for seed in seeds:
                d=dict(np.load(feature_root/'features'/f'seed-{seed}.npz'));rows[seed]=d;x=np.concatenate([(d['features']-stats['feature_mean'])/stats['feature_std'],(d['states']-stats['state_mean'])/stats['state_std']],1)
                with torch.no_grad():
                    v=torch.from_numpy(x).cuda()
                    if history:
                        indices=history_indices(d['ticks']);p=torch.stack([head(v[index][None])[0].cpu() for index in indices]);buffer=[]
                        for i in range(min(25,len(v))):
                            if i and d['ticks'][i]!=d['ticks'][i-1]+1:buffer=[]
                            buffer=(buffer+[i])[-10:];idx=[buffer[0]]*(10-len(buffer))+buffer
                            torch.testing.assert_close(p[i],head(v[idx][None])[0].cpu(),rtol=1e-5,atol=1e-6)
                        predictions[seed]=p
                    else:predictions[seed]=torch.cat([head(v[i:i+512]).cpu() for i in range(0,len(v),512)])
        cal=report['calibration_seeds'];predict(cal);logits=torch.cat([predictions[s][:,7:] for s in cal]);labels=torch.from_numpy(np.concatenate([np.where(rows[s]['actions'][:,5]<-.02,1,np.where(rows[s]['actions'][:,5]>.02,2,0)) for s in cal]))
        logtemp=torch.nn.Parameter(torch.zeros(()));opt=torch.optim.LBFGS([logtemp],lr=.1,max_iter=50)
        def closure():
            opt.zero_grad();loss=torch.nn.functional.cross_entropy(logits/logtemp.exp().clamp(.1,10),labels);loss.backward();return loss
        opt.step(closure);temperature=float(logtemp.exp().clamp(.1,10).detach())
        def evaluate(seeds,threshold,save=False):
            entries=[]
            for seed in seeds:
                p=predictions[seed];raw=p[:,:7]*torch.from_numpy(stats['action_std'])+torch.from_numpy(stats['action_mean']);raw=route(raw,p[:,7:],temperature,threshold).numpy();d=rows[seed];m=metrics(raw,d['actions'],d['ticks']);m['seed']=seed
                er=np.flatnonzero(np.abs(d['actions'][:,5])>.02);pr=np.flatnonzero(np.abs(raw[:,5])>.02);bad=np.flatnonzero((np.abs(d['actions'][:,5])>.02)&(np.abs(raw[:,5]-d['actions'][:,5])>.05))
                m['rotation_onset_delta_s']=float((d['ticks'][pr[0]]-d['ticks'][er[0]])/10) if len(pr) and len(er) else None
                m['first_wrist_error_over005_s']=float((d['ticks'][bad[0]]-d['ticks'][0])/10) if len(bad) else None
                entries.append(m)
                if save:np.savez(out/f'validation-seed-{seed}.npz',raw=raw,expert=d['actions'],ticks=d['ticks'])
            return aggregate(entries),entries
        choices=[]
        for threshold in (.5,.6,.7,.8,.9,.95,.975,.99,.995):
            m,_=evaluate(cal,threshold);objective=float(np.mean(np.array(m['mae_per_action'])/stats['action_std'])+m['rotation_mae']/stats['action_std'][5]+.1*(1-m['rotation_recall']))
            feasible=m['steady_false_rotation']<=.005 and m['rotation_recall']>=.65 and m['rotation_mae']<=.12
            penalty=objective+100*max(0,m['steady_false_rotation']-.005)+5*max(0,.65-m['rotation_recall'])+5*max(0,m['rotation_mae']-.12)
            choices.append({'threshold':threshold,'metrics':m,'objective':objective,'penalized':penalty,'feasible':feasible})
        feasible=[c for c in choices if c['feasible']];chosen=min(feasible,key=lambda c:c['objective']) if feasible else min(choices,key=lambda c:c['penalized'])
        calibration={'temperature':temperature,'chosen':chosen,'choices':choices,'calibration_seeds':cal,'head_sha256':digest,'normalization_sha256':sha(out/'normalization.npz')};atomic_json(out/'calibration.json',calibration)
        predict(report['outer_seeds']);agg,entries=evaluate(report['outer_seeds'],chosen['threshold'],True)
        b=json.loads((root/'wrist_candidate_replay_20261005/summary.json').read_text())['arms']['baseline']['aggregate'];s8=next(e for e in entries if e['seed']==8);ratio=s8['predicted_approach_command_deg']/s8['expert_approach_command_deg']
        gates={'internal_calibration':bool(feasible),'rotation_improved':agg['rotation_mae']<=.8*b['rotation_mae'],'steady_hold':agg['steady_false_rotation']<=.01,
            'other_arm_channels':all(agg['mae_per_action'][i]<=1.15*b['mae_per_action'][i]+.001 for i in range(5)),
            'jaw':agg['mae_per_action'][6]<=1.15*b['mae_per_action'][6]+.00005,'clamps':agg['clamp_fraction']<=.05,'seed8_approach_rotation':.5<=ratio<=1.5}
        assert sha(out/'model.pt')==digest
        report.update(status='completed',checkpoint_reload_pass=True,model_sha256=digest,calibration=calibration,aggregate=agg,episodes=entries,live_preflight_gates=gates,eligible_for_live_preflight=all(gates.values()))
        if history:report.update(inference_evaluation_mode='exact single-window GRU calls for all observations',streaming_prefix_equivalence_pass=True)
        atomic_json(out/'summary.json',report);print(json.dumps({'calibration':chosen,'temperature':temperature,'outer':agg,'gates':gates,'seed8':s8}),flush=True)

if __name__=='__main__':main()
