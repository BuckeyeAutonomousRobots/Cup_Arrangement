"""Offline train/held-out holding and transition audit of two preserved heads."""
import fcntl,json
import numpy as np
import torch
from train_compare import ROOT,check_workloads,atomic_json
from probe_observability import Probe

def main():
    with open('/tmp/baseline_collection_batch.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);check_workloads();torch.set_num_threads(2)
        root=ROOT/'models/policy100/runs';features=root/'observability_20261005';manifest=json.loads((root/'ACT_Diffusion_100_20261004/dataset_manifest.json').read_text())
        report={}
        for name in ('FeatureBC_20261005','FeatureBC_hold_20261005'):
            s=dict(np.load(root/name/'normalization.npz'));model=Probe(3080,10).cuda();model.load_state_dict(torch.load(root/name/'model.pt',weights_only=True));model.eval()
            totals={split:{'steady':0,'false':0,'rotation':0,'rotation_abs_error_sum':0.} for split in ('train','validation')};episodes=[]
            for e in manifest['episodes']:
                d=np.load(features/'features'/f'seed-{e["seed"]}.npz');a=d['actions'];ticks=d['ticks'];rot=np.abs(a[:,5])>.02;steady=np.array([not np.any(rot[np.abs(ticks-t)<=10]) for t in ticks])
                x=np.concatenate([(d['features']-s['feature_mean'])/s['feature_std'],(d['states']-s['state_mean'])/s['state_std']],1)
                with torch.no_grad():p=model(torch.from_numpy(x).cuda()).cpu().numpy()
                raw=p[:,:7]*s['action_std']+s['action_mean'];bad=steady&(np.abs(raw[:,5])>.02);v=totals[e['split']]
                v['steady']+=int(steady.sum());v['false']+=int(bad.sum());v['rotation']+=int(rot.sum());v['rotation_abs_error_sum']+=float(np.abs(raw[rot,5]-a[rot,5]).sum())
                if e['split']=='validation':
                    truth=np.flatnonzero(rot);pred=np.flatnonzero(np.abs(raw[:,5])>.02)
                    episodes.append({'seed':e['seed'],'steady_samples':int(steady.sum()),'false_hold_samples':int(bad.sum()),
                        'false_abs_wrist_p50_p95_max':np.quantile(np.abs(raw[bad,5]),[.5,.95,1]).tolist() if bad.any() else [],
                        'expert_first_rotation_s':float((ticks[truth[0]]-ticks[0])/10) if len(truth) else None,
                        'predicted_first_rotation_s':float((ticks[pred[0]]-ticks[0])/10) if len(pred) else None,
                        'onset_delta_s':float((ticks[pred[0]]-ticks[truth[0]])/10) if len(pred) and len(truth) else None})
            for v in totals.values():v['steady_false_fraction']=v['false']/v['steady'];v['rotation_mae_raw']=v['rotation_abs_error_sum']/v['rotation']
            report[name]={'splits':totals,'validation_episodes':episodes};del model;torch.cuda.empty_cache()
        atomic_json(root/'FeatureBC_hold_20261005/hold_generalization.json',report);print(json.dumps({k:v['splits'] for k,v in report.items()}),flush=True)

if __name__=='__main__':main()
