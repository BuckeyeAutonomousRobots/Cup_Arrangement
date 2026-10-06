"""Post-evaluation coverage diagnosis; no model changes or threshold selection."""
import fcntl,json
import numpy as np
import torch
from train_compare import ROOT,check_workloads,atomic_json

def main():
    with open('/tmp/baseline_collection_batch.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);check_workloads();torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.4)
        out=ROOT/'models/policy100/runs/FeatureBC_calibrated_20261005';report=json.loads((out/'summary.json').read_text());stats=dict(np.load(out/'normalization.npz'))
        references={'hold':[],'rotation':[]};queries={'false_hold':[],'correct_hold':[]};queryseeds={'false_hold':[],'correct_hold':[]}
        def data(seed):
            d=np.load(out/'features'/f'seed-{seed}.npz');rot=np.abs(d['actions'][:,5])>.02;ticks=d['ticks'];steady=np.array([not np.any(rot[np.abs(ticks-t)<=10]) for t in ticks])
            x=np.concatenate([(d['features']-stats['feature_mean'])/stats['feature_std']/np.sqrt(3072),(d['states']-stats['state_mean'])/stats['state_std']/np.sqrt(8)],1).astype(np.float32)
            return x,rot,steady
        for seed in report['fit_seeds']:
            x,r,h=data(seed);references['hold'].append(x[h]);references['rotation'].append(x[r])
        for seed in report['outer_seeds']:
            x,r,h=data(seed);p=np.load(out/f'validation-seed-{seed}.npz')['raw'];bad=h&(np.abs(p[:,5])>.02)
            for key,mask in [('false_hold',bad),('correct_hold',h&~bad)]:
                queries[key].append(x[mask]);queryseeds[key].extend([seed]*int(mask.sum()))
        result={'metric':'Euclidean distance in fit-standardized CNN features and joints, each modality divided by sqrt(number of dimensions); representation-dependent, not proof of physical coverage',
                'no_training_or_selection':True,'groups':{}}
        refs={k:torch.from_numpy(np.concatenate(v)).cuda() for k,v in references.items()}
        for key,pieces in queries.items():
            x=np.concatenate(pieces);seeds=np.array(queryseeds[key]);indices=np.arange(len(x)) if key=='false_hold' else np.linspace(0,len(x)-1,min(300,len(x))).astype(int)
            x=torch.from_numpy(x[indices]).cuda();distances={}
            for phase,ref in refs.items():
                vals=[]
                for i in range(0,len(x),64):vals.append(torch.cdist(x[i:i+64],ref).min(1).values.cpu().numpy())
                distances[phase]=np.concatenate(vals)
            result['groups'][key]={'queries':len(x),'nearest_hold_distance_p50_p95':np.quantile(distances['hold'],[.5,.95]).tolist(),
                'nearest_rotation_distance_p50_p95':np.quantile(distances['rotation'],[.5,.95]).tolist(),
                'fraction_closer_to_training_rotation':float((distances['rotation']<distances['hold']).mean()),
                'seed_counts':{str(int(s)):int((seeds[indices]==s).sum()) for s in np.unique(seeds[indices])}}
        atomic_json(out/'coverage_analysis.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
