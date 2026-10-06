"""Read-only-data joint alias probe and summary-only metric repair."""
import fcntl,json,time
import numpy as np
import torch
from train_compare import ROOT,check_workloads,atomic_json
from probe_observability import scores

def main():
    with open('/tmp/baseline_collection_batch.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);check_workloads();torch.set_num_threads(2)
        root=ROOT/'models/policy100/runs';out=root/'observability_20261005';report=json.loads((out/'summary.json').read_text())
        # Early subset has no hold labels: balanced accuracy must average present classes only.
        for mode in report['probes']:
            d=np.load(out/f'{mode}_predictions.npz');p=d['prediction'];y=d['target'];idx=d['early_indices']
            report['probes'][mode]={'all':scores(p,y),'first10_rotation_observations_per_validation_episode':scores(p[idx],y[idx])}
        report['metric_note']='Balanced accuracy averages only classes present; early-rotation subset contains no hold labels. No model retrained.'
        manifest=json.loads((root/'ACT_Diffusion_100_20261004/dataset_manifest.json').read_text());groups={'train':[],'validation':[]}
        for e in manifest['episodes']:
            d=np.load(out/'features'/f'seed-{e["seed"]}.npz');q=d['states'];a=d['actions'];ticks=d['ticks'];r=np.abs(a[:,5])>.02
            steady=np.array([not np.any(r[np.abs(ticks-t)<=10]) for t in ticks])
            groups[e['split']].append((q,r,steady))
        def pack(split):return tuple(np.concatenate([x[k] for x in groups[split]]) for k in range(3))
        tq,tr,th=pack('train');vq,vr,vh=pack('validation');scale=np.array([.001]*6+[.00005]*2,np.float32)
        result={'threshold':'Every arm joint within 0.001rad and each finger within 0.00005m; holding excludes +-1sim-second around rotation labels',
                'not_proof':'Near-identical joints alone do not imply identical images or impossible control.'}
        for name,query,ref in [('validation_rotation_vs_training_steady_hold',vq[vr],tq[th]),('validation_steady_hold_vs_training_rotation',vq[vh][::5],tq[tr])]:
            a=torch.from_numpy(query/scale).cuda();b=torch.from_numpy(ref/scale).cuda();mins=[]
            for i in range(0,len(a),128):mins.append(torch.cdist(a[i:i+128],b,p=float('inf')).min(1).values.cpu().numpy())
            m=np.concatenate(mins);result[name]={'queries':len(m),'aliases_within_threshold':int((m<=1).sum()),'fraction':float((m<=1).mean()),'nearest_scaled_distance_median':float(np.median(m))}
        atomic_json(out/'joint_aliases.json',result);atomic_json(out/'summary.json',report);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
