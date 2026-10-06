"""Stationary onset versus moving labels; image-only probe uses no future input."""
import json
from pathlib import Path
import numpy as np
import torch
from train_compare import ROOT,atomic_json
from probe_observability import Probe

def main():
    torch.set_num_threads(2);root=ROOT/'models/policy100/runs';stats=dict(np.load(root/'observability_20261005/feature_normalization.npz'))
    model=Probe(3072);model.load_state_dict(torch.load(root/'observability_20261005/images_probe.pt',map_location='cpu',weights_only=True));model.eval()
    manifest=json.loads((root/'ACT_Diffusion_100_20261004/dataset_manifest.json').read_text());groups={k:[] for k in ('stationary_pre_onset','stationary_onset','moving_rotation')};counts={}
    for split in ('train','validation'):
        c={'rotation':0,'rotation_stationary':0,'rotation_moving':0,'onsets':0,'stationary_onsets':0}
        for e in manifest['episodes']:
            if e['split']!=split:continue
            d=np.load(root/'observability_20261005/features'/('seed-'+str(e['seed'])+'.npz'));a=d['actions'];q=d['states'];t=d['ticks'];rot=np.abs(a[:,5])>.02;valid=np.r_[False,np.diff(t)==1];v=np.zeros(len(t));v[1:]=np.diff(q[:,5])/np.maximum(np.diff(t)/10,1e-6)
            onset=rot&~np.r_[False,rot[:-1]]&valid;stationary=valid&(np.abs(v)<=.005);moving=valid&(np.abs(v)>.02)
            c['rotation']+=int(rot.sum());c['rotation_stationary']+=int((rot&stationary).sum());c['rotation_moving']+=int((rot&moving).sum());c['onsets']+=int(onset.sum());c['stationary_onsets']+=int((onset&stationary).sum())
            if split!='validation':continue
            with torch.no_grad():p=model(torch.from_numpy((d['features']-stats['feature_mean'])/stats['feature_std'])).numpy()
            for i in np.flatnonzero(onset):
                if abs(v[i-1])<=.005:groups['stationary_pre_onset'].append((p[i-1],1 if a[i,5]<0 else 2))
                if stationary[i]:groups['stationary_onset'].append((p[i],1 if a[i,5]<0 else 2))
            for i in np.flatnonzero(rot&moving):groups['moving_rotation'].append((p[i],1 if a[i,5]<0 else 2))
        counts[split]=c
    result={'counts':counts,'image_probe':{k:{'n':len(v),'conditional_direction_accuracy':float(np.mean([(np.argmax(p[1:])+1)==y for p,y in v])),
        'full_three_class_rotation_recall':float(np.mean([np.argmax(p)==y for p,y in v]))} for k,v in groups.items()},
        'interpretation':'Pre-onset labels identify upcoming direction only for analysis; input is previous observation, not future image/action. Tiny stationary subset and current-label-trained probe cannot establish unidentifiability.'}
    atomic_json(root/'rotation_initiation_audit_20261005.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
