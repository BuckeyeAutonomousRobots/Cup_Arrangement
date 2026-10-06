"""Episode-held-out image/joint conditioning probes. No simulator interfaces."""
import fcntl,json,os,time
from pathlib import Path
import numpy as np
import torch
from train_compare import ROOT,ACTPolicy,check_workloads,atomic_json,sha

class Probe(torch.nn.Module):
    def __init__(self,width,outputs=3):
        super().__init__();self.net=torch.nn.Sequential(torch.nn.Linear(width,128),torch.nn.ReLU(),torch.nn.Linear(128,128),torch.nn.ReLU(),torch.nn.Linear(128,outputs))
    def forward(self,x):return self.net(x)

def scores(pred,target):
    matrix=np.zeros((3,3),int)
    for a,b in zip(target,pred):matrix[a,b]+=1
    recalls=np.diag(matrix)/np.maximum(matrix.sum(1),1);rot=target!=0;hold=~rot
    return {'confusion_true_rows':matrix.tolist(),'accuracy':float((pred==target).mean()),'balanced_accuracy':float(recalls[matrix.sum(1)>0].mean()),
        'rotation_correct_direction_recall':float((pred[rot]==target[rot]).mean()) if rot.any() else None,
        'holding_false_rotation':float((pred[hold]!=0).mean()) if hold.any() else None,'samples':len(target)}

def main():
    start=time.monotonic()
    with open('/tmp/baseline_collection_batch.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);check_workloads()
        torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.4);torch.manual_seed(42)
        source=ROOT/'models/policy100/runs/ACT_Diffusion_100_20261004';out=ROOT/'models/policy100/runs/observability_20261005'
        assert not out.exists();out.mkdir();(out/'features').mkdir()
        with out.with_suffix('.claim').open('x') as f:json.dump({'pid':os.getpid(),'offline_only':True,'script_sha256':sha(Path(__file__))},f)
        manifest=json.loads((source/'dataset_manifest.json').read_text());stats=dict(np.load(source/'act_normalization.npz'))
        policy=ACTPolicy.from_pretrained(source/'ACT_100/last_policy').cuda().eval()
        mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None];std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
        report={'status':'extracting','pid':os.getpid(),'dataset_sha256':sha(source/'dataset_manifest.json'),
            'backbone_sha256':sha(source/'ACT_100/last_policy/model.safetensors'),'features':'Frozen ACT5k ResNet18 spatial adaptive average 2x3 (3072); measured 8 joints',
            'classes':['hold_abs_wrist_le_0.02','negative_rotation','positive_rotation'],'probes':{}}
        groups={'train':[],'validation':[]};seed_rows={};offsets={'train':0,'validation':0}
        for e in manifest['episodes']:
            assert time.monotonic()-start<600;check_workloads()
            cache=Path(e['cache']);images=np.load(cache/'images.npy',mmap_mode='r');parts=[]
            with torch.inference_mode():
                for i in range(0,len(images),32):
                    rgb=torch.from_numpy(np.asarray(images[i:i+32]).copy()).cuda().permute(0,3,1,2).float()/255
                    f=policy.model.backbone((rgb-mean)/std)['feature_map'];parts.append(torch.nn.functional.adaptive_avg_pool2d(f,(2,3)).flatten(1).cpu().numpy())
            f=np.concatenate(parts);q=np.load(cache/'states.npy');a=np.load(cache/'actions.npy');ticks=np.load(cache/'ticks.npy')
            np.savez(out/'features'/f'seed-{e["seed"]}.npz',features=f,states=q,actions=a,ticks=ticks)
            split=e['split'];seed_rows[e['seed']]=(split,offsets[split],len(a));offsets[split]+=len(a)
            groups[split].append((f,q,a,ticks,e['seed']));report.update(last_seed=e['seed']);atomic_json(out/'summary.json',report)
            if e['seed']%10==9:print(json.dumps({'features_through_seed':e['seed'],'elapsed_s':time.monotonic()-start}),flush=True)
        del policy;torch.cuda.empty_cache()
        trainf=np.concatenate([r[0] for r in groups['train']]);fm=trainf.mean(0);fs=np.maximum(trainf.std(0),.001)
        np.savez(out/'feature_normalization.npz',feature_mean=fm,feature_std=fs,**stats)
        data={}
        for split,rows in groups.items():
            f=np.concatenate([r[0] for r in rows]);q=np.concatenate([r[1] for r in rows]);a=np.concatenate([r[2] for r in rows]);label=np.where(a[:,5]<-.02,1,np.where(a[:,5]>.02,2,0))
            data[split]={'images':torch.from_numpy((f-fm)/fs).cuda(),'joints':torch.from_numpy((q-stats['state_mean'])/stats['state_std']).cuda(),'labels':torch.from_numpy(label).cuda()}
        early=[]
        for e in manifest['episodes']:
            if e['split']!='validation':continue
            _,offset,n=seed_rows[e['seed']];labels=data['validation']['labels'][offset:offset+n].cpu().numpy();r=np.flatnonzero(labels!=0)[:10];early.extend((offset+r).tolist())
        for mode in ('joints','images','combined'):
            check_workloads();torch.manual_seed(42)
            def inputs(split):return torch.cat([data[split]['images'],data[split]['joints']],1) if mode=='combined' else data[split][mode]
            x=inputs('train');v=inputs('validation');y=data['train']['labels'];gt=data['validation']['labels'].cpu().numpy()
            model=Probe(x.shape[1]).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
            for step in range(1000):
                assert time.monotonic()-start<900
                if step%250==0:check_workloads()
                idx=torch.randint(len(x),(256,),device='cuda');loss=torch.nn.functional.cross_entropy(model(x[idx]),y[idx]);opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            with torch.no_grad():pred=torch.cat([model(v[i:i+512]).argmax(1).cpu() for i in range(0,len(v),512)]).numpy()
            report['probes'][mode]={'all':scores(pred,gt),'first10_rotation_observations_per_validation_episode':scores(pred[early],gt[early])}
            torch.save(model.state_dict(),out/f'{mode}_probe.pt');np.savez(out/f'{mode}_predictions.npz',prediction=pred,target=gt,early_indices=early)
            atomic_json(out/'summary.json',report);print(json.dumps({'mode':mode,**report['probes'][mode]}),flush=True)
            del model,opt,x,v
        report.update(status='completed',wall_seconds=time.monotonic()-start,limitations='Frozen-feature probes do not prove all image information is recoverable or absent; classifier accuracy is not action-policy success.')
        atomic_json(out/'summary.json',report)

if __name__=='__main__':main()
