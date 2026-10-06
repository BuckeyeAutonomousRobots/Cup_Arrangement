#!/usr/bin/env python3
"""Offline LeRobot ACT smoke training. Never publishes commands or uploads data.

RGB + 8 measured joints -> 20 x (6 commanded velocities + common jaw target).
Command timing is previous receipt-clock alignment, not exact actuation timing.
"""
import argparse,bisect,datetime,hashlib,json,math,os,random,time,traceback
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from lerobot.configs.types import FeatureType,PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from episode_manifest import load_manifest

JOINTS=['right_'+n+'_joint' for n in ('shoulder_pan','shoulder_lift','elbow','wrist_1','wrist_2','wrist_3')]
JOINTS+=['right_robotiq_hande_left_finger_joint','right_robotiq_hande_right_finger_joint']
ARM='/right_joint_group_velocity_controller/commands'
GRIP='/right_hande_position_controller/commands'
IMAGE='observation.images.wrist'
def now():return datetime.datetime.now().astimezone().isoformat()
def atomic_json(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def prepare_episode(path,cache):
    summary=json.loads((path/'summary.json').read_text())
    if summary['status']!='controller_pass':raise ValueError('Non-success episode: '+str(path))
    events=[json.loads(x) for x in (path/'events.jsonl').read_text().splitlines()]
    begin_event=next(x for x in events if x['event']=='cheat_started')
    end_event=next(x for x in events if x['event']=='cheat_finished')
    begin=begin_event['receipt_sim_s'];end=end_event['receipt_sim_s']
    clocks=[]
    series={k:[] for k in [*JOINTS,ARM,GRIP]}
    for line in (path/'robot_data.jsonl').open():
        row=json.loads(line);msg=row['message'];topic=row['topic']
        if topic=='/clock':
            clocks.append((row['receipt_monotonic_s'],msg['clock']['sec']+msg['clock']['nanosec']*1e-9))
        if topic=='/joint_states':
            stamp=msg['header']['stamp'];t=stamp['sec']+stamp['nanosec']*1e-9
            for name,pos in zip(msg['name'],msg['position']):
                if name in JOINTS:series[name].append((t,pos))
        elif topic in (ARM,GRIP) and row['receipt_sim_s'] is not None:
            series[topic].append((row['receipt_sim_s'],msg['data']))
    for values in series.values():values.sort(key=lambda x:x[0])
    boundary_repaired=False
    if begin is None:
        later=[(wall,t) for wall,t in clocks if wall>=begin_event['receipt_monotonic_s']]
        if not later or later[0][0]-begin_event['receipt_monotonic_s']>1.:
            raise ValueError('No timely clock at controller start')
        begin=later[0][1];boundary_repaired=True
    if end is None:raise ValueError('Missing controller end clock')
    times={k:[x[0] for x in v] for k,v in series.items()}
    frames=[json.loads(x) for x in (path/'frames.jsonl').read_text().splitlines()]
    ft=[x['stamp_sim_s'] for x in frames]
    if any(b<=a for a,b in zip(ft,ft[1:])):raise ValueError('Nonmonotonic images')
    images=[];states=[];actions=[];ticks=[];age_max=0.;dropped=0
    for tick in range(math.ceil(begin*10),math.floor(end*10)+1):
        t=tick/10.;idx=bisect.bisect_right(ft,t)-1
        if idx<0 or t-ft[idx]>.100001:dropped+=1;continue
        prior={};ages=[]
        for key,values in series.items():
            i=bisect.bisect_right(times[key],t)-1
            if i<0 or (key!=GRIP and t-values[i][0]>.100001):break
            prior[key]=values[i][1]
            if key!=GRIP:ages.append(t-values[i][0])
        if len(prior)!=len(series):dropped+=1;continue
        if len(prior[ARM])!=6 or len(prior[GRIP])!=2:raise ValueError('Bad action dimensions')
        if abs(prior[GRIP][0]-prior[GRIP][1])>1e-6:raise ValueError('Independent jaw commands')
        rgb=cv2.imread(str(path/frames[idx]['file']))
        if rgb is None:raise ValueError('Missing frame')
        rgb=cv2.cvtColor(cv2.resize(rgb,(320,240),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
        images.append(rgb);states.append([prior[k] for k in JOINTS])
        actions.append([*prior[ARM],float(np.mean(prior[GRIP]))]);ticks.append(tick)
        age_max=max(age_max,t-ft[idx],*ages)
    if len(ticks)<50:raise ValueError('Too few causal samples')
    states=np.asarray(states,dtype=np.float32);actions=np.asarray(actions,dtype=np.float32)
    if not np.isfinite(states).all() or not np.isfinite(actions).all():raise ValueError('Nonfinite data')
    if np.max(np.abs(actions[:,:6]))>.246:raise ValueError('Unexpected arm speed in dataset')
    if actions[:,6].min()<-.026 or actions[:,6].max()>.001:raise ValueError('Jaw target out of range')
    cache.mkdir(parents=True,exist_ok=False)
    for name,value in [('images',np.asarray(images,dtype=np.uint8)),('states',states),('actions',actions),('ticks',np.asarray(ticks))]:
        np.save(cache/(name+'.npy'),value)
    return dict(episode=str(path),cache=str(cache),samples=len(ticks),dropped_grid_steps=dropped,
        max_causal_age_s=age_max,start_boundary_from_first_clock=boundary_repaired,controller_sha256=summary['source_sha256']['scripts/cup_task.py'],
        summary_sha256=sha(path/'summary.json'),robot_data_sha256=sha(path/'robot_data.jsonl'))

class Chunks(Dataset):
    def __init__(self,episodes,stats,chunk=20):
        self.episodes=[];self.index=[];self.stats=stats;self.chunk=chunk
        for entry in episodes:
            p=Path(entry['cache'])
            e={k:np.load(p/(k+'.npy'),mmap_mode='r') for k in ['images','states','actions','ticks']}
            number=len(self.episodes);self.episodes.append(e)
            for i in range(len(e['ticks'])):self.index.append((number,i))
    def __len__(self):return len(self.index)
    def __getitem__(self,index):
        ep,i=self.index[index];e=self.episodes[ep]
        # Never cross an episode boundary OR a dropped time-grid sample.
        n=1
        while n<self.chunk and i+n<len(e['ticks']) and e['ticks'][i+n]==e['ticks'][i]+n:n+=1
        actions=np.repeat(np.asarray(e['actions'][i+n-1])[None],self.chunk,axis=0)
        actions[:n]=e['actions'][i:i+n]
        actions=(actions-self.stats['action_mean'])/self.stats['action_std']
        state=(e['states'][i]-self.stats['state_mean'])/self.stats['state_std']
        rgb=np.asarray(e['images'][i],dtype=np.float32).transpose(2,0,1)/255.
        rgb=(rgb-np.array([.485,.456,.406],dtype=np.float32)[:,None,None])/np.array([.229,.224,.225],dtype=np.float32)[:,None,None]
        return {IMAGE:torch.from_numpy(rgb), 'observation.state':torch.from_numpy(state.copy()),
            'action':torch.from_numpy(actions.copy()),'action_is_pad':torch.arange(self.chunk)>=n}

def main():
    ap=argparse.ArgumentParser()
    source=ap.add_mutually_exclusive_group(required=True)
    source.add_argument('--source',type=Path)
    source.add_argument('--manifest',type=Path)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--steps',type=int,default=5000)
    ap.add_argument('--batch-size',type=int,default=8);a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    status=dict(status='preparing',started=now(),pid=os.getpid(),step=0,total_steps=a.steps,
        trainer_sha256=sha(Path(__file__)),torch_version=torch.__version__)
    def update(**kw):status.update(kw,updated=now());atomic_json(a.output/'status.json',status)
    update()
    try:
        torch.set_num_threads(2);cv2.setNumThreads(1)
        random.seed(42);np.random.seed(42);torch.manual_seed(42)
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; refusing CPU fallback')
        torch.cuda.set_per_process_memory_fraction(.40)
        selection = load_manifest(a.manifest) if a.manifest else [dict(seed=seed,
            split='train' if seed<8 else 'validation', path=a.source/f'episode-{seed:03d}-seed-{seed}') for seed in range(10)]
        episodes=[]
        for row in selection:
            seed=row['seed']
            update(preparing_seed=seed)
            episodes.append(dict(prepare_episode(row['path'],a.output/'cache'/f'seed-{seed}'),seed=seed,split=row['split']))
        training=[e for e in episodes if e['split']=='train']
        heldout=[e for e in episodes if e['split']=='validation']
        if len(set(e['controller_sha256'] for e in episodes))!=1:raise ValueError('Mixed controller revisions')
        atomic_json(a.output/'dataset_manifest.json',dict(episodes=episodes,train_seeds=[e['seed'] for e in training],validation_seeds=[e['seed'] for e in heldout],
            frequency_hz=10,chunk_size=20,image_wh=[320,240],state_names=JOINTS,
            action_names=[*JOINTS[:6],'common_gripper_position'],action_units=['rad/s']*6+['m'],
            command_timing='previous receipt simulation clock; NOT exact actuation time',
            policy_inputs='RGB and joints only; no cup/plate ground truth'))
        train_s=np.concatenate([np.load(Path(e['cache'])/'states.npy') for e in training])
        train_a=np.concatenate([np.load(Path(e['cache'])/'actions.npy') for e in training])
        stats=dict(state_mean=train_s.mean(0),state_std=np.maximum(train_s.std(0),np.array([.001]*6+[.0001]*2,dtype=np.float32)),
            action_mean=train_a.mean(0),action_std=np.maximum(train_a.std(0),np.array([.001]*6+[.0001],dtype=np.float32)))
        np.savez(a.output/'normalization.npz',**stats)
        train=Chunks(training,stats);val=Chunks(heldout,stats)
        assert set(e['episode'] for e in training).isdisjoint(e['episode'] for e in heldout)
        # DataLoader workers only read cached images; no ROS or robot access.
        loader=DataLoader(train,batch_size=a.batch_size,shuffle=True,num_workers=2,pin_memory=True,drop_last=True,persistent_workers=True)
        val_indices=np.linspace(0,len(val)-1,min(256,len(val)),dtype=int).tolist()
        validation=DataLoader(torch.utils.data.Subset(val,val_indices),batch_size=a.batch_size,num_workers=0)
        config=ACTConfig(input_features={IMAGE:PolicyFeature(type=FeatureType.VISUAL,shape=(3,240,320)),
            'observation.state':PolicyFeature(type=FeatureType.STATE,shape=(8,))},
            output_features={'action':PolicyFeature(type=FeatureType.ACTION,shape=(7,))},
            chunk_size=20,n_action_steps=1,temporal_ensemble_coeff=.01,
            optimizer_lr=1e-4,optimizer_lr_backbone=1e-5,device='cuda',push_to_hub=False)
        config.save_pretrained(a.output/'policy_config')
        policy=ACTPolicy(config).cuda()
        optimizer=torch.optim.AdamW(policy.get_optim_params(),lr=1e-4,weight_decay=1e-4)
        update(status='training',train_samples=len(train),validation_samples=len(val),
            gpu=torch.cuda.get_device_name(),parameters=sum(p.numel() for p in policy.parameters()))
        def gpu(batch):return {k:v.cuda(non_blocking=True) for k,v in batch.items()}
        def evaluate():
            policy.eval();error=np.zeros(7);count=0;norm=0.
            with torch.no_grad():
                for batch in validation:
                    batch=gpu(batch);pred=policy.predict_action_chunk({k:v for k,v in batch.items() if k not in ('action','action_is_pad')})
                    if not torch.isfinite(pred).all():raise RuntimeError('Nonfinite validation prediction')
                    mask=(~batch['action_is_pad']).unsqueeze(-1)
                    err=(pred-batch['action']).abs()*mask
                    norm+=err.sum().item();count+=mask.sum().item()
                    error+=err.sum((0,1)).cpu().numpy()*stats['action_std']
            policy.train()
            return dict(normalized_l1=norm/(count*7),mae_per_action=(error/count).tolist(),validation_windows=len(val_indices))
        metrics=(a.output/'metrics.jsonl').open('w',buffering=1)
        best=float('inf');iterator=iter(loader);start=time.monotonic();policy.train()
        for step in range(1,a.steps+1):
            try:batch=next(iterator)
            except StopIteration:iterator=iter(loader);batch=next(iterator)
            loss,details=policy(gpu(batch))
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(),1.);optimizer.step()
            # Small yield limits contention with concurrent Gazebo rendering.
            time.sleep(.02)
            if step==1 or step%25==0:
                elapsed=time.monotonic()-start
                update(step=step,loss=float(loss.item()),training_elapsed_s=elapsed,
                    estimated_remaining_s=(a.steps-step)*elapsed/step,
                    cuda_peak_memory_gb=torch.cuda.max_memory_allocated()/1e9)
                print(json.dumps(status),flush=True)
            if step==1 or step%500==0 or step==a.steps:
                report=dict(step=step,train_loss=float(loss.item()),**evaluate())
                metrics.write(json.dumps(report)+'\n')
                if report['normalized_l1']<best:
                    best=report['normalized_l1'];policy.save_pretrained(a.output/'best_policy')
                    atomic_json(a.output/'best_metrics.json',report)
                policy.save_pretrained(a.output/'last_policy')
                torch.save({'step':step,'optimizer':optimizer.state_dict()},a.output/'optimizer.pt')
                update(validation=report,best_validation_l1=best)
        policy.eval()
        # Verify saved weights can reload and produce the same inference chunk.
        probe=gpu(next(iter(validation)));probe={k:v for k,v in probe.items() if k not in ('action','action_is_pad')}
        with torch.no_grad():expected=policy.predict_action_chunk(probe)
        reloaded=ACTPolicy.from_pretrained(a.output/'last_policy').cuda().eval()
        with torch.no_grad():actual=reloaded.predict_action_chunk(probe)
        torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-5)
        update(status='completed',finished=now(),checkpoint_reload_pass=True,closed_loop_evaluated=False)
    except Exception:
        update(status='failed',finished=now(),error=traceback.format_exc())
        raise

if __name__=='__main__':main()
