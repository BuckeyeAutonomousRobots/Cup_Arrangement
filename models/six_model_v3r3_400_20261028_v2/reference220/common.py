from pathlib import Path
import json,hashlib,os,datetime
import numpy as np,torch
from torch.utils.data import Dataset
from lerobot.configs.types import FeatureType,PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
O=Path(__file__).resolve().parent
S=Path(__file__).resolve().parents[2]/'ros_backend1.1/runtime/fixed_angle_constant_speed_position_only_v3r3_20261025'
IMAGE='observation.images.wrist';KINDS=['act','diffusion'];REPS=['velocity','joint_position','ee_pose'];HORIZON=16;EXECUTE=4
MEAN=np.array([.485,.456,.406],np.float32)[:,None,None];STD=np.array([.229,.224,.225],np.float32)[:,None,None]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(p,obj):
 p=Path(p);t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2));os.replace(t,p)
def make_stats(entries,rep):
 ss=np.concatenate([np.load(Path(e['cache'])/'states.npy') for e in entries]);aa=np.concatenate([np.load(Path(e['cache'])/(rep+'.npy')) for e in entries]);stats={}
 for key,arr in [('state',ss),('action',aa)]:
  floor=np.full(arr.shape[1],.001,np.float32);floor[-(2 if key=='state' else 1):]=.0001
  stats[key+'_center']=(arr.max(0)+arr.min(0))/2;stats[key+'_scale']=np.maximum((arr.max(0)-arr.min(0))/2,floor)
  stats[key+'_std']=np.maximum(arr.std(0),floor);stats[key+'_min']=arr.min(0);stats[key+'_max']=arr.max(0)
 return stats
class Chunks(Dataset):
 def __init__(self,entries,stats,rep,kind):
  self.entries=entries;self.stats=stats;self.rep=rep;self.kind=kind;self.episodes=[];self.index=[]
  for en in entries:
   c=Path(en['cache']);e={n:np.load(c/(n+'.npy'),mmap_mode='r') for n in ['images','states',rep,'ticks']};num=len(self.episodes);self.episodes.append(e)
   for segment in en['segments']:
    assert all(e['ticks'][b]==e['ticks'][a]+1 for a,b in zip(segment,segment[1:]))
    self.index.extend((num,i,segment[-1]+1) for i in segment)
 def __len__(self):return len(self.index)
 def __getitem__(self,index):
  ep,i,end=self.index[index];e=self.episodes[ep];n=min(HORIZON,end-i);a=np.repeat(np.asarray(e[self.rep][i+n-1])[None],HORIZON,axis=0);a[:n]=e[self.rep][i:i+n]
  a=(a-self.stats['action_center'])/self.stats['action_scale'];state=(e['states'][i]-self.stats['state_center'])/self.stats['state_scale']
  image=(np.asarray(e['images'][i],np.float32).transpose(2,0,1)/255-MEAN)/STD
  if self.kind=='diffusion':image=image[None];state=state[None]
  return {IMAGE:torch.from_numpy(image.copy()),'observation.state':torch.from_numpy(state.copy()),'action':torch.from_numpy(a.copy()),'action_is_pad':torch.arange(HORIZON)>=n}
def create_policy(kind,dim):
 inputs={IMAGE:PolicyFeature(type=FeatureType.VISUAL,shape=(3,240,320)),'observation.state':PolicyFeature(type=FeatureType.STATE,shape=(8,))};outputs={'action':PolicyFeature(type=FeatureType.ACTION,shape=(dim,))}
 if kind=='act':
  cfg=ACTConfig(input_features=inputs,output_features=outputs,chunk_size=HORIZON,n_action_steps=EXECUTE,temporal_ensemble_coeff=None,pretrained_backbone_weights=None,optimizer_lr=1e-4,optimizer_lr_backbone=1e-5,device='cuda',push_to_hub=False);cls=ACTPolicy
 else:
  cfg=DiffusionConfig(input_features=inputs,output_features=outputs,n_obs_steps=1,horizon=HORIZON,n_action_steps=EXECUTE,noise_scheduler_type='DDIM',num_train_timesteps=100,num_inference_steps=16,do_mask_loss_for_padding=True,crop_shape=None,resize_shape=None,pretrained_backbone_weights=None,use_group_norm=True,device='cuda',push_to_hub=False);cls=DiffusionPolicy
 return cls(cfg).cuda(),cfg
@torch.inference_mode()
def predict(policy,kind,batch):
 if kind=='act':out=policy.predict_action_chunk({k:v for k,v in batch.items() if k in [IMAGE,'observation.state']})[:,:EXECUTE]
 else:out=policy.diffusion.generate_actions({'observation.state':batch['observation.state'],'observation.images':batch[IMAGE].unsqueeze(2)})
 # Same train-range output restriction for both policy families; count saturation in deployment.
 return out.clamp(-1,1)
def load_policy(kind,path):return (ACTPolicy if kind=='act' else DiffusionPolicy).from_pretrained(path).cuda().eval()
