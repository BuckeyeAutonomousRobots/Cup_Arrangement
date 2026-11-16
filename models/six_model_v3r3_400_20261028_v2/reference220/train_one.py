"""One bounded initial model; validation-only selection, untouched held-out test and raw data."""
import argparse,json,time,random,shutil,traceback,os,signal,importlib.metadata
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader,Subset
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from common import *
STOP=False
def cancel(*_):
 global STOP;STOP=True

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--kind',choices=KINDS,required=True);ap.add_argument('--rep',choices=REPS,required=True);ap.add_argument('--steps',type=int,default=10000);ap.add_argument('--batch',type=int,default=16);a=ap.parse_args()
 run=O/'runs'/f'{a.kind}_{a.rep}';run.mkdir(parents=True,exist_ok=False);status={'kind':a.kind,'representation':a.rep,'pid':os.getpid(),'status':'preparing','step':0,'steps':a.steps,'batch':a.batch,'started':now()}
 def update(**kw):status.update(kw,updated=now());save(run/'status.json',status);print(json.dumps(status),flush=True)
 deadline=time.monotonic()+5400
 def guard():
  assert (S/'control/USER_COLLECTION_PAUSED.json').exists();assert shutil.disk_usage(O).free/2**30>65
  if STOP or time.monotonic()>deadline:raise InterruptedError('Bounded 90minute model deadline or user stop')
 signal.signal(signal.SIGTERM,cancel);signal.signal(signal.SIGINT,cancel)
 try:
  torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.85);random.seed(42);np.random.seed(42);torch.manual_seed(42)
  manifest=json.loads((O/'FROZEN_MANIFEST.json').read_text());train=[e for e in manifest['episodes'] if e['split']=='train'];val=[e for e in manifest['episodes'] if e['split']=='validation'];assert len(train)==176 and len(val)==22
  for e in train+val:
   for n in ['states',a.rep,'ticks']:assert sha(Path(e['cache'])/(n+'.npy'))==e['cache_sha256'][n]
  stats=make_stats(train,a.rep);np.savez(run/'normalization.npz',**stats)
  ds=Chunks(train,stats,a.rep,a.kind);vs=Chunks(val,stats,a.rep,a.kind)
  for dataset in [ds,vs]:
   for idx in np.linspace(0,len(dataset)-1,64,dtype=int):
    ep,i,end=dataset.index[idx];sample=dataset[int(idx)];np.testing.assert_allclose(sample['action'][0].numpy()*stats['action_scale']+stats['action_center'],dataset.episodes[ep][a.rep][i],atol=3e-7)
    assert not sample['action_is_pad'][0];n=min(HORIZON,end-i);assert int((~sample['action_is_pad']).sum())==n
  save(run/'DATA_ALIGNMENT.json',{'status':'passed','anchors_checked':128,'current_action_index':0,'history_steps':1,'horizon':16,'execution_steps':4,'view':'cycle_excluded','gap_padding_masked':True,'train_only_normalization':True,'splits':[176,22,22],'frozen_manifest_sha256':sha(O/'FROZEN_MANIFEST.json')})
  policy,cfg=create_policy(a.kind,len(stats['action_center']));cfg.save_pretrained(run/'policy_config')
  if a.kind=='act':optimizer=torch.optim.AdamW(policy.get_optim_params(),lr=1e-4,weight_decay=1e-4);ema=None;scheduler=None
  else:optimizer=torch.optim.Adam(policy.parameters(),lr=1e-4,betas=(.95,.999),eps=1e-8,weight_decay=1e-6);scheduler=get_scheduler('cosine',optimizer,num_warmup_steps=500,num_training_steps=a.steps);ema=EMAModel(policy.parameters(),power=.75,use_ema_warmup=True)
  loader=DataLoader(ds,batch_size=a.batch,shuffle=True,num_workers=2,persistent_workers=True,pin_memory=True,drop_last=True,generator=torch.Generator().manual_seed(42));vl=DataLoader(Subset(vs,np.linspace(0,len(vs)-1,min(256,len(vs)),dtype=int).tolist()),batch_size=a.batch,num_workers=0)
  gpu=lambda b:{k:v.cuda(non_blocking=True) for k,v in b.items()}
  probe=gpu(next(iter(loader)));loss,_=policy(probe);assert torch.isfinite(loss);loss.backward();optimizer.zero_grad(set_to_none=True);policy.eval();out=predict(policy,a.kind,probe);assert out.shape==(a.batch,4,len(stats['action_center'])) and torch.isfinite(out).all()
  def evaluate():
   policy.eval();errs=[]
   with torch.random.fork_rng(devices=[0]),torch.inference_mode():
    torch.manual_seed(1234)
    for b in vl:
     guard();b=gpu(b);pr=predict(policy,a.kind,b);errs.append((pr[:,0]-b['action'][:,0]).abs().cpu().numpy()*stats['action_scale'])
   er=np.concatenate(errs);return {'validation_first_step_std_scaled_l1':float((er/stats['action_std']).mean()),'mae_per_action_physical_units':er.mean(0).tolist(),'windows':len(er),'selection':'validation first-step physical error divided by train-only action std; never compare training losses across policy families','noise_seed':1234}
  save(run/'CONFIG.json',{'algorithm':a.kind,'representation':a.rep,'steps':a.steps,'batch':a.batch,'random_seed':42,'image_preprocessing':manifest['image_preprocessing'],'observations':'current wrist RGB+8 measured joint positions','action_horizon':16,'execute':4,'frequency_hz':10,'normalization':'Train-only per-dimension minmax with floors .001 for arm/pose coordinates and .0001m for gripper; same across families within representation. ImageNet RGB mean/std. Both outputs bounded to normalized[-1,1].','pretrained_vision':False,'checkpoint_selection':'validation only','source_sha256':{n:sha(O/n) for n in ['common.py','train_one.py','FROZEN_MANIFEST.json']},'libraries':{n:importlib.metadata.version(n) for n in ['torch','torchvision','lerobot','diffusers']},'initial_window_passes':a.steps*a.batch/len(ds)})
  best=float('inf');it=iter(loader);start=time.monotonic();update(status='training',train_windows=len(ds),validation_windows=len(vs),parameters=sum(p.numel() for p in policy.parameters()),gpu=torch.cuda.get_device_name(),initial_window_passes=a.steps*a.batch/len(ds),smoke_pass=True)
  for step in range(1,a.steps+1):
   if step%25==1:guard()
   try:b=next(it)
   except StopIteration:it=iter(loader);b=next(it)
   policy.train();loss,_=policy(gpu(b));assert torch.isfinite(loss);optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(policy.parameters(),1);optimizer.step()
   if scheduler:scheduler.step();ema.step(policy.parameters())
   if step==1 or step%100==0:
    elapsed=time.monotonic()-start;update(step=step,training_loss=float(loss.detach()),elapsed_s=elapsed,estimated_remaining_s=(a.steps-step)*elapsed/step,cuda_peak_GiB=torch.cuda.max_memory_allocated()/2**30)
   if step%1000==0 or step==a.steps:
    if ema:ema.store(policy.parameters());ema.copy_to(policy.parameters())
    try:
     metric=evaluate();metric['step']=step
     with (run/'metrics.jsonl').open('a') as f:f.write(json.dumps(metric)+'\n')
     policy.save_pretrained(run/'last_policy')
     if metric['validation_first_step_std_scaled_l1']<best:best=metric['validation_first_step_std_scaled_l1'];policy.save_pretrained(run/'best_policy');save(run/'best_metrics.json',metric)
     update(validation=metric,best_validation=best)
    finally:
     if ema:ema.restore(policy.parameters())
  del policy;torch.cuda.empty_cache();bestp=load_policy(a.kind,run/'best_policy');probe=gpu(next(iter(vl)))
  with torch.random.fork_rng(devices=[0]):
   torch.manual_seed(1234);expected=predict(bestp,a.kind,probe);reloaded=load_policy(a.kind,run/'best_policy');torch.manual_seed(1234);actual=predict(reloaded,a.kind,probe);torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-5)
  update(status='completed',checkpoint_reload_pass=True,best_checkpoint_sha256=sha(run/'best_policy/model.safetensors'),finished=now())
 except BaseException:update(status='failed_or_interrupted',error=traceback.format_exc());raise
if __name__=='__main__':main()
