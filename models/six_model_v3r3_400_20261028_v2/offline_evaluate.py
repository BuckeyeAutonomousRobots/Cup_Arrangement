"""One held-out report per frozen best checkpoint; no training/test tuning."""
from common import *
from torch.utils.data import DataLoader,Subset
import argparse,time
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();cfg=json.loads((a.run/'CONFIG.json').read_text());kind=cfg['algorithm'];rep=cfg['representation'];stats=dict(np.load(a.run/'normalization.npz'));m=json.loads((O/'FROZEN_MANIFEST.json').read_text())
assert not (a.run/'TEST_METRICS.json').exists();torch.set_num_threads(4);torch.manual_seed(4242)
entries=[e for e in m['episodes'] if e['split']=='test'];ds=Chunks(entries,stats,rep,kind)
indices=np.linspace(0,len(ds)-1,min(1024,len(ds)),dtype=int).tolist();loader=DataLoader(Subset(ds,indices),batch_size=16,num_workers=0);policy=load_policy(kind,a.run/'best_policy')
errors=[];angle=[];quatnorm=[];latencies=[];saturated=0;count=0;chunkerrors=[];validchunks=0
with torch.inference_mode():
 for batch in loader:
  batch={k:v.cuda() for k,v in batch.items()};torch.cuda.synchronize();t=time.monotonic();pred=predict(policy,kind,batch);torch.cuda.synchronize();latencies.append(time.monotonic()-t)
  pp=pred.cpu().numpy()*stats['action_scale']+stats['action_center'];truth=batch['action'][:,:4].cpu().numpy()*stats['action_scale']+stats['action_center'];pad=batch['action_is_pad'][:,:4].cpu().numpy()
  errors.append(abs(pp[:,0]-truth[:,0]));chunkerrors.extend(abs(pp-truth)[~pad].tolist());validchunks+=int((~pad).sum());saturated+=int((abs(pred)>=.999999).sum());count+=pred.numel()
  if rep=='ee_pose':
   norms=np.linalg.norm(pp[:,0,3:7],axis=1);quatnorm.extend(norms.tolist());safe=np.maximum(norms,1e-12)
   # A zero quaternion is an invalid target, never normalized into a valid score.
   valid=norms>.5;angle.extend((2*np.arccos(np.clip(np.abs(np.sum((pp[valid,0,3:7]/safe[valid,None])*truth[valid,0,3:7],axis=1)),0,1))).tolist())
e=np.concatenate(errors);ce=np.array(chunkerrors)
report={'status':'completed','kind':kind,'representation':rep,'episodes':40,'test_seeds':m['test_seeds'],'windows':len(e),'fixed_noise_seed':4242,'anchor_selection':'1024 deterministic evenly spaced held-out windows; all40 episodes','first_step_mae_physical':e.mean(0).tolist(),'first_step_rmse_physical':np.sqrt((e**2).mean(0)).tolist(),'first_step_train_std_scaled_l1':float((e/stats['action_std']).mean()),'executed4step_valid_mae_physical':ce.mean(0).tolist(),'valid_executed_action_points':validchunks,'padding_excluded':True,'saturated_coordinate_fraction':saturated/count,'batch16_inference_wall_s_median':float(np.median(latencies)),'batch16_inference_not_control_latency':True,'units':['rad/s']*6+['m'] if rep=='velocity' else ['rad']*6+['m'] if rep=='joint_position' else ['m']*3+['unitless_quaternion_xyzw']*4+['m'],'best_checkpoint_sha256':sha(a.run/'best_policy/model.safetensors'),'manifest_sha256':sha(O/'FROZEN_MANIFEST.json'),'checkpoint_selection':'validation only; test never used to select or tune','training_loss_not_compared_between_families':True}
if rep=='ee_pose':report.update(orientation_geodesic_mean_deg=float(np.degrees(angle).mean()) if angle else None,orientation_geodesic_p95_deg=float(np.percentile(np.degrees(angle),95)) if angle else None,invalid_quaternion_count=sum(x<=.5 for x in quatnorm),predicted_quaternion_norm_min=min(quatnorm),predicted_quaternion_norm_max=max(quatnorm))
save(a.run/'TEST_METRICS.json',report);print(json.dumps(report,indent=2))
