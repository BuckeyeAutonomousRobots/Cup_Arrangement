"""Freeze accepted v3r3 data and create one shared, non-destructive cycle-excluded cache."""
from pathlib import Path
import json,hashlib,datetime,os,fcntl,shutil,subprocess
import numpy as np,cv2
R=Path(__file__).resolve().parents[2];S=R/'ros_backend1.1/runtime/fixed_angle_constant_speed_position_only_v3r3_20261025';P=S/'control/parallel-segment-07';O=R/'models/six_model_v3r3_400_20261028_v2'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));os.replace(t,p)
from transition_guard import completed
def guard():
 completed()

def main():
 cv2.setNumThreads(1)
 with open('/tmp/baseline_collection_batch.lock','a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guard()
  x=json.loads((P/'progress.json').read_text());assert len(x['accepted'])==400 and not x['active_claims'] and not Path('/proc/'+str(x['pid'])).exists()
  assert not (O/'FROZEN_MANIFEST.json').exists()
  assert not (O/'cache').exists() or not any((O/'cache').iterdir()),'Existing prepared cache requires review'
  for w in ['a','b']:assert not json.loads(subprocess.check_output(['docker','inspect','cup_bench_v3r3_'+w+'_20261025']))[0]['State']['Running']
  seeds=sorted(map(int,x['accepted']));order=np.random.default_rng(42).permutation(seeds).tolist();splits={s:('train' if i<320 else 'validation' if i<360 else 'test') for i,s in enumerate(order)}
  manifest={'created':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_progress':str(P/'progress.json'),'source_progress_sha256':sha(P/'progress.json'),'accepted_count':400,'excluded':'All historical failed/interrupted attempts, benchmark/evaluation recordings and older dataset versions; see COLLECTION_COMPLETE.json','view':'cycle_excluded','split_seed':42,'train_seeds':sorted(s for s in seeds if splits[s]=='train'),'validation_seeds':sorted(s for s in seeds if splits[s]=='validation'),'test_seeds':sorted(s for s in seeds if splits[s]=='test'),'episodes':[]}
  manifest['rollout_seeds']=manifest['test_seeds'][:10]
  manifest['collection_audit_sha256']=sha(O/'COLLECTION_COMPLETE.json')
  manifest['collection_audit']=json.loads((O/'COLLECTION_COMPLETE.json').read_text())
  save(O/'SPLIT.json', {k:manifest[k] for k in ['split_seed','train_seeds','validation_seeds','test_seeds']})
  (O/'cache').mkdir(exist_ok=True)
  for j,seed in enumerate(seeds):
   guard();a=x['accepted'][str(seed)];ep=Path(a['path']);raw=json.loads((ep/'RAW_MANIFEST.json').read_text());raw={z['path']:z for z in raw};assert sha(ep/'RAW_MANIFEST.json')==a['manifest_sha256']
   v=json.loads((ep/'DATASET_VIEWS.json').read_text());schema=json.loads((ep/'ACTION_SCHEMA_VALIDATION.json').read_text());assert v['status']==schema['status']=='passed'
   clock=json.loads((ep/'CLOCK_READINESS_VALIDATION.json').read_text());assert clock['status']=='passed' and not clock['synthetic_timestamps_used']
   for rel,digest in v['source_hashes'].items():assert sha(ep/rel)==digest
   for rel in ['actions_all_10hz.jsonl','summary.json','action_schema.json']:assert sha(ep/rel)==raw[rel]['sha256']
   rows=[json.loads(l) for l in (ep/'actions_all_10hz.jsonl').open()];view=v['views']['cycle_excluded'];selected=[z['source_row_index_zero_based'] for z in view['samples']]
   images=[];states=[];vel=[];pos=[];ee=[];ticks=[];stamps=[];srcidx=[];segments=[];segment=[];last=None;lastquat=None
   for idx in selected:
    z=rows[idx];assert z['valid'] and z['max_causal_age_s']<=.100000001 and z['reference_horizon_s']==0 and z['native_joint_position_command'] is None
    assert z['stamp_sim_s']>=view['time_origin_sim_s'];e=z['ee_reference'];assert e['frame']=='world' and e['link']=='right_tool0'
    if last is None or z['tick']!=last+1:
     if segment:segments.append(segment)
     segment=[]
    segment.append(len(images));last=z['tick']
    frame=ep/z['image_file'];assert frame.stat().st_size==raw[z['image_file']]['bytes']
    assert sha(frame)==raw[z['image_file']]['sha256']
    im=cv2.imread(str(frame));assert im is not None and im.shape==(480,640,3)
    images.append(cv2.cvtColor(cv2.resize(im,(320,240),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB))
    g=z['commanded_common_gripper_target_m'];q=np.array(e['quaternion_xyzw'],np.float64);assert abs(np.linalg.norm(q)-1)<1e-6
    # Deterministic equivalent quaternion hemisphere; continuity is verified.
    if q[np.argmax(np.abs(q))]<0:q=-q
    if lastquat is not None:assert np.dot(q,lastquat)>0
    lastquat=q
    states.append(z['measured_state']['joint_positions']);vel.append(z['commanded_joint_velocity_rad_s']+[g]);pos.append(z['joint_position_reference_rad']+[g]);ee.append(e['position_m']+q.tolist()+[g]);ticks.append(z['tick']);stamps.append(z['stamp_sim_s']);srcidx.append(idx)
   if segment:segments.append(segment)
   assert len(images)>50
   c=O/'cache'/f'seed-{seed:03d}';c.mkdir()
   arrays={'images':np.asarray(images,np.uint8),'states':np.asarray(states,np.float32),'velocity':np.asarray(vel,np.float32),'joint_position':np.asarray(pos,np.float32),'ee_pose':np.asarray(ee,np.float32),'ticks':np.asarray(ticks,np.int64),'stamp_sim_s':np.asarray(stamps,np.float64),'source_rows':np.asarray(srcidx,np.int64)}
   for n,arr in arrays.items():assert np.isfinite(arr).all();np.save(c/(n+'.npy'),arr)
   meta={'seed':seed,'split':splits[seed],'episode':str(ep),'cache':str(c),'samples':len(images),'segments':segments,'view_time_origin_sim_s':view['time_origin_sim_s'],'initial_poses':a['initial_poses'],'raw_manifest_sha256':a['manifest_sha256'],'source_sha256':{n:sha(ep/n) for n in ['ACCEPTANCE.json','DATASET_VIEWS.json','actions_all_10hz.jsonl','action_schema.json','summary.json']},'cache_sha256':{n:sha(c/(n+'.npy')) for n in arrays},'joint_order':schema['joint_order']}
   save(c/'metadata.json',meta);manifest['episodes'].append(meta)
   save(O/'preparation_status.json',{'status':'preparing','episodes_done':j+1,'total':400,'last_seed':seed,'samples_done':sum(e['samples'] for e in manifest['episodes']),'free_GiB':shutil.disk_usage(O).free/2**30})
   if (j+1)%10==0:print(json.dumps({'prepared':j+1,'seed':seed}),flush=True)
  manifest.update(frequency_hz=10,image_preprocessing='640x480 RGB lossless source ->320x240 INTER_AREA; no crop; ImageNet channel mean/std applied identically during training/inference',state_dim=8,action_representations={'velocity':'6 native published joint velocities rad/s + common gripper position m','joint_position':'6 IK-derived planned joint references rad at horizon0; NOT native position actuator commands + common gripper target m','ee_pose':'world right_tool0 XYZ metres/quaternion xyzw + common gripper target m; quaternion sign canonicalized without changing rotation'},raw_preserved=True)
  save(O/'FROZEN_MANIFEST.json',manifest);save(O/'preparation_status.json',{'status':'completed','episodes_done':400,'samples':sum(e['samples'] for e in manifest['episodes']),'manifest_sha256':sha(O/'FROZEN_MANIFEST.json'),'free_GiB':shutil.disk_usage(O).free/2**30});print((O/'preparation_status.json').read_text(),flush=True)
if __name__=='__main__':main()
