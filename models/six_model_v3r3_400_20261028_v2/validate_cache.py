"""Read-only shared-cache parity and split validation before any training."""
from common import *
import cv2,shutil
from transition_guard import completed
completed(full=True)
m=json.loads((O/'FROZEN_MANIFEST.json').read_text());assert len(m['episodes'])==400 and len({e['seed'] for e in m['episodes']})==400
sets={k:set(m[k+'_seeds']) for k in ['train','validation','test']};assert [len(sets[k]) for k in sets]==[320,40,40]
assert not sets['train']&sets['validation'] and not sets['train']&sets['test'] and not sets['validation']&sets['test']
assert len(m['rollout_seeds'])==10 and set(m['rollout_seeds'])<=sets['test']
assert sha(Path(m['source_progress']))==m['source_progress_sha256']
count=0;anchors=0
for e in m['episodes']:
 c=Path(e['cache']);ep=Path(e['episode']);assert e['seed'] in sets[e['split']]
 for n,h in e['cache_sha256'].items():assert sha(c/(n+'.npy'))==h,(e['seed'],n)
 for n,h in e['source_sha256'].items():assert sha(ep/n)==h
 rows=[json.loads(l) for l in (ep/'actions_all_10hz.jsonl').open()];ids=np.load(c/'source_rows.npy');ticks=np.load(c/'ticks.npy');states=np.load(c/'states.npy');vel=np.load(c/'velocity.npy');pos=np.load(c/'joint_position.npy');ee=np.load(c/'ee_pose.npy');ims=np.load(c/'images.npy',mmap_mode='r')
 assert len(ids)==e['samples']==len(ims) and sorted(sum(e['segments'],[]))==list(range(len(ids)))
 assert all(all(ticks[b]==ticks[a]+1 for a,b in zip(seg,seg[1:])) for seg in e['segments'])
 for i in np.linspace(0,len(ids)-1,8,dtype=int):
  z=rows[ids[i]];g=z['commanded_common_gripper_target_m'];assert z['valid'] and z['stamp_sim_s']>=e['view_time_origin_sim_s']
  np.testing.assert_allclose(states[i],z['measured_state']['joint_positions'],atol=2e-7)
  np.testing.assert_allclose(vel[i],z['commanded_joint_velocity_rad_s']+[g],atol=2e-7)
  np.testing.assert_allclose(pos[i],z['joint_position_reference_rad']+[g],atol=2e-7)
  np.testing.assert_allclose(ee[i,:3],z['ee_reference']['position_m'],atol=1e-7)
  assert abs(abs(np.dot(ee[i,3:7],z['ee_reference']['quaternion_xyzw']))-1)<1e-6
  im=cv2.imread(str(ep/z['image_file']));im=cv2.cvtColor(cv2.resize(im,(320,240),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB);np.testing.assert_array_equal(im,ims[i]);anchors+=1
 count+=len(ids)
out={'status':'passed','utc':now(),'episodes':400,'samples':count,'pixel_and_action_anchors':anchors,'all_cache_hashes_verified':True,'source_hashes_verified':True,'episode_disjoint_split_counts':[320,40,40],'rollout_seeds':m['rollout_seeds'],'all_three_representations_verified':True,'canonical_quaternion_rotation_equivalence':True,'raw_unchanged':True,'manifest_sha256':sha(O/'FROZEN_MANIFEST.json'),'free_GiB':shutil.disk_usage(O).free/2**30}
save(O/'CACHE_VALIDATION.json',out);print(json.dumps(out,indent=2))
