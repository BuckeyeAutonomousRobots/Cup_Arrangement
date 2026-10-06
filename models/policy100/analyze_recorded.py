"""Offline outcome/timing summary from a completed robot recording."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,'/workspace/scripts')
from analyze_camera_latency import analyze

p=argparse.ArgumentParser();p.add_argument('episode',type=Path);a=p.parse_args();ep=a.episode
s=json.loads((ep/'summary.json').read_text())
rows=[json.loads(line) for line in (ep/'policy_actions.jsonl').open()]
result={'policy':s['policy'],'actions':len(rows),'note':'Measured joints and recorded commands; closure does not by itself prove contact grasp.'}
if rows:
    states=np.array([r['state'] for r in rows]);raw=np.array([r['raw'] for r in rows]);applied=np.array([r['applied'] for r in rows])
    result.update(wrist_net_deg=float(np.degrees(states[-1,5]-s['initial_q'][5])),
        finger_target_min_m=float(applied[:,6].min()),finger_target_final_m=float(applied[-1,6]),
        measured_finger_min_m=states[:,6:].min(0).tolist(),measured_finger_final_m=states[-1,6:].tolist(),
        clamp_counts_per_action=np.sum(raw!=applied,axis=0).tolist(),raw_peak_abs=np.max(np.abs(raw),axis=0).tolist())
last=[]
for line in (ep/'robot_data.jsonl').open():
    r=json.loads(line)
    if r['topic']=='/right_joint_group_velocity_controller/commands':
        last.append(r['message']['data']);last=last[-10:]
result['last_velocity_messages']=last
(ep/'outcome_analysis.json').write_text(json.dumps(result,indent=2))
(ep/'latency_analysis.json').write_text(json.dumps(analyze(ep),indent=2))
print(json.dumps(result,indent=2))
