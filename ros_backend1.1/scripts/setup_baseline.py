"""Create isolated baseline profiles and solve a ready pose, without ROS motion."""
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import yaml
from cup_task import GroundTruthController

root=Path(__file__).resolve().parents[1]
c=GroundTruthController.__new__(GroundTruthController)
c.np=np
c.spawn=[0,-.6,0,0,0,0]
urdf=ET.parse('/tmp/right_ur5e_hande_dual.urdf')
by_child={j.find('child').get('link'):j for j in urdf.findall('joint')}
c.chain=[]
child='right_tool0'
while child in by_child:
    joint=by_child[child];c.chain.insert(0,joint);child=joint.find('parent').get('link')
q=c.solve_ik(np.array([.25,0,.50]),math.pi/2,np.array([1.,-1.1,1.2,-1.6,-1.57,-2.2]))
assert np.linalg.norm(c.fk(q)[:3,3]-[.25,0,.50])<.001
scene=yaml.safe_load((root/'profiles/scenes/dual_arm_cup_arrangement/scene.yaml').read_text())
robot=yaml.safe_load((root/'profiles/robots/ur5e_hande_dual/robot.yaml').read_text())
workspace=yaml.safe_load((root/'profiles/workspaces/dual_arm_tabletop/workspace.yaml').read_text())
task=yaml.safe_load((root/'profiles/tasks/cup_arrangement/task.yaml').read_text())
task['debug'].update(release_clearance_m=0.005,arm_speed_scale=0.7,approach_speed_scale=0.3,initial_lift_speed_scale=0.3,gripper_closing_speed_m_s=0.005,gripper_timeout_sec=40,open_gripper_position_m=-0.001)
for r in robot['robots']:
    if r['arm_id']=='right_arm':r['home_joint_positions']=q.tolist()
workspace['table'].update(color_rgba=[1,1,1,1])
for obj in task['objects']:
    if obj['id']=='Sync_EspressoCup':obj['radius']=0.0225
    if obj['id']=='Sync_Saucer':obj['color_rgba']=[0.45,0.72,0.95,1.0]
    if obj['id']=='Sync_TaskPlatform':obj.update(color_rgba=[1,1,1,1])
scene.update(scene_profile='baseline_v1',workspace_profile='../../workspaces/baseline_v1/workspace.yaml',robot_profile='../../robots/baseline_v1/robot.yaml',task_profile='../../tasks/baseline_v1/task.yaml',camera_hfov=math.radians(120))
for kind,name,data in [('scenes','scene',scene),('robots','robot',robot),('workspaces','workspace',workspace),('tasks','task',task)]:
    path=root/'profiles'/kind/'baseline_v1'/f'{name}.yaml';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(yaml.safe_dump(data,sort_keys=False))
print(json.dumps(dict(tool_xyz=[.25,0,.5],tool_rpy=[math.pi,0,math.pi/2],right_home=q.tolist(),camera_horizontal_fov_degrees=120),indent=2))
