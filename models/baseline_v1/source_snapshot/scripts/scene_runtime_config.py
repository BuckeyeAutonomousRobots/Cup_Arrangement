"""Resolve active scene home joints and camera calibration for startup/reset."""
from pathlib import Path
import math
import yaml
root=Path(__file__).resolve().parents[1]
path=root/(root/'profiles/active_scene_profile.txt').read_text().strip()
scene=yaml.safe_load(path.read_text())
robot=yaml.safe_load((path.parent/scene['robot_profile']).read_text())
names=['shoulder_pan_joint','shoulder_lift_joint','elbow_joint','wrist_1_joint','wrist_2_joint','wrist_3_joint']
for r in robot['robots']:
    side=r['arm_id'].split('_')[0]
    q=r['home_joint_positions']
    if len(q)!=6 or not all(math.isfinite(float(x)) for x in q):raise ValueError('Invalid home joints')
    Path(f'/tmp/scene_{side}_initial.yaml').write_text(yaml.safe_dump(dict(zip(names,q))))
    Path(f'/tmp/scene_{side}_reset.yaml').write_text(yaml.safe_dump({f'/{side}_arm/reset_manager':{'ros__parameters':{'home_joint_positions':q,'home_timeout_sec':30.0}}}))
fov=float(scene.get('camera_hfov',math.pi/2))
if not 0<fov<math.pi:raise ValueError('Invalid camera field of view')
Path('/tmp/scene_camera_hfov').write_text(str(fov))
