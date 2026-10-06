"""Offline posture regression against the running simulation's URDF."""
import json
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from cup_task import GroundTruthController, active_task
from cup_layout import sample_layout

c = GroundTruthController.__new__(GroundTruthController)
c.np = np
c.spawn = [0, -.60, 0, 0, 0, 0]
urdf = ET.parse('/tmp/right_ur5e_hande_dual.urdf')
by_child = {j.find('child').get('link'): j for j in urdf.findall('joint')}
c.chain = []
child = 'right_tool0'
while child in by_child:
    j = by_child[child]
    c.chain.insert(0, j)
    child = j.find('parent').get('link')
task = active_task()
count = 0
for seed in range(3):
    layout = sample_layout(task, seed)
    cup, saucer = layout['Sync_EspressoCup'], layout['Sync_Saucer']
    q = np.array([0,-1.57,1.57,0,1.57,0])
    for pose, dz in [(cup,.34),(cup,.18),(saucer,.369),(saucer,.209)]:
        xyz = np.array(pose[:3])+[0,0,dz]
        q = c.solve_ik(xyz,cup[3],q)
        tf = c.fk(q)
        assert np.linalg.norm(tf[:3,3]-xyz)<.001
        desired = Rotation.from_euler('xyz',[np.pi,0,cup[3]]).as_matrix()
        assert np.linalg.norm(Rotation.from_matrix(desired@tf[:3,:3].T).as_rotvec())<.004
        assert q[1]<-.15 and q[2]>.15 and q[2]<=np.pi
        count+=1
        print(json.dumps(dict(seed=seed,xyz=xyz.tolist(),q=q.tolist())))
try:
    c.solve_ik(np.array([10,10,10]),0,q)
except RuntimeError:
    pass
else:
    raise AssertionError('Unreachable target accepted')
print(f'PASS: {count} waypoints, orientation and posture constraints; unreachable target rejected')
