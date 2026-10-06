"""Offline IK branch inspection; no ROS initialization or motion."""
import json
import math
import xml.etree.ElementTree as ET
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from cup_task import GroundTruthController

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
joints = [j for j in c.chain if j.get('type') == 'revolute']
lower = np.array([float(j.find('limit').get('lower')) for j in joints])
upper = np.array([float(j.find('limit').get('upper')) for j in joints])
yaw = -0.070594
# Recover exact requested yaw from the recorded target's FK.
old = np.array([.940123709694,.197821496504,-1.177389959324,-.591227864181,4.71238898059,-.701513817306])
desired = c.fk(old)[:3,:3]
for z in [.465, .305]:
    xyz = np.array([.325773,.0722272,z])
    def residual(q):
        a = c.fk(q)
        return np.r_[a[:3,3]-xyz,.25*Rotation.from_matrix(desired@a[:3,:3].T).as_rotvec()]
    for seed in [[.94,-1.3,1.3,-1.57,1.57,2.44], [.94,-1.3,1.3,-1.57,-1.57,-.70], old]:
        s = least_squares(residual,seed,bounds=(lower,upper),max_nfev=400)
        print(json.dumps(dict(z=z,q=s.x.tolist(),residual=float(np.linalg.norm(residual(s.x))))))
