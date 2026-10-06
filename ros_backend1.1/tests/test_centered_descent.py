import sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from cup_task import GroundTruthController as C

def test_world_jacobian_translation_and_rotation():
    c=C.__new__(C);c.np=np
    def fk(q):
        tf=np.eye(4);tf[:3,3]=q[:3]
        tf[:3,:3]=Rotation.from_rotvec(q[3:]).as_matrix()
        return tf
    c.fk=fk
    assert np.allclose(c.cartesian_jacobian(np.zeros(6)),np.eye(6),atol=1e-8)

def test_orientation_uses_downward_tool():
    c=C.__new__(C);c.np=np
    tf=np.eye(4);tf[:3,:3]=Rotation.from_euler('xyz',[np.pi,0,.7]).as_matrix()
    c.fk=lambda q:tf
    assert c.orientation_error(np.zeros(6),.7)[1]<1e-12
    assert c.orientation_error(np.zeros(6),.8)[1]==pytest.approx(.1)

@pytest.mark.parametrize('lateral,angle,drift,message',[
    (.004,0.,0.,'Unsafe rim'),(0.,.01,0.,'Unsafe rim'),
    (0.,0.,.002,'Cup disturbed'),
])
def test_descent_stops_before_unsafe_command(lateral,angle,drift,message):
    c=C.__new__(C);c.np=np;c.task={'debug':{'waypoint_timeout_sec':1},'objects':[{'id':'Sync_EspressoCup','height':.05}]}
    c.poses={'Sync_EspressoCup':[0.,0.,.125]};c.tilt={'Sync_EspressoCup':0.}
    c.last_joint=time.monotonic();c.pose_received={'Sync_EspressoCup':time.monotonic()}
    c.sim_time=0.;c.names=['q']*6;c.q={'q':0.};c.node=None
    def spin(*a,**kw):c.poses['Sync_EspressoCup'][0]=drift
    c.ros=SimpleNamespace(spin_once=spin);c.check_publishers=lambda:None;c.hold_gripper=lambda:None
    tf=np.eye(4);tf[2,3]=.32;c.fk=lambda q:tf
    c.orientation_error=lambda q,y:(np.zeros(3),angle)
    c.cup_in_tool=lambda q:np.array([lateral,0.,.195])
    c.finger_positions=lambda:[-.001,-.001];c.open_finger_gap=.054;c.cup_diameter=.045
    commands=[];c.command=lambda v:commands.append(v)
    with pytest.raises(RuntimeError,match=message):c.descend_centered([0.,0.,.295],0.)
    assert commands==[]
