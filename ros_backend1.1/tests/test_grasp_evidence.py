from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from cup_task import GroundTruthController as C

@pytest.mark.parametrize('fingers,diameter,ready',[
    ([0.,0.],.045,False),
    ([-.0005,0.],.045,False),
    ([-.0005,0.],.0486,False),
    ([-.0045,-.0045],.045,True),
    ([-.0027,-.0027],.0486,True),
    ([-.01,-.01],.045,False),
    ([-.009,0.],.045,False),
])
def test_actual_aperture_required(fingers,diameter,ready):
    assert C.aperture_evidence(fingers,diameter,.054)['ready'] is ready

def test_invalid_geometry():
    for fingers,d,g in [([0.,float('nan')],.045,.054),([0.,0.],.06,.054),([0.],.045,.054)]:
        with pytest.raises(ValueError):C.aperture_evidence(fingers,d,g)

def test_no_full_transport_when_cup_does_not_follow():
    import numpy as np
    c=C.__new__(C);c.np=np
    c.task={'debug':{'tool_to_cup_m':.17,'clearance_m':.16}}
    c.poses={'Sync_EspressoCup':[.3,0,.125,0],'Sync_Saucer':[.2,0,.104,0]}
    c.names=['j'+str(i) for i in range(6)];c.q=dict.fromkeys(c.names,0.)
    c.solve_ik=lambda xyz,yaw,q:q
    c.command=lambda q:None;c.set_gripper=lambda q:None
    c.cup_in_tool=lambda q:np.array([0.,0.,.17])
    moves=[]
    c.move=lambda xyz,*args,**kwargs:moves.append((xyz,kwargs))
    c.descend_centered=lambda xyz,yaw:moves.append((xyz,{'cartesian':True}))
    c.settle=lambda *args:None;c.check_physical_hold=lambda q:None
    with pytest.raises(RuntimeError,match='did not follow'):c.run()
    carry=[xyz for xyz,kw in moves if kw.get('carry')]
    assert len(carry)==1
    assert carry[0][2]==pytest.approx(.125+.17+.01)
