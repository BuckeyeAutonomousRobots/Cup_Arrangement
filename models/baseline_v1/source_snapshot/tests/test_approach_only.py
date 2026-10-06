import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from cup_task import GroundTruthController as C

@pytest.mark.parametrize('close_fails',[False,True])
def test_diagnostic_never_lifts_and_reopens(close_fails):
    c=C.__new__(C);c.np=np
    c.task={'debug':{'tool_to_cup_m':.17,'clearance_m':.16}}
    c.poses={'Sync_EspressoCup':[.3,0,.125,0],'Sync_Saucer':[.2,0,.104,0]}
    c.names=['j'+str(i) for i in range(6)];c.q=dict.fromkeys(c.names,0.)
    c.solve_ik=lambda xyz,yaw,q:q;c.command=lambda q:None
    c.cup_in_tool=lambda q:np.array([0.,0.,.17]);c.fk=lambda q:np.eye(4)
    c.finger_positions=lambda:[0.,0.]
    moves=[];grips=[]
    c.move=lambda xyz,*a,**kw:moves.append((xyz,kw))
    def grip(q):
        grips.append(q)
        if q<0 and close_fails:raise RuntimeError('blocked')
    c.set_gripper=grip
    if close_fails:
        with pytest.raises(RuntimeError,match='blocked'):c.run(approach_only=True)
    else:c.run(approach_only=True)
    assert len(moves)==2
    assert not any(kw.get('carry') for xyz,kw in moves)
    assert grips==[0.,-.01,0.]
