import ast
import inspect
from pathlib import Path
import sys
import time
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from cup_task import GroundTruthController

def test_manipulation_has_no_pose_service_calls():
    tree=ast.parse(inspect.getsource(GroundTruthController))
    calls={n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}
    assert not {'set_pose','ign_service'} & calls

def test_slip_and_stale_pose_rejected():
    c=GroundTruthController.__new__(GroundTruthController)
    c.np=np
    c.pose_received={'Sync_EspressoCup':time.monotonic()}
    c.grasp_relative=np.array([0,0,.17])
    c.cup_in_tool=lambda q:np.array([0,0,.175])
    c.check_physical_hold(None)
    c.cup_in_tool=lambda q:np.array([0,0,.22])
    with pytest.raises(RuntimeError,match='slipped'):c.check_physical_hold(None)
    c.pose_received['Sync_EspressoCup']=0
    with pytest.raises(RuntimeError,match='Stale'):c.check_physical_hold(None)
