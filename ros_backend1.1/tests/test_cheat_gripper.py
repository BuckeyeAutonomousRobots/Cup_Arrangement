from types import SimpleNamespace
from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from cup_task import GroundTruthController


@pytest.mark.parametrize('positions,target,expected', [
    ([0.,0.],-.01,-.0005),
    ([-.002,-.012],-.01,-.0025),
    ([-.01,-.01],-.01,-.01),
    ([-.002,-.025],0.,0.),
])
def test_shared_aperture_backs_off_to_blocked_jaw(positions,target,expected):
    controller = GroundTruthController.__new__(GroundTruthController)
    controller.q = dict(zip(['right_robotiq_hande_'+s+'_finger_joint' for s in ('left','right')],positions))
    controller.gripper_target=target
    controller.Msg=SimpleNamespace
    output=[]
    controller.grip=SimpleNamespace(publish=output.append)
    controller.hold_gripper()
    assert output[0].data == pytest.approx([expected,expected])
