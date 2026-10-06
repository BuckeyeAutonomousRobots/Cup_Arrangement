from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from cup_task import GroundTruthController as C

@pytest.mark.parametrize('elapsed,expected', [(0,0),(.1,-.0005),(1,-.005),(2,-.01),(10,-.01),(-1,0)])
def test_closing_target_rate_and_limits(elapsed,expected):
    assert C.closing_target(0.,-.01,elapsed,.005)==pytest.approx(expected)

def test_legacy_closure_and_invalid_speed():
    assert C.closing_target(0.,-.01,0.,0.)==-.01
    for speed in (-1.,float('nan'),float('inf')):
        with pytest.raises(ValueError): C.closing_target(0.,-.01,1.,speed)

@pytest.mark.parametrize('scale', [0, -1, 1.1, float('nan')])
def test_invalid_motion_scale(scale):
    with pytest.raises(ValueError): C.__new__(C).move([0,0,0],speed_scale=scale)
