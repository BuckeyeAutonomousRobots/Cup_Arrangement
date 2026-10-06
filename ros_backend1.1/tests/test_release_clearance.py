import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from cup_task import GroundTruthController as C

@pytest.mark.parametrize('gap',[-.001,.02,float('nan'),float('inf')])
def test_unsafe_release_clearance_rejected_before_motion(gap):
    c=C.__new__(C);c.task={'debug':{'release_clearance_m':gap}}
    with pytest.raises(ValueError,match='Release clearance'):c.run()
