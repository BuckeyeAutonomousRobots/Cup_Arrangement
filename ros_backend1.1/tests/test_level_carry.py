import ast
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from cup_task import GroundTruthController as C

@pytest.mark.parametrize('scale',[.3,1.])
def test_every_carry_move_uses_level_controller(scale):
    c=C.__new__(C);calls=[]
    c.move_level=lambda *args:calls.append(args)
    c.move([.3,0.,.4],.7,carry=True,speed_scale=scale,position_tolerance=.002)
    assert calls==[([.3,0.,.4],.7,scale,.002)]

def test_level_controller_retains_safety_and_scaled_commands():
    tree=ast.parse(Path(sys.modules[C.__module__].__file__).read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='move_level')
    text=ast.unparse(method)
    for required in ['check_physical_hold(q)','orientation_error(q, yaw)','cartesian_jacobian(q)',
                     'self.command(scale * velocity)','math.radians(1.0)']:
        assert required in text
