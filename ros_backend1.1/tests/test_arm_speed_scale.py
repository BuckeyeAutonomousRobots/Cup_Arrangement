import ast
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from cup_task import GroundTruthController as C

@pytest.mark.parametrize('scale',[.5,1.,.1])
def test_valid_scale(scale):
    c=C.__new__(C);c.task={'debug':{'arm_speed_scale':scale}}
    assert c.arm_speed_scale()==scale

@pytest.mark.parametrize('scale',[0,-1,1.01,float('nan'),float('inf')])
def test_bad_scale(scale):
    c=C.__new__(C);c.task={'debug':{'arm_speed_scale':scale}}
    with pytest.raises(ValueError):c.arm_speed_scale()

def test_default_and_both_motion_paths_scaled():
    c=C.__new__(C);c.task={'debug':{}}
    assert c.arm_speed_scale()==1.
    tree=ast.parse(Path(sys.modules[C.__module__].__file__).read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
    for name in ['move','descend_centered']:
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
        calls=[n for n in ast.walk(method) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='command']
        assert any('global_scale' in ast.unparse(n.args[0]) for n in calls)
