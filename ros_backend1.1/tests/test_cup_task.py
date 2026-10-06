import copy
import importlib.util
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'simulation/tools'))
sys.path.insert(0, str(ROOT/'scripts'))
from cup_layout import sample_layout
from cup_task import ign_service
from generate_dual_arm_world_from_profiles import generate, write_unity_task_json

TASK = yaml.safe_load((ROOT/'profiles/tasks/cup_arrangement/task.yaml').read_text())
SCENE = ROOT/'profiles/scenes/dual_arm_cup_arrangement/scene.yaml'


def test_bounds_separation_reproducibility():
    for seed in range(1000):
        layout = sample_layout(TASK, seed)
        assert layout == sample_layout(TASK, seed)
        c, s = layout.values()
        for x, y, z, yaw in layout.values():
            assert 0.14 <= x <= 0.36
            assert -0.14 <= y <= 0.14
            assert -math.pi <= yaw <= math.pi
            assert z > 0.1
        assert math.hypot(c[0]-s[0], c[1]-s[1]) >= 0.14
    assert sample_layout(TASK, 1) != sample_layout(TASK, 2)


def test_impossible_bounds_fail():
    task = copy.deepcopy(TASK)
    task['randomization'].update(x_bounds=[0, 0], y_bounds=[0, 0], max_attempts=3)
    with pytest.raises(ValueError): sample_layout(task, 0)
    task['randomization']['x_bounds'] = [float('nan'), 1]
    with pytest.raises(ValueError): sample_layout(task, 0)


def test_world_has_only_requested_task_models(tmp_path):
    world_path = tmp_path/'world.sdf'
    generate(SCENE, world_path)
    world = ET.parse(world_path).find('world')
    models = {m.get('name'): m for m in world.findall('model')}
    assert set(models) == {'table', 'Sync_TaskPlatform', 'Sync_EspressoCup', 'Sync_Saucer'}
    assert len(models['Sync_EspressoCup'].findall('.//collision')) == 20
    for name, pose in sample_layout(TASK, 0).items():
        actual = list(map(float, models[name].findtext('pose').split()))
        assert actual[:3] == pytest.approx(pose[:3], abs=1e-6)
        assert actual[-1] == pytest.approx(pose[-1], abs=1e-5)
    # Manifest uses the same sampled layout and coordinate conversion.
    import json
    manifest = tmp_path/'task.json'
    write_unity_task_json(SCENE, manifest)
    data = json.loads(manifest.read_text())
    cup = next(o for o in data['objects'] if o['id'] == 'Sync_EspressoCup')
    c = sample_layout(TASK, 0)['Sync_EspressoCup']
    assert cup['localPosition']['z']+data['taskGroupLocalPosition']['z'] == pytest.approx(c[0])


def test_false_service_reply_is_failure(monkeypatch):
    import subprocess
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess([], 0, 'data: false', ''))
    with pytest.raises(RuntimeError): ign_service('set_pose', 'ignition.msgs.Pose', '')
