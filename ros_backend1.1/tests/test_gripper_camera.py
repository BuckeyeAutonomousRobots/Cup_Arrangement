"""Camera mount/schema regression without commanding any robot motion."""
import pathlib
import subprocess
import xml.etree.ElementTree as ET

ROOT = pathlib.Path(__file__).resolve().parents[1]


def robot(enabled):
    return ET.fromstring(subprocess.check_output([
        'xacro', str(ROOT / 'src/ur_hande_description/urdf/ur_hande.urdf.xacro'),
        'ur_type:=ur5e', 'name:=right_ur5e', 'tf_prefix:=right_', 'sim_ignition:=true',
        f'enable_gripper_camera:={str(enabled).lower()}',
    ], text=True))


def test_camera_opt_in():
    assert not robot(False).findall('.//sensor')


def test_camera_mount_and_schema():
    root = robot(True)
    sensor = root.find('.//sensor')
    assert sensor.get('type') == 'camera'
    assert sensor.findtext('update_rate') == '30'
    assert sensor.findtext('camera/image/width') == '640'
    assert sensor.findtext('camera/image/height') == '480'
    mount = root.find("joint[@name='right_gripper_camera_joint']")
    assert mount.get('type') == 'fixed'
    assert mount.find('parent').get('link') == 'right_tool0'
    assert mount.find('origin').get('xyz') == '0 0.065 0.055'
    assert mount.find('origin').get('rpy') == '0 -1.57079632679 1.57079632679'
    housing = root.find("link[@name='right_gripper_camera_link']")
    assert len(housing.findall('visual')) == 2
    assert housing.find('collision') is None
    assert housing.find('inertial') is None
    optical = root.find("joint[@name='right_gripper_camera_optical_joint']")
    assert optical.get('type') == 'fixed'
    assert optical.find('child').get('link') == 'right_gripper_camera_optical_frame'
