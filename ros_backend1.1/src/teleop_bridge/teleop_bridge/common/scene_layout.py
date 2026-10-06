import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SceneObjectPose:
    name: str
    link_name: str
    pose_xyz: tuple[float, float, float]
    pose_xyzw: tuple[float, float, float, float]
    is_static: bool


def _normalize_quaternion(qx: float, qy: float, qz: float, qw: float) -> tuple[float, float, float, float]:
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm < 1e-9:
        return (0.0, 0.0, 0.0, 1.0)
    inv = 1.0 / norm
    return (qx * inv, qy * inv, qz * inv, qw * inv)


def _quat_from_rpy(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy
    return _normalize_quaternion(qx, qy, qz, qw)


def _parse_pose_text(pose_text: str | None) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    values = [float(v) for v in (pose_text or "").split()]
    if len(values) == 6:
        x, y, z, roll, pitch, yaw = values
        return (x, y, z), _quat_from_rpy(roll, pitch, yaw)
    if len(values) == 7:
        x, y, z, qx, qy, qz, qw = values
        return (x, y, z), _normalize_quaternion(qx, qy, qz, qw)
    raise ValueError(f"Unsupported SDF pose format: expected 6 or 7 values, got {len(values)}")


def _parse_bool(text: str | None) -> bool:
    return str(text or "").strip().lower() in {"1", "true", "yes"}


def load_scene_object_poses_from_sdf(
    sdf_path: str,
    *,
    name_prefix: str | None = None,
) -> dict[str, SceneObjectPose]:
    path = Path(sdf_path)
    if not path.exists():
        return {}

    root = ET.parse(path).getroot()
    world = root.find("world")
    if world is None:
        world = root.find(".//world")
    if world is None:
        return {}

    poses: dict[str, SceneObjectPose] = {}
    for model in world.findall("model"):
        model_name = str(model.get("name", "")).strip()
        if not model_name:
            continue
        if name_prefix and not model_name.startswith(name_prefix):
            continue

        pose_xyz, pose_xyzw = _parse_pose_text(model.findtext("pose", default="0 0 0 0 0 0"))
        is_static = _parse_bool(model.findtext("static", default="false"))
        first_link = model.find("link")
        link_name = str(first_link.get("name", "body_link")).strip() if first_link is not None else "body_link"
        poses[model_name] = SceneObjectPose(
            name=model_name,
            link_name=link_name,
            pose_xyz=pose_xyz,
            pose_xyzw=pose_xyzw,
            is_static=is_static,
        )

    return poses
