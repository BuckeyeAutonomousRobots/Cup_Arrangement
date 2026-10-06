#!/usr/bin/env python3
"""Generate the dual-arm Gazebo world from workspace + task profiles.

This keeps task objects modular: a scene profile selects one workspace profile
and one task profile. The task objects are described relative to a task group in
Unity workspace axes, then flattened into top-level Gazebo models for physics.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import yaml


DEFAULT_SCENE_PROFILE = "profiles/scenes/dual_arm_tabletop/scene.yaml"


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def resolve_profile_path(scene_profile_path: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (scene_profile_path.parent / path).resolve()


def vec3(value: Any, fallback=(0.0, 0.0, 0.0)) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return tuple(float(v) for v in fallback)
    return (float(value[0]), float(value[1]), float(value[2]))


def vec4(value: Any, fallback=(1.0, 1.0, 1.0, 1.0)) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return tuple(float(v) for v in fallback)
    return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))


def bool_value(value: Any, fallback: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if value is None:
        return fallback
    return bool(value)


def fmt_float(value: float) -> str:
    if abs(value) < 0.0000005:
        value = 0.0
    return f"{value:.6g}"


def fmt_values(values) -> str:
    return " ".join(fmt_float(float(v)) for v in values)


def unity_vector_to_gazebo(unity_xyz: tuple[float, float, float]) -> tuple[float, float, float]:
    ux, uy, uz = unity_xyz
    return (uz, -ux, uy)


def unity_size_to_gazebo(unity_xyz: tuple[float, float, float]) -> tuple[float, float, float]:
    ux, uy, uz = unity_xyz
    return (abs(uz), abs(ux), abs(uy))


def add_vec(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def rpy_deg_to_matrix(euler_xyz_deg: tuple[float, float, float]):
    roll, pitch, yaw = (math.radians(v) for v in euler_xyz_deg)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    # Unity-style XYZ input is used here only for task group organization. Current profiles keep it at zero.
    rx = ((1, 0, 0), (0, cr, -sr), (0, sr, cr))
    ry = ((cp, 0, sp), (0, 1, 0), (-sp, 0, cp))
    rz = ((cy, -sy, 0), (sy, cy, 0), (0, 0, 1))
    return mat_mul(rz, mat_mul(ry, rx))


def mat_mul(a, b):
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )


def mat_vec_mul(m, v):
    return tuple(sum(m[i][j] * v[j] for j in range(3)) for i in range(3))


def mat_transpose(m):
    return tuple(tuple(m[j][i] for j in range(3)) for i in range(3))


UNITY_TO_GAZEBO_ROT = (
    (0.0, 0.0, 1.0),
    (-1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
)


def clamp(value: float, low: float, high: float) -> float:
    if low > high:
        return 0.5 * (low + high)
    return min(max(value, low), high)


def matrix_to_rpy(m) -> tuple[float, float, float]:
    # Standard SDF/Gazebo convention: R = Rz(yaw) * Ry(pitch) * Rx(roll).
    if abs(m[2][0]) < 0.999999:
        pitch = math.asin(-m[2][0])
        roll = math.atan2(m[2][1], m[2][2])
        yaw = math.atan2(m[1][0], m[0][0])
    else:
        pitch = math.pi * 0.5 if m[2][0] <= -0.999999 else -math.pi * 0.5
        roll = math.atan2(-m[0][1], m[1][1])
        yaw = 0.0
    return roll, pitch, yaw


def unity_euler_to_gazebo_rpy(euler_xyz_deg: tuple[float, float, float]) -> tuple[float, float, float]:
    unity_rot = rpy_deg_to_matrix(euler_xyz_deg)
    gazebo_rot = mat_mul(UNITY_TO_GAZEBO_ROT, mat_mul(unity_rot, mat_transpose(UNITY_TO_GAZEBO_ROT)))
    return matrix_to_rpy(gazebo_rot)


def rgba_text(rgba: tuple[float, float, float, float]) -> str:
    return fmt_values(rgba)


def sub(parent: ET.Element, tag: str, text: str | None = None, attrib: dict[str, str] | None = None) -> ET.Element:
    elem = ET.SubElement(parent, tag, attrib or {})
    if text is not None:
        elem.text = text
    return elem


def add_common_plugins(world: ET.Element, ambient, background):
    physics = sub(world, "physics", attrib={"name": "1ms", "type": "ignored"})
    sub(physics, "max_step_size", "0.001")
    sub(physics, "real_time_factor", "1.0")

    sub(world, "plugin", attrib={"filename": "ignition-gazebo-physics-system", "name": "ignition::gazebo::systems::Physics"})
    sensors = sub(world, "plugin", attrib={"filename": "ignition-gazebo-sensors-system", "name": "ignition::gazebo::systems::Sensors"})
    sub(sensors, "render_engine", "ogre2")
    sub(world, "plugin", attrib={"filename": "ignition-gazebo-user-commands-system", "name": "ignition::gazebo::systems::UserCommands"})
    sub(world, "plugin", attrib={"filename": "ignition-gazebo-scene-broadcaster-system", "name": "ignition::gazebo::systems::SceneBroadcaster"})

    scene = sub(world, "scene")
    sub(scene, "ambient", rgba_text(ambient))
    sub(scene, "background", rgba_text(background))
    sub(scene, "shadows", "true")

    light = sub(world, "light", attrib={"type": "directional", "name": "sun"})
    sub(light, "cast_shadows", "true")
    sub(light, "pose", "0 0 10 0 0 0")
    sub(light, "diffuse", "0.9 0.9 0.9 1")
    sub(light, "specular", "0.2 0.2 0.2 1")
    attenuation = sub(light, "attenuation")
    sub(attenuation, "range", "1000")
    sub(attenuation, "constant", "0.9")
    sub(attenuation, "linear", "0.01")
    sub(attenuation, "quadratic", "0.001")
    sub(light, "direction", "-0.4 0.2 -0.9")


def add_box_geometry(parent: ET.Element, size_gazebo):
    geometry = sub(parent, "geometry")
    box = sub(geometry, "box")
    sub(box, "size", fmt_values(size_gazebo))


def add_cylinder_geometry(parent: ET.Element, radius: float, length: float):
    geometry = sub(parent, "geometry")
    cylinder = sub(geometry, "cylinder")
    sub(cylinder, "radius", fmt_float(radius))
    sub(cylinder, "length", fmt_float(length))


def add_friction(collision: ET.Element, mu: float, mu2: float):
    surface = sub(collision, "surface")
    friction = sub(surface, "friction")
    ode = sub(friction, "ode")
    sub(ode, "mu", fmt_float(mu))
    sub(ode, "mu2", fmt_float(mu2))


def add_box_collision(link: ET.Element, name: str, size_xyz, pose_xyz, mu: float, mu2: float):
    collision = sub(link, "collision", attrib={"name": name})
    sub(collision, "pose", fmt_values((*pose_xyz, 0.0, 0.0, 0.0)))
    add_box_geometry(collision, size_xyz)
    add_friction(collision, mu, mu2)
    return collision


def safe_name(value: str, fallback: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value).strip())
    return cleaned or fallback


def gazebo_port_defs_from_unity(obj: dict[str, Any]) -> list[dict[str, float | str]]:
    raw_ports = obj.get("ports")
    if not isinstance(raw_ports, list) or not raw_ports:
        raw_ports = [
            {
                "label": "default",
                "center_xy": [0.0, 0.0],
                "size_xyz": obj.get("port_size_xyz", (0.018, 0.012, 0.018)),
            }
        ]

    ports = []
    for index, port in enumerate(raw_ports):
        if not isinstance(port, dict):
            continue
        label = safe_name(str(port.get("label", f"port_{index}")), f"port_{index}")
        size_gz = unity_size_to_gazebo(vec3(port.get("size_xyz"), obj.get("port_size_xyz", (0.018, 0.012, 0.018))))
        center_xy = port.get("center_xy", port.get("center", (0.0, 0.0, 0.0)))
        if isinstance(center_xy, (list, tuple)) and len(center_xy) >= 2:
            center_x = float(center_xy[0])
            center_y = float(center_xy[1])
        else:
            center_x = 0.0
            center_y = 0.0
        ports.append(
            {
                "label": label,
                "center_y": -center_x,
                "center_z": center_y,
                "width": abs(float(size_gz[1])),
                "height": abs(float(size_gz[2])),
            }
        )

    return ports or gazebo_port_defs_from_unity({"port_size_xyz": obj.get("port_size_xyz", (0.018, 0.012, 0.018))})


def normalize_port_rects(size_xyz, port_defs: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    sx, sy, sz = (abs(float(v)) for v in size_xyz)
    rects = []
    for index, port in enumerate(port_defs):
        width = min(max(float(port.get("width", 0.001)), 0.001), sy * 0.90)
        height = min(max(float(port.get("height", 0.001)), 0.001), sz * 0.90)
        cy = clamp(float(port.get("center_y", 0.0)), -sy * 0.5 + width * 0.5, sy * 0.5 - width * 0.5)
        cz = clamp(float(port.get("center_z", 0.0)), -sz * 0.5 + height * 0.5, sz * 0.5 - height * 0.5)
        rects.append(
            {
                "label": safe_name(str(port.get("label", f"port_{index}")), f"port_{index}"),
                "center_y": cy,
                "center_z": cz,
                "width": width,
                "height": height,
            }
        )
    return rects


def add_port_box_frame_collisions(link: ET.Element, size_xyz, port_defs: list[dict[str, float | str]], mu: float, mu2: float):
    sx, sy, sz = (abs(float(v)) for v in size_xyz)
    ports = normalize_port_rects(size_xyz, port_defs)
    back_wall_depth = min(max(sx * 0.12, 0.003), sx * 0.35)
    back_wall_x = -(sx * 0.5) + back_wall_depth * 0.5

    current_y = -sy * 0.5
    for index, port in enumerate(sorted(ports, key=lambda p: float(p["center_y"]) - float(p["width"]) * 0.5)):
        start_y = float(port["center_y"]) - float(port["width"]) * 0.5
        end_y = float(port["center_y"]) + float(port["width"]) * 0.5
        if start_y > current_y + 0.001:
            gap_width = start_y - current_y
            add_box_collision(
                link,
                f"port_frame_gap_{index}_collision",
                (sx, gap_width, sz),
                (0.0, current_y + gap_width * 0.5, 0.0),
                mu,
                mu2,
            )
        current_y = max(current_y, end_y)
    if current_y < sy * 0.5 - 0.001:
        gap_width = sy * 0.5 - current_y
        add_box_collision(
            link,
            "port_frame_gap_end_collision",
            (sx, gap_width, sz),
            (0.0, current_y + gap_width * 0.5, 0.0),
            mu,
            mu2,
        )

    for port in ports:
        label = str(port["label"])
        center_y = float(port["center_y"])
        center_z = float(port["center_z"])
        port_width = float(port["width"])
        port_height = float(port["height"])
        top_height = sz * 0.5 - (center_z + port_height * 0.5)
        bottom_height = (center_z - port_height * 0.5) + sz * 0.5
        if top_height > 0.001:
            add_box_collision(
                link,
                f"port_frame_{label}_top_collision",
                (sx, port_width, top_height),
                (0.0, center_y, center_z + port_height * 0.5 + top_height * 0.5),
                mu,
                mu2,
            )
        if bottom_height > 0.001:
            add_box_collision(
                link,
                f"port_frame_{label}_bottom_collision",
                (sx, port_width, bottom_height),
                (0.0, center_y, center_z - port_height * 0.5 - bottom_height * 0.5),
                mu,
                mu2,
            )

    add_box_collision(
        link,
        "port_back_wall_collision",
        (back_wall_depth, sy, sz),
        (back_wall_x, 0.0, 0.0),
        mu,
        mu2,
    )


def add_port_box_frame_visuals(link: ET.Element, size_xyz, port_defs: list[dict[str, float | str]], rgba):
    sx, sy, sz = (abs(float(v)) for v in size_xyz)
    ports = normalize_port_rects(size_xyz, port_defs)
    back_wall_depth = min(max(sx * 0.12, 0.003), sx * 0.35)
    back_wall_x = -(sx * 0.5) + back_wall_depth * 0.5

    current_y = -sy * 0.5
    for index, port in enumerate(sorted(ports, key=lambda p: float(p["center_y"]) - float(p["width"]) * 0.5)):
        start_y = float(port["center_y"]) - float(port["width"]) * 0.5
        end_y = float(port["center_y"]) + float(port["width"]) * 0.5
        if start_y > current_y + 0.001:
            gap_width = start_y - current_y
            add_box_visual(
                link,
                f"port_frame_gap_{index}_visual",
                (sx, gap_width, sz),
                (0.0, current_y + gap_width * 0.5, 0.0),
                rgba,
            )
        current_y = max(current_y, end_y)
    if current_y < sy * 0.5 - 0.001:
        gap_width = sy * 0.5 - current_y
        add_box_visual(
            link,
            "port_frame_gap_end_visual",
            (sx, gap_width, sz),
            (0.0, current_y + gap_width * 0.5, 0.0),
            rgba,
        )

    for port in ports:
        label = str(port["label"])
        center_y = float(port["center_y"])
        center_z = float(port["center_z"])
        port_width = float(port["width"])
        port_height = float(port["height"])
        top_height = sz * 0.5 - (center_z + port_height * 0.5)
        bottom_height = (center_z - port_height * 0.5) + sz * 0.5
        if top_height > 0.001:
            add_box_visual(
                link,
                f"port_frame_{label}_top_visual",
                (sx, port_width, top_height),
                (0.0, center_y, center_z + port_height * 0.5 + top_height * 0.5),
                rgba,
            )
        if bottom_height > 0.001:
            add_box_visual(
                link,
                f"port_frame_{label}_bottom_visual",
                (sx, port_width, bottom_height),
                (0.0, center_y, center_z - port_height * 0.5 - bottom_height * 0.5),
                rgba,
            )

    add_box_visual(
        link,
        "port_back_wall_visual",
        (back_wall_depth, sy, sz),
        (back_wall_x, 0.0, 0.0),
        rgba,
    )


def add_material(visual: ET.Element, rgba, albedo_map=None):
    material = sub(visual, "material")
    sub(material, "ambient", rgba_text(rgba))
    sub(material, "diffuse", rgba_text(rgba))
    sub(material, "specular", "0.1 0.1 0.1 1")
    if albedo_map:
        pbr = sub(material, 'pbr')
        metal = sub(pbr, 'metal')
        sub(metal, 'albedo_map', str(albedo_map))
        sub(metal, 'metalness', '0')
        sub(metal, 'roughness', '0.95')


def add_box_visual(link: ET.Element, name: str, size_xyz, pose_xyz, rgba):
    visual = sub(link, "visual", attrib={"name": name})
    sub(visual, "pose", fmt_values((*pose_xyz, 0.0, 0.0, 0.0)))
    add_box_geometry(visual, size_xyz)
    add_material(visual, rgba)


def wood_enabled(obj: dict[str, Any]) -> bool:
    return str(obj.get("material_style", "")).strip().lower() in {"wood", "wooden"}


def add_wood_grain_visuals(
    link: ET.Element,
    base_name: str,
    size_xyz,
    center_xyz=(0.0, 0.0, 0.0),
    rgba=(0.30, 0.15, 0.06, 1.0),
    face_axis: str = "z",
    stripe_count: int = 6,
):
    sx, sy, sz = (abs(float(v)) for v in size_xyz)
    cx, cy, cz = (float(v) for v in center_xyz)
    stripe_count = max(1, int(stripe_count))
    min_size = max(0.0001, min(sx, sy, sz))
    line_thickness = max(min_size * 0.045, 0.00035)
    surface_thickness = max(min_size * 0.018, 0.00020)

    for index in range(stripe_count):
        offset = ((index + 0.5) / stripe_count - 0.5) * 0.82
        if face_axis == "x":
            stripe_size = (surface_thickness, max(sy * 0.72, line_thickness), line_thickness)
            stripe_pose = (cx + sx * 0.5 + surface_thickness * 0.5, cy, cz + offset * sz)
        elif face_axis == "y":
            stripe_size = (max(sx * 0.72, line_thickness), surface_thickness, line_thickness)
            stripe_pose = (cx, cy + sy * 0.5 + surface_thickness * 0.5, cz + offset * sz)
        else:
            stripe_size = (max(sx * 0.72, line_thickness), line_thickness, surface_thickness)
            stripe_pose = (cx, cy + offset * sy, cz + sz * 0.5 + surface_thickness * 0.5)
        add_box_visual(link, f"{base_name}_{index:02d}", stripe_size, stripe_pose, rgba)


def rubik_face_colors():
    # Match Unity face colors after Unity->Gazebo axis conversion:
    # Gazebo x=Unity z, Gazebo y=-Unity x, Gazebo z=Unity y.
    return {
        "x+": (0.08, 0.65, 0.12, 1.0),
        "x-": (0.04, 0.18, 0.85, 1.0),
        "y+": (1.00, 0.42, 0.02, 1.0),
        "y-": (0.85, 0.05, 0.04, 1.0),
        "z+": (0.95, 0.95, 0.90, 1.0),
        "z-": (0.95, 0.85, 0.06, 1.0),
    }


def add_rubik_stickers(link: ET.Element, size_xyz, order: int, sticker_gap: float):
    order = max(1, int(order))
    sx, sy, sz = (abs(float(v)) for v in size_xyz)
    thickness = max(min(sx, sy, sz) * 0.018, 0.0005)
    gap = max(0.0, float(sticker_gap))

    face_colors = rubik_face_colors()

    def offsets(length: float):
        cell = length / order
        start = -length * 0.5 + cell * 0.5
        return [start + i * cell for i in range(order)]

    def tile(length: float):
        return max(length / order - gap, length / order * 0.5)

    xs, ys, zs = offsets(sx), offsets(sy), offsets(sz)
    tx, ty, tz = tile(sx), tile(sy), tile(sz)
    index = 0

    for sign, color_key in ((1.0, "x+"), (-1.0, "x-")):
        for y in ys:
            for z in zs:
                add_box_visual(
                    link,
                    f"rubik_sticker_{index}",
                    (thickness, ty, tz),
                    (sign * (sx * 0.5 + thickness * 0.5), y, z),
                    face_colors[color_key],
                )
                index += 1

    for sign, color_key in ((1.0, "y+"), (-1.0, "y-")):
        for x in xs:
            for z in zs:
                add_box_visual(
                    link,
                    f"rubik_sticker_{index}",
                    (tx, thickness, tz),
                    (x, sign * (sy * 0.5 + thickness * 0.5), z),
                    face_colors[color_key],
                )
                index += 1

    for sign, color_key in ((1.0, "z+"), (-1.0, "z-")):
        for x in xs:
            for y in ys:
                add_box_visual(
                    link,
                    f"rubik_sticker_{index}",
                    (tx, ty, thickness),
                    (x, y, sign * (sz * 0.5 + thickness * 0.5)),
                    face_colors[color_key],
                )
                index += 1


def rubik_cubie_name(root_name: str, sx: int, sy: int, sz: int) -> str:
    def token(value: int) -> str:
        return "p" if value > 0 else "n"

    return f"{root_name}_cubie_x{token(sx)}_y{token(sy)}_z{token(sz)}"


def add_rubik_cubie_stickers(link: ET.Element, cubie_size, signs, sticker_gap: float):
    sx, sy, sz = (abs(float(v)) for v in cubie_size)
    thickness = max(min(sx, sy, sz) * 0.040, 0.00055)
    gap = max(0.0, float(sticker_gap))
    face_colors = rubik_face_colors()
    tile_x = max(sx - gap, sx * 0.5)
    tile_y = max(sy - gap, sy * 0.5)
    tile_z = max(sz - gap, sz * 0.5)
    sign_x, sign_y, sign_z = signs

    if sign_x > 0:
        add_box_visual(link, "sticker_x_pos", (thickness, tile_y, tile_z), (sx * 0.5 + thickness * 0.5, 0, 0), face_colors["x+"])
    else:
        add_box_visual(link, "sticker_x_neg", (thickness, tile_y, tile_z), (-(sx * 0.5 + thickness * 0.5), 0, 0), face_colors["x-"])

    if sign_y > 0:
        add_box_visual(link, "sticker_y_pos", (tile_x, thickness, tile_z), (0, sy * 0.5 + thickness * 0.5, 0), face_colors["y+"])
    else:
        add_box_visual(link, "sticker_y_neg", (tile_x, thickness, tile_z), (0, -(sy * 0.5 + thickness * 0.5), 0), face_colors["y-"])

    if sign_z > 0:
        add_box_visual(link, "sticker_z_pos", (tile_x, tile_y, thickness), (0, 0, sz * 0.5 + thickness * 0.5), face_colors["z+"])
    else:
        add_box_visual(link, "sticker_z_neg", (tile_x, tile_y, thickness), (0, 0, -(sz * 0.5 + thickness * 0.5)), face_colors["z-"])


def add_rubik_multibody(world: ET.Element, obj: dict[str, Any], task_group: dict[str, Any]):
    root_name = str(obj["id"])
    size = unity_size_to_gazebo(vec3(obj.get("size_xyz"), (0.04, 0.04, 0.04)))
    sx, sy, sz = size
    order = max(2, int(obj.get("rubik_order", 2)))
    if order != 2:
        raise ValueError("rubik_2x2 currently supports rubik_order: 2")

    group_pos = vec3(task_group.get("unity_local_position_xyz"))
    group_rot = rpy_deg_to_matrix(vec3(task_group.get("unity_local_euler_xyz")))
    local_pos = vec3(obj.get("local_position_xyz"))
    unity_world_pos = add_vec(group_pos, mat_vec_mul(group_rot, local_pos))
    center = unity_vector_to_gazebo(unity_world_pos)

    cubie_gap = max(0.0, float(obj.get("cubie_gap_m", 0.0012)))
    sticker_gap = max(0.0, float(obj.get("sticker_gap_m", 0.002)))
    cubie_size = (
        max(0.002, sx * 0.5 - cubie_gap),
        max(0.002, sy * 0.5 - cubie_gap),
        max(0.002, sz * 0.5 - cubie_gap),
    )
    collision_size = tuple(v * 0.985 for v in cubie_size)
    center_offset = (sx * 0.25, sy * 0.25, sz * 0.25)
    mass_each = max(0.001, float(obj.get("mass_kg", 0.055)) / 8.0)
    plastic = vec4(obj.get("color_rgba"), (0.025, 0.025, 0.025, 1.0))

    for sign_x in (-1, 1):
        for sign_y in (-1, 1):
            for sign_z in (-1, 1):
                name = rubik_cubie_name(root_name, sign_x, sign_y, sign_z)
                pose_xyz = (
                    center[0] + sign_x * center_offset[0],
                    center[1] + sign_y * center_offset[1],
                    center[2] + sign_z * center_offset[2],
                )
                model = sub(world, "model", attrib={"name": name})
                sub(model, "static", "false")
                sub(model, "pose", fmt_values((*pose_xyz, 0.0, 0.0, 0.0)))
                link = sub(model, "link", attrib={"name": "body_link"})
                add_inertial(link, "box", mass_each)

                collision = sub(link, "collision", attrib={"name": "collision"})
                add_box_geometry(collision, collision_size)
                add_friction(collision, 1.2, 1.2)

                visual = sub(link, "visual", attrib={"name": "plastic"})
                add_box_geometry(visual, cubie_size)
                add_material(visual, plastic)
                add_rubik_cubie_stickers(link, cubie_size, (sign_x, sign_y, sign_z), sticker_gap)


def add_table(world: ET.Element, table: dict[str, Any]):
    name = str(table.get("name", "table"))
    model = sub(world, "model", attrib={"name": name})
    sub(model, "static", "true")
    pose_xyz = vec3(table.get("gazebo_pose_xyz"), (0.0, 0.0, -0.4))
    pose_rpy = vec3(table.get("gazebo_pose_rpy"), (0.0, 0.0, 0.0))
    sub(model, "pose", fmt_values((*pose_xyz, *pose_rpy)))
    link = sub(model, "link", attrib={"name": "table_link"})
    size = vec3(table.get("gazebo_size_xyz"), (2.0, 2.0, 0.8))
    color = vec4(table.get("color_rgba"), (0.75, 0.75, 0.75, 1.0))

    collision = sub(link, "collision", attrib={"name": "table_collision"})
    add_box_geometry(collision, size)
    add_friction(collision, 1.0, 1.0)

    visual = sub(link, "visual", attrib={"name": "table_visual"})
    add_box_geometry(visual, size)
    add_material(visual, color, table.get('albedo_map'))


def add_inertial(link: ET.Element, object_type: str, mass: float):
    inertial = sub(link, "inertial")
    sub(inertial, "mass", fmt_float(mass))
    inertia = sub(inertial, "inertia")
    if object_type == "cylinder":
        values = {"ixx": 0.0000015, "ixy": 0, "ixz": 0, "iyy": 0.0000015, "iyz": 0, "izz": 0.0000010}
    else:
        values = {"ixx": 0.000002, "ixy": 0, "ixz": 0, "iyy": 0.000002, "iyz": 0, "izz": 0.000002}
    for key, value in values.items():
        sub(inertia, key, fmt_float(value))


def add_espresso_cup(link, obj, color):
    """Hollow espresso cup: disk base, 16 wall segments and three handle bars."""
    radius, height, wall = float(obj['radius']), float(obj['height']), 0.003
    for tag in ('collision', 'visual'):
        part = sub(link, tag, attrib={'name': 'cup_base_' + tag})
        sub(part, 'pose', fmt_values((0, 0, -height / 2 + wall / 2, 0, 0, 0)))
        add_cylinder_geometry(part, radius, wall)
        if tag == 'visual': add_material(part, color)
        else: add_friction(part, 0.9, 0.9)
    for i in range(16):
        angle = i * 2 * math.pi / 16
        for tag in ('collision', 'visual'):
            part = sub(link, tag, attrib={'name': f'wall_{i}_{tag}'})
            sub(part, 'pose', fmt_values(((radius-wall/2)*math.cos(angle),
                (radius-wall/2)*math.sin(angle), 0, 0, 0, angle)))
            add_box_geometry(part, (wall, 2*radius*math.tan(math.pi/16), height))
            if tag == 'visual': add_material(part, color)
            else: add_friction(part, 0.9, 0.9)
    for i, (size, center) in enumerate([
        ((0.018, 0.006, 0.006), (radius+0.006, 0, 0.015)),
        ((0.018, 0.006, 0.006), (radius+0.006, 0, -0.015)),
        ((0.006, 0.006, 0.036), (radius+0.015, 0, 0)),
    ]):
        add_box_collision(link, f'handle_{i}', size, center, 0.9, 0.9)
        add_box_visual(link, f'handle_visual_{i}', size, center, color)


def add_task_object(world: ET.Element, obj: dict[str, Any], task_group: dict[str, Any]):
    object_id = str(obj["id"])
    object_type = str(obj.get("type", "box")).lower()
    if object_type in {"rubik", "rubik_2x2", "rubiks_cube"} and bool_value(obj.get("multi_body"), True):
        add_rubik_multibody(world, obj, task_group)
        return

    link_name = str(obj.get("link_name") or ("plate_link" if object_type == "plate" else "body_link"))
    is_static = bool_value(obj.get("static"), object_type == "plate")
    color = vec4(obj.get("color_rgba"), (0.8, 0.8, 0.8, 1.0))
    group_pos = vec3(task_group.get("unity_local_position_xyz"))
    group_rot = rpy_deg_to_matrix(vec3(task_group.get("unity_local_euler_xyz")))
    local_pos = vec3(obj.get("local_position_xyz"))
    unity_world_pos = add_vec(group_pos, mat_vec_mul(group_rot, local_pos))
    gazebo_pose_xyz = unity_vector_to_gazebo(unity_world_pos)
    gazebo_pose_rpy = unity_euler_to_gazebo_rpy(vec3(obj.get("local_euler_xyz")))

    model = sub(world, "model", attrib={"name": object_id})
    sub(model, "static", "true" if is_static else "false")
    sub(model, "pose", fmt_values((*gazebo_pose_xyz, *gazebo_pose_rpy)))
    link = sub(model, "link", attrib={"name": link_name})
    if not is_static:
        mass = float(obj.get("mass_kg", 0.03))
        add_inertial(link, object_type, mass)
        if object_type in {"espresso_cup", "saucer"}:
            radius, height = float(obj['radius']), float(obj['height'])
            # Cylinder approximation, avoiding the tiny cube inertia used by legacy objects.
            for axis in ('ixx', 'iyy'):
                link.find('inertial/inertia/' + axis).text = fmt_float(mass*(3*radius**2+height**2)/12)
            link.find('inertial/inertia/izz').text = fmt_float(mass*radius**2/2)

    collision = sub(link, "collision", attrib={"name": "collision"})
    visual = sub(link, "visual", attrib={"name": "visual"})
    default_collision_friction = True
    if object_type == "espresso_cup":
        link.remove(collision)
        link.remove(visual)
        default_collision_friction = False
        add_espresso_cup(link, obj, color)
    elif object_type in {"cylinder", "saucer"}:
        radius = float(obj.get("radius", 0.01))
        height = float(obj.get("height", 0.048))
        add_cylinder_geometry(collision, radius, height)
        add_cylinder_geometry(visual, radius, height)
        add_material(visual, color)
    elif object_type in {"rubik", "rubik_2x2", "rubiks_cube"}:
        size = unity_size_to_gazebo(vec3(obj.get("size_xyz"), (0.04, 0.04, 0.04)))
        add_box_geometry(collision, size)
        add_box_geometry(visual, size)
        add_material(visual, color)
        add_rubik_stickers(
            link,
            size,
            int(obj.get("rubik_order", 2)),
            float(obj.get("sticker_gap_m", 0.002)),
        )
    elif object_type == "port_box":
        size = unity_size_to_gazebo(vec3(obj.get("size_xyz"), (0.04, 0.04, 0.04)))
        port_defs = gazebo_port_defs_from_unity(obj)
        link.remove(collision)
        link.remove(visual)
        default_collision_friction = False
        add_port_box_frame_collisions(
            link,
            size,
            port_defs,
            1.0 if is_static else 0.9,
            1.0 if is_static else 0.9,
        )
        add_port_box_frame_visuals(link, size, port_defs, color)
        if wood_enabled(obj):
            grain_color = vec4(obj.get("wood_grain_color_rgba"), (0.30, 0.15, 0.06, 1.0))
            add_wood_grain_visuals(link, "port_box_wood_top", size, rgba=grain_color, face_axis="z")
            add_wood_grain_visuals(link, "port_box_wood_side", size, rgba=grain_color, face_axis="y", stripe_count=5)
    elif object_type == "cable_rod":
        size = unity_size_to_gazebo(vec3(obj.get("size_xyz"), (0.009, 0.009, 0.075)))
        add_box_geometry(collision, size)
        add_box_geometry(visual, size)
        add_material(visual, color)
        grain_color = vec4(obj.get("wood_grain_color_rgba"), (0.33, 0.18, 0.08, 1.0))
        if wood_enabled(obj):
            add_wood_grain_visuals(link, "cable_rod_wood_top", size, rgba=grain_color, face_axis="z", stripe_count=4)
        plug_size = unity_size_to_gazebo(vec3(obj.get("plug_size_xyz"), (0.016, 0.010, 0.016)))
        plug_color = vec4(obj.get("plug_color_rgba"), (0.12, 0.12, 0.13, 1.0))
        plug_center = (-(size[0] * 0.5 + plug_size[0] * 0.5), 0.0, 0.0)
        add_box_collision(link, "plug_end_collision", plug_size, plug_center, 0.9, 0.9)
        add_box_visual(
            link,
            "plug_end_visual",
            plug_size,
            plug_center,
            plug_color,
        )
        if wood_enabled(obj):
            add_wood_grain_visuals(link, "cable_plug_wood_top", plug_size, center_xyz=plug_center, rgba=grain_color, face_axis="z", stripe_count=3)
    else:
        size = unity_size_to_gazebo(vec3(obj.get("size_xyz"), (0.02, 0.04, 0.02)))
        add_box_geometry(collision, size)
        add_box_geometry(visual, size)
        add_material(visual, color, obj.get('albedo_map'))
    if default_collision_friction:
        add_friction(collision, 1.0 if is_static else 0.9, 1.0 if is_static else 0.9)


def indent(elem: ET.Element, level: int = 0):
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i


def apply_cup_layout(task, task_group):
    if task.get("task_profile") == "cup_arrangement":
        from cup_layout import sample_layout
        layout = sample_layout(task, int(task['randomization'].get('seed', 0)))
        group_xyz = unity_vector_to_gazebo(vec3(task_group.get('unity_local_position_xyz')))
        if any(vec3(task_group.get('unity_local_euler_xyz'))):
            raise ValueError('cup_arrangement requires an unrotated task group')
        for obj in task['objects']:
            if obj['id'] in layout:
                x, y, z, yaw = layout[obj['id']]
                obj['local_position_xyz'] = [-(y-group_xyz[1]), z-group_xyz[2], x-group_xyz[0]]
                obj['local_euler_xyz'] = [0, -math.degrees(yaw), 0]


def generate(scene_profile_path: Path, output_path: Path):
    scene = read_yaml(scene_profile_path)
    workspace = read_yaml(resolve_profile_path(scene_profile_path, scene["workspace_profile"]))
    task = read_yaml(resolve_profile_path(scene_profile_path, scene["task_profile"]))

    task_group_name = str(scene.get("active_task_group") or task.get("task_group_name") or "TaskGroup_Main")
    task_groups = workspace.get("task_groups") or []
    task_group = next((g for g in task_groups if str(g.get("name")) == task_group_name), None)
    if task_group is None:
        raise ValueError(f"Task group '{task_group_name}' not found in workspace profile")

    apply_cup_layout(task, task_group)

    world_name = str((workspace.get("world") or {}).get("name", "ur_hande_dual_arm_tabletop"))
    sdf = ET.Element("sdf", attrib={"version": "1.8"})
    world = sub(sdf, "world", attrib={"name": world_name})

    world_cfg = workspace.get("world") or {}
    add_common_plugins(
        world,
        vec4(world_cfg.get("ambient_rgba"), (0.6, 0.6, 0.6, 1.0)),
        vec4(world_cfg.get("background_rgba"), (0.85, 0.87, 0.9, 1.0)),
    )
    add_table(world, workspace.get("table") or {})

    comment = ET.Comment(" Task objects generated from workspace/task profiles; Gazebo keeps them as top-level models for physics. ")
    world.append(comment)
    for obj in task.get("objects") or []:
        add_task_object(world, obj, task_group)

    indent(sdf)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(sdf)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    print(f"Generated {output_path} from {scene_profile_path}")


def write_unity_task_json(scene_profile_path: Path, unity_output_path: Path):
    scene = read_yaml(scene_profile_path)
    workspace = read_yaml(resolve_profile_path(scene_profile_path, scene["workspace_profile"]))
    task = read_yaml(resolve_profile_path(scene_profile_path, scene["task_profile"]))
    task_group_name = str(scene.get("active_task_group") or task.get("task_group_name") or "TaskGroup_Main")
    task_group = next(
        (g for g in (workspace.get("task_groups") or []) if str(g.get("name")) == task_group_name),
        None,
    )
    if task_group is None:
        raise ValueError(f"Task group '{task_group_name}' not found in workspace profile")

    apply_cup_layout(task, task_group)

    def json_vec3(values):
        x, y, z = vec3(values)
        return {"x": x, "y": y, "z": z}

    def json_color(values):
        r, g, b, a = vec4(values)
        return {"r": r, "g": g, "b": b, "a": a}

    def json_ports(obj: dict[str, Any]):
        raw_ports = obj.get("ports")
        if not isinstance(raw_ports, list) or not raw_ports:
            return []
        out = []
        for index, port in enumerate(raw_ports):
            if not isinstance(port, dict):
                continue
            center_xy = port.get("center_xy", port.get("center", (0.0, 0.0, 0.0)))
            if isinstance(center_xy, (list, tuple)) and len(center_xy) >= 2:
                center = (float(center_xy[0]), float(center_xy[1]), 0.0)
            else:
                center = (0.0, 0.0, 0.0)
            out.append(
                {
                    "label": str(port.get("label", f"port_{index}")),
                    "center": json_vec3(center),
                    "size": json_vec3(port.get("size_xyz", obj.get("port_size_xyz", (0.018, 0.012, 0.018)))),
                }
            )
        return out

    unity_task = {
        "taskProfile": str(task.get("task_profile", "unnamed_task")),
        "taskGroupName": task_group_name,
        "taskGroupLocalPosition": json_vec3(task_group.get("unity_local_position_xyz")),
        "taskGroupLocalEuler": json_vec3(task_group.get("unity_local_euler_xyz")),
        "objects": [],
    }

    def append_unity_object(obj: dict[str, Any]):
        unity_task["objects"].append(
            {
                "id": str(obj["id"]),
                "type": str(obj.get("type", "box")),
                "localPosition": json_vec3(obj.get("local_position_xyz")),
                "localEuler": json_vec3(obj.get("local_euler_xyz")),
                "size": json_vec3(obj.get("size_xyz", (0.02, 0.04, 0.02))),
                "radius": float(obj.get("radius", 0.01)),
                "height": float(obj.get("height", 0.048)),
                "rubikOrder": int(obj.get("rubik_order", 2)),
                "rubikX": int(obj.get("rubik_x", 0)),
                "rubikY": int(obj.get("rubik_y", 0)),
                "rubikZ": int(obj.get("rubik_z", 0)),
                "stickerGap": float(obj.get("sticker_gap_m", 0.002)),
                "portSize": json_vec3(obj.get("port_size_xyz", (0.018, 0.012, 0.018))),
                "ports": json_ports(obj),
                "portColor": json_color(obj.get("port_color_rgba", (0.01, 0.01, 0.012, 1.0))),
                "plugSize": json_vec3(obj.get("plug_size_xyz", (0.016, 0.010, 0.016))),
                "plugColor": json_color(obj.get("plug_color_rgba", (0.12, 0.12, 0.13, 1.0))),
                "color": json_color(obj.get("color_rgba", (0.8, 0.8, 0.8, 1.0))),
                "materialStyle": str(obj.get("material_style", "")),
                "woodGrainColor": json_color(obj.get("wood_grain_color_rgba", (0.30, 0.15, 0.06, 1.0))),
            }
        )

    for obj in task.get("objects") or []:
        object_type = str(obj.get("type", "box")).lower()
        if object_type in {"rubik", "rubik_2x2", "rubiks_cube"} and bool_value(obj.get("multi_body"), True):
            root_name = str(obj["id"])
            ux, uy, uz = vec3(obj.get("size_xyz"), (0.04, 0.04, 0.04))
            cubie_gap = max(0.0, float(obj.get("cubie_gap_m", 0.0012)))
            cubie_size = (
                max(0.002, abs(ux) * 0.5 - cubie_gap),
                max(0.002, abs(uy) * 0.5 - cubie_gap),
                max(0.002, abs(uz) * 0.5 - cubie_gap),
            )
            center = vec3(obj.get("local_position_xyz"))
            for gz_x in (-1, 1):
                for gz_y in (-1, 1):
                    for gz_z in (-1, 1):
                        unity_sign_x = -gz_y
                        unity_sign_y = gz_z
                        unity_sign_z = gz_x
                        cubie = dict(obj)
                        cubie["id"] = rubik_cubie_name(root_name, gz_x, gz_y, gz_z)
                        cubie["type"] = "rubik_cubie"
                        cubie["local_position_xyz"] = [
                            center[0] + unity_sign_x * abs(ux) * 0.25,
                            center[1] + unity_sign_y * abs(uy) * 0.25,
                            center[2] + unity_sign_z * abs(uz) * 0.25,
                        ]
                        cubie["size_xyz"] = cubie_size
                        cubie["rubik_x"] = unity_sign_x
                        cubie["rubik_y"] = unity_sign_y
                        cubie["rubik_z"] = unity_sign_z
                        append_unity_object(cubie)
            continue

        append_unity_object(obj)

    unity_output_path.parent.mkdir(parents=True, exist_ok=True)
    unity_output_path.write_text(json.dumps(unity_task, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {unity_output_path} from {scene_profile_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--scene-profile", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--unity-task-json",
        type=Path,
        default=None,
        help="Optional Unity Resources JSON output. Defaults to ../UnityApp/Assets/Resources/TaskProfiles/active_task.json when UnityApp exists.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    scene_profile = args.scene_profile or (repo_root / DEFAULT_SCENE_PROFILE)
    scene_profile = scene_profile.resolve()
    scene = read_yaml(scene_profile)
    workspace = read_yaml(resolve_profile_path(scene_profile, scene["workspace_profile"]))
    output = args.output
    if output is None:
        output_raw = (workspace.get("world") or {}).get("sdf_output") or scene.get("gazebo_world")
        output = repo_root / output_raw
    generate(scene_profile, output.resolve())

    unity_output = args.unity_task_json
    if unity_output is None:
        candidate = repo_root.parent / "UnityApp" / "Assets" / "Resources" / "TaskProfiles" / "active_task.json"
        unity_output = candidate if candidate.parent.parent.parent.exists() else None
    if unity_output is not None:
        write_unity_task_json(scene_profile, unity_output.resolve())


if __name__ == "__main__":
    main()
