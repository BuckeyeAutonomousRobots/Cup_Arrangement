#!/usr/bin/env python3
"""Check task YAML, generated Gazebo SDF, and Unity active_task.json agree on task objects."""

from __future__ import annotations

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from eval_common import BACKEND_ROOT, REPO_ROOT, print_done, results_dir, write_csv, write_json


def load_yaml(path: Path) -> dict:
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise SystemExit("PyYAML is required for this script. Install with: python3 -m pip install PyYAML") from exc
    return yaml.safe_load(path.read_text()) or {}


def active_scene_profile_path() -> Path:
    pointer = BACKEND_ROOT / "profiles/active_scene_profile.txt"
    if pointer.exists():
        value = pointer.read_text().strip()
        if value:
            p = (BACKEND_ROOT / value).resolve()
            if p.exists():
                return p
    return BACKEND_ROOT / "profiles/scenes/dual_arm_tabletop/scene.yaml"


def resolve_profile(base: Path, value: str) -> Path:
    return (base.parent / value).resolve()


def sdf_models(path: Path) -> dict[str, dict]:
    root = ET.parse(path).getroot()
    out = {}
    for model in root.findall(".//model"):
        name = model.attrib.get("name", "")
        if not name.startswith("Sync_"):
            continue
        pose_vals = [float(v) for v in (model.findtext("pose") or "0 0 0 0 0 0").split()[:6]]
        while len(pose_vals) < 6:
            pose_vals.append(0.0)
        out[name] = {"pose": pose_vals, "has_link": model.find("link") is not None}
    return out


def unity_objects(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text())
    return {obj["id"]: obj for obj in data.get("objects", []) if "id" in obj}


def task_objects(path: Path) -> dict[str, dict]:
    data = load_yaml(path)
    objects = data.get("objects", [])
    return {obj["id"]: obj for obj in objects if isinstance(obj, dict) and "id" in obj}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--position-tolerance", type=float, default=1e-5, help="Unity JSON local pose vs task YAML local pose tolerance.")
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    out = results_dir("task_profile_sdf_unity_consistency", args.output_root)

    scene_path = active_scene_profile_path()
    scene = load_yaml(scene_path)
    task_path = resolve_profile(scene_path, scene.get("task_profile", "../../tasks/pick_place_basic/task.yaml"))
    unity_path = REPO_ROOT / "UnityApp/Assets/Resources/TaskProfiles/active_task.json"
    sdf_path = BACKEND_ROOT / "simulation/worlds/ur_hande_dual_arm_tabletop.sdf"

    task = task_objects(task_path)
    unity = unity_objects(unity_path)
    sdf = sdf_models(sdf_path)
    all_ids = sorted(set(task) | set(unity) | set(sdf))

    rows = []
    failures = []
    for obj_id in all_ids:
        task_obj = task.get(obj_id)
        unity_obj = unity.get(obj_id)
        sdf_obj = sdf.get(obj_id)
        row = {
            "id": obj_id,
            "in_task_yaml": task_obj is not None,
            "in_unity_json": unity_obj is not None,
            "in_sdf": sdf_obj is not None,
            "type_task": task_obj.get("type") if task_obj else None,
            "type_unity": unity_obj.get("type") if unity_obj else None,
            "local_position_error": None,
            "ok": True,
        }
        if not (task_obj and unity_obj and sdf_obj):
            row["ok"] = False
        if task_obj and unity_obj:
            tp = task_obj.get("local_position_xyz") or task_obj.get("localPosition") or [0, 0, 0]
            up = unity_obj.get("localPosition", {})
            uv = [float(up.get("x", 0.0)), float(up.get("y", 0.0)), float(up.get("z", 0.0))]
            err = math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(tp, uv)))
            row["local_position_error"] = err
            if err > args.position_tolerance:
                row["ok"] = False
        if task_obj and unity_obj and task_obj.get("type") != unity_obj.get("type"):
            row["ok"] = False
        if not row["ok"]:
            failures.append(row)
        rows.append(row)

    summary = {
        "scene_profile": str(scene_path.relative_to(BACKEND_ROOT)),
        "task_profile": str(task_path.relative_to(BACKEND_ROOT)),
        "unity_json": str(unity_path.relative_to(REPO_ROOT)),
        "sdf": str(sdf_path.relative_to(BACKEND_ROOT)),
        "objects_checked": len(all_ids),
        "failures": len(failures),
    }
    write_csv(out / "object_consistency.csv", rows)
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    if failures:
        print(json.dumps(failures[:10], indent=2))
    print_done(out)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
