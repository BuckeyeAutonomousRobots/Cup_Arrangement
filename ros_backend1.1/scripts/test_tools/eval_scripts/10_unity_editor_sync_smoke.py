#!/usr/bin/env python3
"""Smoke-check saved Unity scene contains the generated workspace/task objects.

This script is file-based so it can run without opening Unity. For the real editor/MCP smoke,
run this first, then in Codex/Unity MCP execute:
Tools/Gazebo Replica/Rebuild Dual Arm Workspace In Active Scene
and read the Unity console.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval_common import REPO_ROOT, print_done, results_dir, write_csv, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="UnityApp/Assets/Scenes/GazeboReplica_DualArm_MR.unity")
    parser.add_argument("--task-json", default="UnityApp/Assets/Resources/TaskProfiles/active_task.json")
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()

    out = results_dir("unity_editor_sync_smoke", args.output_root)
    scene_path = REPO_ROOT / args.scene
    task_path = REPO_ROOT / args.task_json
    scene_text = scene_path.read_text(errors="ignore") if scene_path.exists() else ""
    task_data = json.loads(task_path.read_text()) if task_path.exists() else {"objects": []}

    required_groups = [
        ("workspace_root", ["GazeboWorkspace"]),
        ("task_groups_root", ["TaskGroups"]),
        ("active_task_group", [task_data.get("taskGroupName", "TaskGroup_Main")]),
        ("sync_root", ["GameObjects_sync"]),
        ("central_control_panel", ["MR_Central_ControlPanel", "MR_Central_ControlPanel_Controller"]),
    ]
    object_ids = [obj.get("id") for obj in task_data.get("objects", []) if obj.get("id")]
    rows = []
    for label, aliases in required_groups:
        found_aliases = [name for name in aliases if f"m_Name: {name}" in scene_text]
        rows.append({
            "name": label,
            "aliases": ",".join(aliases),
            "found_in_scene_yaml": bool(found_aliases),
            "found_as": ",".join(found_aliases),
            "category": "root_or_panel",
        })
    for name in object_ids:
        rows.append({
            "name": name,
            "aliases": name,
            "found_in_scene_yaml": f"m_Name: {name}" in scene_text,
            "found_as": name if f"m_Name: {name}" in scene_text else "",
            "category": "task_object",
        })
    missing = [row for row in rows if not row["found_in_scene_yaml"]]
    summary = {
        "scene": args.scene,
        "task_json": args.task_json,
        "checked_names": len(rows),
        "missing": len(missing),
        "mcp_followup": "Use Unity MCP execute_menu_item('Tools/Gazebo Replica/Rebuild Dual Arm Workspace In Active Scene'), then read_console and save scene.",
    }
    write_csv(out / "scene_name_presence.csv", rows)
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    if missing:
        print(json.dumps(missing[:20], indent=2))
    print_done(out)
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
