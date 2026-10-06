#!/usr/bin/env python3
"""Trigger robot/task reset and compare synced object poses against generated SDF poses."""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

from eval_common import BACKEND_ROOT, docker_exec, docker_ros_popen, print_done, require_container, results_dir, write_csv, write_json

RESET_AND_COLLECT = r'''
import json, os, time
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from teleop_bridge_msgs.msg import TargetTwistStates

object_names = [v for v in os.environ.get("EVAL_OBJECTS", "").split(",") if v]
reset_robot = os.environ.get("EVAL_RESET_ROBOT", "0") == "1"
reset_scene = os.environ.get("EVAL_RESET_SCENE", "1") == "1"
publish_sec = float(os.environ.get("EVAL_RESET_PUBLISH_SEC", "0.8"))
settle_sec = float(os.environ.get("EVAL_SETTLE_SEC", "3.0"))

class ResetProbe(Node):
    def __init__(self):
        super().__init__("eval_reset_reliability")
        self.latest = {}
        self.left_pub = self.create_publisher(TargetTwistStates, "/left_arm/target_twist_states", 20)
        self.right_pub = self.create_publisher(TargetTwistStates, "/right_arm/target_twist_states", 20)
        for name in object_names:
            self.create_subscription(
                PoseStamped,
                f"/unity_sync/{name}_pose",
                lambda msg, name=name: self._on_pose(name, msg),
                10,
            )
    def _on_pose(self, name, msg):
        self.latest[name] = {
            "x": float(msg.pose.position.x),
            "y": float(msg.pose.position.y),
            "z": float(msg.pose.position.z),
            "qx": float(msg.pose.orientation.x),
            "qy": float(msg.pose.orientation.y),
            "qz": float(msg.pose.orientation.z),
            "qw": float(msg.pose.orientation.w),
            "stamp_sec": int(msg.header.stamp.sec) + int(msg.header.stamp.nanosec) * 1e-9,
            "wall_time": time.time(),
        }
    def _msg(self):
        msg = TargetTwistStates()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "eval_reset"
        msg.tracked = True
        msg.reset_enable = bool(reset_robot or reset_scene)
        msg.reset_robot_enable = bool(reset_robot)
        msg.reset_scene_enable = bool(reset_scene)
        return msg
    def publish_reset(self):
        msg = self._msg()
        self.left_pub.publish(msg)
        self.right_pub.publish(msg)

def spin_for(node, seconds, publish=False):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if publish:
            node.publish_reset()
        rclpy.spin_once(node, timeout_sec=0.05)

rclpy.init()
node = ResetProbe()
spin_for(node, 0.5, publish=False)
spin_for(node, publish_sec, publish=True)
spin_for(node, settle_sec, publish=False)
print(json.dumps({"latest": node.latest, "reset_robot": reset_robot, "reset_scene": reset_scene}))
node.destroy_node()
rclpy.shutdown()
'''


def load_sdf_poses(path: Path) -> dict[str, dict[str, float]]:
    root = ET.parse(path).getroot()
    poses: dict[str, dict[str, float]] = {}
    for model in root.findall(".//model"):
        name = model.attrib.get("name", "")
        if not name.startswith("Sync_"):
            continue
        values = [float(v) for v in (model.findtext("pose") or "0 0 0 0 0 0").split()[:6]]
        while len(values) < 6:
            values.append(0.0)
        poses[name] = {"x": values[0], "y": values[1], "z": values[2], "roll": values[3], "pitch": values[4], "yaw": values[5]}
    return poses


def disturb_object(name: str, expected: dict[str, float], dx: float, dy: float, dz: float) -> None:
    req = (
        f'name: "{name}" '
        f'position: {{x: {expected["x"] + dx} y: {expected["y"] + dy} z: {expected["z"] + dz}}} '
        'orientation: {x: 0 y: 0 z: 0 w: 1}'
    )
    cmd = (
        'ign service -s /world/ur_hande_dual_arm_tabletop/set_pose '
        '--reqtype ignition.msgs.Pose --reptype ignition.msgs.Boolean --timeout 2000 '
        f'--req {json.dumps(req)}'
    )
    docker_exec(cmd, check=False, timeout=5.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects", default="", help="Comma-separated Sync_* objects. Empty means all generated Sync_* models.")
    parser.add_argument("--reset-robot", action="store_true", help="Also request robot reset. Default only resets task objects.")
    parser.add_argument("--settle", type=float, default=4.0, help="Seconds to wait after reset request before comparing poses.")
    parser.add_argument("--tolerance", type=float, default=0.05, help="Position tolerance in meters.")
    parser.add_argument("--disturb", action="store_true", help="Move tested objects before resetting them. Use only when Gazebo is running.")
    parser.add_argument("--disturb-dx", type=float, default=0.08)
    parser.add_argument("--disturb-dy", type=float, default=0.0)
    parser.add_argument("--disturb-dz", type=float, default=0.08)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()

    require_container()
    out = results_dir("reset_reliability", args.output_root)
    sdf_path = BACKEND_ROOT / "simulation/worlds/ur_hande_dual_arm_tabletop.sdf"
    expected = load_sdf_poses(sdf_path)
    names = [v.strip() for v in args.objects.split(",") if v.strip()] or sorted(expected)
    names = [name for name in names if name in expected]

    if args.disturb:
        for name in names:
            disturb_object(name, expected[name], args.disturb_dx, args.disturb_dy, args.disturb_dz)

    env = (
        f"EVAL_OBJECTS='{','.join(names)}' "
        f"EVAL_RESET_ROBOT={'1' if args.reset_robot else '0'} "
        "EVAL_RESET_SCENE=1 "
        f"EVAL_SETTLE_SEC={args.settle} python3 -u -"
    )
    proc = docker_ros_popen(env, input_text=RESET_AND_COLLECT)
    stdout, stderr = proc.communicate(timeout=max(15.0, args.settle + 15.0))
    (out / "stderr.txt").write_text(stderr or "")
    data = json.loads((stdout or "{}").strip().splitlines()[-1])
    observed = data.get("latest", {})

    rows = []
    failures = []
    for name in names:
        obs = observed.get(name)
        exp = expected[name]
        if not obs:
            row = {"object": name, "ok": False, "reason": "no_sync_pose"}
            failures.append(row)
            rows.append(row)
            continue
        error = math.sqrt((obs["x"] - exp["x"]) ** 2 + (obs["y"] - exp["y"]) ** 2 + (obs["z"] - exp["z"]) ** 2)
        row = {
            "object": name,
            "ok": error <= args.tolerance,
            "error_m": error,
            "expected_x": exp["x"], "expected_y": exp["y"], "expected_z": exp["z"],
            "observed_x": obs["x"], "observed_y": obs["y"], "observed_z": obs["z"],
        }
        if not row["ok"]:
            failures.append(row)
        rows.append(row)

    write_csv(out / "reset_pose_errors.csv", rows)
    write_json(out / "observed_sync_poses.json", observed)
    summary = {"tested_objects": len(names), "failures": len(failures), "tolerance_m": args.tolerance, "disturbed_first": args.disturb}
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    print_done(out)
    return 0 if not failures and proc.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
