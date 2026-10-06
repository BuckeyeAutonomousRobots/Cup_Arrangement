#!/usr/bin/env python3
"""Command gripper open/close cycles and record finger joint symmetry."""

from __future__ import annotations

import argparse
import json

from eval_common import docker_ros_popen, print_done, require_container, results_dir, write_csv, write_json

GRIPPER_TEST = r'''
import json, os, time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

cycles = int(os.environ.get("EVAL_CYCLES", "3"))
hold = float(os.environ.get("EVAL_HOLD", "1.5"))
open_pos = float(os.environ.get("EVAL_OPEN", "0.0"))
close_pos = float(os.environ.get("EVAL_CLOSE", "0.025"))

class GripperTest(Node):
    def __init__(self):
        super().__init__("eval_gripper_timing_symmetry")
        self.rows = []
        self.latest = {}
        self.create_subscription(JointState, "/joint_states", self.on_js, 50)
        self.pub = {
            "left": self.create_publisher(Float64MultiArray, "/left_hande_position_controller/commands", 10),
            "right": self.create_publisher(Float64MultiArray, "/right_hande_position_controller/commands", 10),
        }
    def on_js(self, msg):
        now = time.time()
        for name, pos in zip(msg.name, msg.position):
            if "robotiq_hande" in name and "finger_joint" in name:
                self.latest[name] = float(pos)
        row = {"wall_time": now}
        row.update(self.latest)
        self.rows.append(row)
    def command(self, value):
        msg = Float64MultiArray()
        msg.data = [float(value)]
        for pub in self.pub.values():
            pub.publish(msg)

def spin_for(node, seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.05)

rclpy.init()
node = GripperTest()
spin_for(node, 0.5)
events = []
for cycle in range(cycles):
    for label, value in (("close", close_pos), ("open", open_pos)):
        node.command(value)
        events.append({"cycle": cycle, "command": label, "value": value, "wall_time": time.time()})
        spin_for(node, hold)
print(json.dumps({"events": events, "rows": node.rows}))
node.destroy_node()
rclpy.shutdown()
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--hold", type=float, default=1.5)
    parser.add_argument("--open", type=float, default=0.0)
    parser.add_argument("--close", type=float, default=0.025)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    require_container()
    out = results_dir("gripper_timing_symmetry", args.output_root)
    env = f"EVAL_CYCLES={args.cycles} EVAL_HOLD={args.hold} EVAL_OPEN={args.open} EVAL_CLOSE={args.close} python3 -u -"
    proc = docker_ros_popen(env, input_text=GRIPPER_TEST)
    stdout, stderr = proc.communicate(timeout=max(20.0, args.cycles * args.hold * 3.0 + 15.0))
    (out / "stderr.txt").write_text(stderr or "")
    data = json.loads((stdout or "{}").strip().splitlines()[-1])
    write_json(out / "events.json", data.get("events", []))
    write_csv(out / "finger_joint_samples.csv", data.get("rows", []))
    write_json(out / "summary.json", {"cycles": args.cycles, "samples": len(data.get("rows", [])), "note": "Use CSV to compute open/close time and left/right finger symmetry."})
    print_done(out)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
