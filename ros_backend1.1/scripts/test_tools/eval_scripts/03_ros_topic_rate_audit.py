#!/usr/bin/env python3
"""Measure ROS topic rates in-container with dynamic message subscriptions."""

from __future__ import annotations

import argparse
import json

from eval_common import add_common_args, docker_ros_popen, print_done, require_container, results_dir, write_csv, write_json

TOPIC_AUDIT = r'''
import json, os, time
import rclpy
from rclpy.node import Node
from rosidl_runtime_py.utilities import get_message

duration = float(os.environ.get("EVAL_DURATION", "20"))
topics = [t for t in os.environ.get("EVAL_TOPICS", "").split(",") if t]

class TopicAudit(Node):
    def __init__(self):
        super().__init__("eval_topic_rate_audit")
        self.data = {t: [] for t in topics}
        self.types = {}

    def attach_subscribers(self):
        available = {name: types for name, types in self.get_topic_names_and_types()}
        for topic in topics:
            type_names = available.get(topic, [])
            if not type_names:
                self.types[topic] = None
                continue
            self.types[topic] = type_names[0]
            msg_type = get_message(type_names[0])
            self.create_subscription(msg_type, topic, lambda msg, topic=topic: self.data[topic].append(time.monotonic()), 50)

rclpy.init()
node = TopicAudit()
discovery_deadline = time.monotonic() + float(os.environ.get("EVAL_DISCOVERY_TIMEOUT", "2.0"))
while time.monotonic() < discovery_deadline:
    available = {name: types for name, types in node.get_topic_names_and_types()}
    if all((topic in available) for topic in topics):
        break
    rclpy.spin_once(node, timeout_sec=0.05)
node.attach_subscribers()
deadline = time.monotonic() + duration
while rclpy.ok() and time.monotonic() < deadline:
    rclpy.spin_once(node, timeout_sec=0.05)
summary = []
for topic in topics:
    times = node.data.get(topic, [])
    gaps = [b-a for a,b in zip(times, times[1:])]
    hz = (len(times)-1)/(times[-1]-times[0]) if len(times) > 1 and times[-1] > times[0] else 0.0
    summary.append({
        "topic": topic,
        "type": node.types.get(topic),
        "messages": len(times),
        "mean_hz": hz,
        "min_gap_sec": min(gaps) if gaps else None,
        "max_gap_sec": max(gaps) if gaps else None,
    })
print(json.dumps({"duration_sec": duration, "topics": summary}))
node.destroy_node()
rclpy.shutdown()
'''

DEFAULT_TOPICS = [
    "/clock",
    "/joint_states",
    "/left_arm/joint_states_servo",
    "/right_arm/joint_states_servo",
    "/left_arm/received_pose_states",
    "/right_arm/received_pose_states",
    "/left_arm/target_twist_states",
    "/right_arm/target_twist_states",
    "/left_arm/servo_node/delta_twist_cmds",
    "/right_arm/servo_node/delta_twist_cmds",
    "/left_joint_group_velocity_controller/commands",
    "/right_joint_group_velocity_controller/commands",
    "/unity_sync/Sync_RedCube_pose",
    "/unity_sync/Sync_CableRod_pose",
    "/left_arm/haptics/contact_amplitude",
    "/right_arm/haptics/contact_amplitude",
    "/task_manager/status",
    "/task_manager/active_task_manifest",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--topics", default=",".join(DEFAULT_TOPICS), help="Comma-separated topic list.")
    args = parser.parse_args()
    require_container()
    out = results_dir("ros_topic_rate_audit", args.output_root)
    env = f"EVAL_DURATION={args.duration} EVAL_TOPICS='{args.topics}' python3 -u -"
    proc = docker_ros_popen(env, input_text=TOPIC_AUDIT)
    stdout, stderr = proc.communicate(timeout=max(10.0, args.duration + 10.0))
    (out / "stderr.txt").write_text(stderr or "")
    data = json.loads((stdout or "{}").strip().splitlines()[-1])
    write_json(out / "topic_rates.json", data)
    write_csv(out / "topic_rates.csv", data.get("topics", []))
    print(json.dumps(data, indent=2))
    print_done(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
