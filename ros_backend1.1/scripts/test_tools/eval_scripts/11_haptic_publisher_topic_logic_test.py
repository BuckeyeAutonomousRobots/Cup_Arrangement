#!/usr/bin/env python3
"""Record haptic output topics and summarize pulse/continuous behavior."""

from __future__ import annotations

import argparse
import json

from eval_common import add_common_args, docker_ros_popen, print_done, require_container, results_dir, write_csv, write_json

HAPTIC_MONITOR = r'''
import json, os, time
import rclpy
from rclpy.node import Node
from rosidl_runtime_py.utilities import get_message

duration = float(os.environ.get("EVAL_DURATION", "20"))
topics = [t for t in os.environ.get("EVAL_TOPICS", "").split(",") if t]

class HapticMonitor(Node):
    def __init__(self):
        super().__init__("eval_haptic_topic_logic")
        self.rows = []
        self.types = {}

    def attach_subscribers(self):
        available = {name: types for name, types in self.get_topic_names_and_types()}
        for topic in topics:
            type_names = available.get(topic, [])
            self.types[topic] = type_names[0] if type_names else None
            if not type_names:
                continue
            msg_type = get_message(type_names[0])
            self.create_subscription(msg_type, topic, lambda msg, topic=topic: self._on_msg(topic, msg), 50)
    def _on_msg(self, topic, msg):
        value = getattr(msg, "data", None)
        self.rows.append({"wall_time": time.time(), "topic": topic, "value": float(value) if value is not None else None})

rclpy.init()
node = HapticMonitor()
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
print(json.dumps({"types": node.types, "rows": node.rows}))
node.destroy_node()
rclpy.shutdown()
'''


def summarize_rows(rows: list[dict]) -> dict:
    by_topic = {}
    for row in rows:
        topic = row["topic"]
        by_topic.setdefault(topic, []).append(row)
    summary = {}
    for topic, vals in by_topic.items():
        nums = [float(v["value"]) for v in vals if isinstance(v.get("value"), (int, float))]
        nonzero = [v for v in nums if abs(v) > 1e-4]
        rising_edges = 0
        was_zero = True
        for value in nums:
            is_nonzero = abs(value) > 1e-4
            if is_nonzero and was_zero:
                rising_edges += 1
            was_zero = not is_nonzero
        summary[topic] = {
            "messages": len(vals),
            "nonzero_messages": len(nonzero),
            "rising_edges": rising_edges,
            "max_amplitude": max(nonzero) if nonzero else 0.0,
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--topics", default="/left_arm/haptics/contact_amplitude,/right_arm/haptics/contact_amplitude")
    args = parser.parse_args()
    require_container()
    out = results_dir("haptic_topic_logic", args.output_root)
    env = f"EVAL_DURATION={args.duration} EVAL_TOPICS='{args.topics}' python3 -u -"
    proc = docker_ros_popen(env, input_text=HAPTIC_MONITOR)
    stdout, stderr = proc.communicate(timeout=max(10.0, args.duration + 10.0))
    (out / "stderr.txt").write_text(stderr or "")
    data = json.loads((stdout or "{}").strip().splitlines()[-1])
    rows = data.get("rows", [])
    summary = {"duration_sec": args.duration, "types": data.get("types", {}), "topics": summarize_rows(rows)}
    write_csv(out / "haptic_samples.csv", rows)
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    print_done(out)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
