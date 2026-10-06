#!/usr/bin/env python3
"""Trace live headset-to-backend latency through ROS topics.

This test is meant to run while the Quest app is connected. It estimates backend
pipeline latency from received controller messages to mapper output, Servo input,
and Gazebo velocity command topics using subscriber arrival times in one process.

It does not measure optical/controller-to-visible-MR latency by itself because the
Quest packet timestamp is Unity `Time.time`, not a clock synchronized to ROS.
Use this together with a Quest screen recording or external video for visual
end-to-end latency.
"""

from __future__ import annotations

import argparse
import json

from eval_common import docker_ros_popen, print_done, require_container, results_dir, write_csv, write_json


TRACE_SCRIPT = r'''
import json
import math
import os
import time
from collections import deque

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Float64MultiArray
from teleop_bridge_msgs.msg import ReceivedPoseStates, TargetTwistStates

duration = float(os.environ.get("EVAL_DURATION", "30"))
motion_threshold_mps = float(os.environ.get("EVAL_MOTION_THRESHOLD_MPS", "0.08"))
event_cooldown_sec = float(os.environ.get("EVAL_EVENT_COOLDOWN_SEC", "0.50"))
target_norm_threshold = float(os.environ.get("EVAL_TARGET_NORM_THRESHOLD", "0.03"))
servo_norm_threshold = float(os.environ.get("EVAL_SERVO_NORM_THRESHOLD", "0.03"))
joint_norm_threshold = float(os.environ.get("EVAL_JOINT_NORM_THRESHOLD", "0.03"))
trace_samples_limit = int(os.environ.get("EVAL_TRACE_SAMPLES_LIMIT", "20000"))

ARM_TOPICS = {
    "left": {
        "received": "/left_arm/received_pose_states",
        "target": "/left_arm/target_twist_states",
        "servo": "/left_arm/servo_node/delta_twist_cmds",
        "joint_cmd": "/left_joint_group_velocity_controller/commands",
    },
    "right": {
        "received": "/right_arm/received_pose_states",
        "target": "/right_arm/target_twist_states",
        "servo": "/right_arm/servo_node/delta_twist_cmds",
        "joint_cmd": "/right_joint_group_velocity_controller/commands",
    },
}


def vec_norm(values):
    return math.sqrt(sum(float(v) * float(v) for v in values))


def twist_norm(twist):
    return vec_norm([
        twist.linear.x,
        twist.linear.y,
        twist.linear.z,
        twist.angular.x,
        twist.angular.y,
        twist.angular.z,
    ])


class LatencyTrace(Node):
    def __init__(self):
        super().__init__("eval_headset_backend_latency_trace")
        self.samples = []
        self.events = []
        self.topic_times = {arm: {stage: [] for stage in topics} for arm, topics in ARM_TOPICS.items()}
        self.last_arrival = {arm: {} for arm in ARM_TOPICS}
        self.last_pose = {arm: None for arm in ARM_TOPICS}
        self.last_event_time = {arm: -1e9 for arm in ARM_TOPICS}
        self.open_events = {arm: deque() for arm in ARM_TOPICS}
        self.start_time = time.monotonic()

        for arm, topics in ARM_TOPICS.items():
            self.create_subscription(
                ReceivedPoseStates,
                topics["received"],
                lambda msg, arm=arm: self.on_received(arm, msg),
                100,
            )
            self.create_subscription(
                TargetTwistStates,
                topics["target"],
                lambda msg, arm=arm: self.on_target(arm, msg),
                100,
            )
            self.create_subscription(
                TwistStamped,
                topics["servo"],
                lambda msg, arm=arm: self.on_servo(arm, msg),
                100,
            )
            self.create_subscription(
                Float64MultiArray,
                topics["joint_cmd"],
                lambda msg, arm=arm: self.on_joint_cmd(arm, msg),
                100,
            )

    def rel_time(self):
        return time.monotonic() - self.start_time

    def record_sample(self, arm, stage, t, value_norm=None, tracked=None, teleop=None):
        self.topic_times[arm][stage].append(t)
        self.last_arrival[arm][stage] = t
        if len(self.samples) < trace_samples_limit:
            self.samples.append({
                "time_sec": t - self.start_time,
                "arm": arm,
                "stage": stage,
                "value_norm": value_norm,
                "tracked": tracked,
                "teleop_enable": teleop,
            })

    def record_event_stage(self, arm, stage, t, threshold_norm):
        while self.open_events[arm] and t - self.open_events[arm][0]["input_time"] > 1.5:
            self.open_events[arm].popleft()
        for event in self.open_events[arm]:
            key = f"{stage}_time"
            if key not in event and t >= event["input_time"]:
                event[key] = t
                event[f"{stage}_latency_ms"] = (t - event["input_time"]) * 1000.0
                event[f"{stage}_norm"] = threshold_norm
                break

    def on_received(self, arm, msg):
        t = time.monotonic()
        p = msg.pose.position
        current = (float(p.x), float(p.y), float(p.z))
        speed = 0.0
        previous = self.last_pose[arm]
        if previous is not None:
            prev_t, prev_p = previous
            dt = max(t - prev_t, 1e-6)
            speed = vec_norm([current[i] - prev_p[i] for i in range(3)]) / dt
        self.last_pose[arm] = (t, current)
        self.record_sample(arm, "received", t, speed, bool(msg.tracked), bool(msg.teleop_enable))

        if bool(msg.tracked) and bool(msg.teleop_enable):
            if speed >= motion_threshold_mps and t - self.last_event_time[arm] >= event_cooldown_sec:
                event = {
                    "event_id": f"{arm}_{len(self.events) + 1}",
                    "arm": arm,
                    "input_time": t,
                    "input_time_sec": t - self.start_time,
                    "input_speed_mps": speed,
                }
                self.events.append(event)
                self.open_events[arm].append(event)
                self.last_event_time[arm] = t

    def on_target(self, arm, msg):
        t = time.monotonic()
        norm = twist_norm(msg.twist)
        self.record_sample(arm, "target", t, norm, bool(msg.tracked), None)
        if norm >= target_norm_threshold:
            self.record_event_stage(arm, "target", t, norm)

    def on_servo(self, arm, msg):
        t = time.monotonic()
        norm = twist_norm(msg.twist)
        self.record_sample(arm, "servo", t, norm, None, None)
        if norm >= servo_norm_threshold:
            self.record_event_stage(arm, "servo", t, norm)

    def on_joint_cmd(self, arm, msg):
        t = time.monotonic()
        norm = max([abs(float(v)) for v in msg.data], default=0.0)
        self.record_sample(arm, "joint_cmd", t, norm, None, None)
        if norm >= joint_norm_threshold:
            self.record_event_stage(arm, "joint_cmd", t, norm)

    def summarize_topic_rates(self):
        rows = []
        for arm, stages in self.topic_times.items():
            for stage, times in stages.items():
                gaps = [b - a for a, b in zip(times, times[1:])]
                hz = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 and times[-1] > times[0] else 0.0
                rows.append({
                    "arm": arm,
                    "stage": stage,
                    "messages": len(times),
                    "mean_hz": hz,
                    "min_gap_sec": min(gaps) if gaps else None,
                    "max_gap_sec": max(gaps) if gaps else None,
                })
        return rows

    def summarize_hop_delays(self):
        rows = []
        pairs = [
            ("received", "target"),
            ("target", "servo"),
            ("servo", "joint_cmd"),
            ("received", "joint_cmd"),
        ]
        for arm in ARM_TOPICS:
            for upstream, downstream in pairs:
                upstream_times = self.topic_times[arm][upstream]
                downstream_times = self.topic_times[arm][downstream]
                delays = []
                idx = 0
                for t_down in downstream_times:
                    while idx + 1 < len(upstream_times) and upstream_times[idx + 1] <= t_down:
                        idx += 1
                    if upstream_times and upstream_times[idx] <= t_down:
                        delays.append((t_down - upstream_times[idx]) * 1000.0)
                if delays:
                    delays_sorted = sorted(delays)
                    mean = sum(delays) / len(delays)
                    rows.append({
                        "arm": arm,
                        "upstream": upstream,
                        "downstream": downstream,
                        "samples": len(delays),
                        "mean_delay_ms": mean,
                        "min_delay_ms": delays_sorted[0],
                        "max_delay_ms": delays_sorted[-1],
                        "p95_delay_ms": delays_sorted[int(0.95 * (len(delays_sorted) - 1))],
                    })
                else:
                    rows.append({
                        "arm": arm,
                        "upstream": upstream,
                        "downstream": downstream,
                        "samples": 0,
                        "mean_delay_ms": None,
                        "min_delay_ms": None,
                        "max_delay_ms": None,
                        "p95_delay_ms": None,
                    })
        return rows


rclpy.init()
node = LatencyTrace()
deadline = time.monotonic() + duration
while rclpy.ok() and time.monotonic() < deadline:
    rclpy.spin_once(node, timeout_sec=0.01)

events_out = []
for event in node.events:
    row = dict(event)
    # Drop absolute monotonic timestamps; keep relative time and latency columns.
    row.pop("input_time", None)
    for key in list(row.keys()):
        if key.endswith("_time") and key != "input_time":
            row[f"{key}_sec"] = row.pop(key) - node.start_time
    events_out.append(row)

result = {
    "duration_sec": duration,
    "motion_threshold_mps": motion_threshold_mps,
    "event_cooldown_sec": event_cooldown_sec,
    "target_norm_threshold": target_norm_threshold,
    "servo_norm_threshold": servo_norm_threshold,
    "joint_norm_threshold": joint_norm_threshold,
    "topic_rates": node.summarize_topic_rates(),
    "hop_delays": node.summarize_hop_delays(),
    "events": events_out,
    "note": "Arrival-time latency measured inside ROS/container. Quest Unity timestamp is not host-synchronized, so this is backend pipeline latency after receiver publication, not optical end-to-end latency.",
}
print(json.dumps(result))
node.destroy_node()
rclpy.shutdown()
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--motion-threshold-mps", type=float, default=0.08)
    parser.add_argument("--event-cooldown-sec", type=float, default=0.50)
    parser.add_argument("--target-norm-threshold", type=float, default=0.03)
    parser.add_argument("--servo-norm-threshold", type=float, default=0.03)
    parser.add_argument("--joint-norm-threshold", type=float, default=0.03)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()

    require_container()
    out = results_dir("headset_backend_latency_trace", args.output_root)
    env = (
        f"EVAL_DURATION={args.duration} "
        f"EVAL_MOTION_THRESHOLD_MPS={args.motion_threshold_mps} "
        f"EVAL_EVENT_COOLDOWN_SEC={args.event_cooldown_sec} "
        f"EVAL_TARGET_NORM_THRESHOLD={args.target_norm_threshold} "
        f"EVAL_SERVO_NORM_THRESHOLD={args.servo_norm_threshold} "
        f"EVAL_JOINT_NORM_THRESHOLD={args.joint_norm_threshold} "
        "python3 -u -"
    )
    proc = docker_ros_popen(env, input_text=TRACE_SCRIPT)
    stdout, stderr = proc.communicate(timeout=max(15.0, args.duration + 15.0))
    (out / "stderr.txt").write_text(stderr or "", encoding="utf-8")
    if proc.returncode != 0:
        (out / "stdout.txt").write_text(stdout or "", encoding="utf-8")
        raise SystemExit(f"latency trace failed with code {proc.returncode}; see {out}")
    data = json.loads((stdout or "{}").strip().splitlines()[-1])
    write_json(out / "summary.json", data)
    write_csv(out / "topic_rates.csv", data.get("topic_rates", []))
    write_csv(out / "hop_delays_ms.csv", data.get("hop_delays", []))
    write_csv(out / "motion_events.csv", data.get("events", []))
    print(json.dumps(data, indent=2))
    print_done(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
