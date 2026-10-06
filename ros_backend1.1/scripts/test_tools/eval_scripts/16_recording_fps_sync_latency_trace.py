#!/usr/bin/env python3
"""Trace recording FPS impact plus MR sync latency.

Run this before pressing the camera recording button in the Quest app. The Unity
``RecordingPerformanceTraceLogger`` publishes recording state and FPS samples;
this script waits for recording to start, records FPS plus Gazebo-vs-Unity visual
alignment/motion-onset data, and exits after recording has been stopped for the
configured post-recording tail.
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
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_msgs.msg import String
from teleop_bridge_msgs.msg import ReceivedPoseStates

max_wait_sec = float(os.environ.get("EVAL_MAX_WAIT_SEC", "120"))
max_duration_sec = float(os.environ.get("EVAL_MAX_DURATION_SEC", "240"))
post_recording_tail_sec = float(os.environ.get("EVAL_POST_RECORDING_TAIL_SEC", "15"))
object_id = os.environ.get("EVAL_OBJECT_ID", "Sync_RedCube")
robot_profile_path = os.environ.get(
    "EVAL_ROBOT_PROFILE",
    "/workspace/profiles/robots/ur5e_hande_dual/robot.yaml",
)
max_pair_age_sec = float(os.environ.get("EVAL_MAX_PAIR_AGE_SEC", "0.10"))
hand_motion_threshold_mps = float(os.environ.get("EVAL_HAND_MOTION_THRESHOLD_MPS", "0.08"))
ee_motion_threshold_mps = float(os.environ.get("EVAL_EE_MOTION_THRESHOLD_MPS", "0.015"))
object_motion_threshold_mps = float(os.environ.get("EVAL_OBJECT_MOTION_THRESHOLD_MPS", "0.010"))
event_cooldown_sec = float(os.environ.get("EVAL_EVENT_COOLDOWN_SEC", "0.50"))
event_expire_sec = float(os.environ.get("EVAL_EVENT_EXPIRE_SEC", "2.0"))
pose_samples_limit = int(os.environ.get("EVAL_POSE_SAMPLES_LIMIT", "50000"))
alignment_samples_limit = int(os.environ.get("EVAL_ALIGNMENT_SAMPLES_LIMIT", "50000"))
fps_samples_limit = int(os.environ.get("EVAL_FPS_SAMPLES_LIMIT", "20000"))


def vec_norm(values):
    return math.sqrt(sum(float(v) * float(v) for v in values))


def pose_tuple(msg):
    p = msg.pose.position
    q = msg.pose.orientation
    return (
        float(p.x), float(p.y), float(p.z),
        float(q.x), float(q.y), float(q.z), float(q.w),
    )


def quat_normalize(q):
    n = math.sqrt(sum(float(v) * float(v) for v in q))
    if n <= 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    return tuple(float(v) / n for v in q)


def quat_multiply(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return quat_normalize((
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ))


def quat_conjugate(q):
    return (-q[0], -q[1], -q[2], q[3])


def quat_rotate(q, v):
    rotated = quat_multiply(quat_multiply(q, (v[0], v[1], v[2], 0.0)), quat_conjugate(q))
    return (rotated[0], rotated[1], rotated[2])


def rpy_to_quat(roll, pitch, yaw):
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return quat_normalize((
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ))


def load_robot_spawn_transforms(path):
    defaults = {
        "left": ((0.0, 0.60, 0.0), (0.0, 0.0, 0.0, 1.0)),
        "right": ((0.0, -0.60, 0.0), (0.0, 0.0, 0.0, 1.0)),
    }
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return defaults

    transforms = dict(defaults)
    for robot in data.get("robots", []) or []:
        arm_id = str(robot.get("arm_id", ""))
        arm = "left" if arm_id.startswith("left") else "right" if arm_id.startswith("right") else None
        pose = robot.get("spawn_pose_xyz_rpy")
        if arm is None or not isinstance(pose, (list, tuple)) or len(pose) < 6:
            continue
        xyz = tuple(float(v) for v in pose[:3])
        rpy = tuple(float(v) for v in pose[3:6])
        transforms[arm] = (xyz, rpy_to_quat(*rpy))
    return transforms


def transform_base_pose_to_world(pose, base_transform):
    if base_transform is None:
        return pose
    base_xyz, base_q = base_transform
    rotated_pos = quat_rotate(base_q, pose[:3])
    world_pos = (
        base_xyz[0] + rotated_pos[0],
        base_xyz[1] + rotated_pos[1],
        base_xyz[2] + rotated_pos[2],
    )
    world_q = quat_multiply(base_q, pose[3:7])
    return world_pos + world_q


def pose_speed(prev, current, dt):
    if prev is None or dt <= 1e-6:
        return 0.0
    return vec_norm([current[i] - prev[i] for i in range(3)]) / dt


def quat_angle_deg(a, b):
    dot = abs(a[3] * b[3] + a[4] * b[4] + a[5] * b[5] + a[6] * b[6])
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(2.0 * math.acos(dot))


def stamp_to_sec(stamp):
    try:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9
    except Exception:
        return None


def parse_json_payload(msg):
    try:
        return json.loads(msg.data or "{}")
    except Exception as exc:
        return {"parse_error": str(exc), "raw": msg.data or ""}


class RecordingFpsSyncTrace(Node):
    def __init__(self):
        super().__init__("eval_recording_fps_sync_latency_trace")
        self.start_time = time.monotonic()
        self.first_recording_time = None
        self.last_recording_time = None
        self.tail_deadline = None
        self.done_reason = None
        self.robot_spawn_transforms = load_robot_spawn_transforms(robot_profile_path)
        self.fps_samples = []
        self.recording_events = []
        self.pose_samples = []
        self.alignment_samples = []
        self.motion_events = []
        self.topic_times = {}
        self.latest = {}
        self.previous_pose = {}
        self.previous_time = {}
        self.open_arm_events = {"left": deque(), "right": deque()}
        self.open_object_events = deque()
        self.last_hand_event_time = {"left": -1e9, "right": -1e9}
        self.last_object_event_time = -1e9

        self.create_subscription(String, "/unity_eval/recording_state", self.on_recording_state, 100)
        self.create_subscription(String, "/unity_eval/fps_sample", self.on_fps_sample, 100)
        self.create_subscription(PoseStamped, f"/unity_sync/{object_id}_pose",
                                 lambda msg: self.on_pose("object", "gazebo", object_id, msg), 100)
        self.create_subscription(PoseStamped, f"/unity_eval/{object_id}_visual_pose",
                                 lambda msg: self.on_pose("object", "unity", object_id, msg), 100)
        for arm in ("left", "right"):
            self.create_subscription(ReceivedPoseStates, f"/{arm}_arm/received_pose_states",
                                     lambda msg, arm=arm: self.on_received(arm, msg), 100)
            self.create_subscription(PoseStamped, f"/{arm}_arm/teleop/actual_ee_pose",
                                     lambda msg, arm=arm: self.on_pose("ee", "sim", arm, msg), 100)
            self.create_subscription(PoseStamped, f"/unity_eval/{arm}_tool0_visual_pose",
                                     lambda msg, arm=arm: self.on_pose("ee", "unity", arm, msg), 100)
            self.create_subscription(PoseStamped, f"/unity_eval/{arm}_wrist_3_visual_pose",
                                     lambda msg, arm=arm: self.on_pose("ee", "unity_wrist_3", arm, msg), 100)
            self.create_subscription(PoseStamped, f"/unity_eval/{arm}_hande_end_visual_pose",
                                     lambda msg, arm=arm: self.on_pose("ee", "unity_hande_end", arm, msg), 100)

    def rel(self, t=None):
        return (time.monotonic() if t is None else t) - self.start_time

    def trace_window_open(self):
        now = time.monotonic()
        if self.first_recording_time is None:
            return False
        if self.tail_deadline is None:
            return True
        return now <= self.tail_deadline

    def update_recording_window(self, active, t, source, payload):
        if active:
            if self.first_recording_time is None:
                self.first_recording_time = t
            self.last_recording_time = t
            self.tail_deadline = None
        elif self.first_recording_time is not None and self.tail_deadline is None:
            self.tail_deadline = t + post_recording_tail_sec

    def maybe_finish(self):
        now = time.monotonic()
        if self.first_recording_time is None and now - self.start_time >= max_wait_sec:
            self.done_reason = "max_wait_elapsed_without_recording"
        elif self.tail_deadline is not None and now >= self.tail_deadline:
            self.done_reason = "post_recording_tail_complete"
        elif now - self.start_time >= max_duration_sec:
            self.done_reason = "max_duration_elapsed"

    def record_topic_time(self, stream, t):
        if not self.trace_window_open():
            return
        self.topic_times.setdefault(stream, []).append(t)

    def on_recording_state(self, msg):
        t = time.monotonic()
        payload = parse_json_payload(msg)
        active = bool(payload.get("recording_active", False))
        self.update_recording_window(active, t, "recording_state", payload)
        row = dict(payload)
        row.update({"time_sec": self.rel(t), "source": "recording_state"})
        self.recording_events.append(row)

    def on_fps_sample(self, msg):
        t = time.monotonic()
        payload = parse_json_payload(msg)
        active = bool(payload.get("recording_active", False))
        self.update_recording_window(active, t, "fps_sample", payload)
        if not self.trace_window_open():
            return
        if len(self.fps_samples) < fps_samples_limit:
            row = dict(payload)
            row.update({"time_sec": self.rel(t), "source": "fps_sample"})
            self.fps_samples.append(row)
        self.record_topic_time("unity_eval:fps_sample", t)

    def on_received(self, arm, msg):
        if not self.trace_window_open():
            return
        t = time.monotonic()
        stream = f"{arm}:hand_input"
        self.record_topic_time(stream, t)
        p = msg.pose.position
        current = (float(p.x), float(p.y), float(p.z))
        prev = self.previous_pose.get(stream)
        prev_t = self.previous_time.get(stream)
        speed = pose_speed(prev, current, t - prev_t if prev_t is not None else 0.0)
        self.previous_pose[stream] = current
        self.previous_time[stream] = t
        if len(self.pose_samples) < pose_samples_limit:
            self.pose_samples.append({
                "time_sec": self.rel(t), "stream": stream, "source": "ros", "label": arm,
                "speed_mps": speed, "tracked": bool(msg.tracked), "teleop_enable": bool(msg.teleop_enable),
                "x": current[0], "y": current[1], "z": current[2],
                "qx": float(msg.pose.orientation.x), "qy": float(msg.pose.orientation.y),
                "qz": float(msg.pose.orientation.z), "qw": float(msg.pose.orientation.w),
            })
        if bool(msg.tracked) and bool(msg.teleop_enable):
            if speed >= hand_motion_threshold_mps and t - self.last_hand_event_time[arm] >= event_cooldown_sec:
                event = {
                    "event_id": f"{arm}_ee_{len(self.motion_events) + 1}",
                    "event_type": "recording_hand_to_ee_visual",
                    "arm": arm,
                    "input_start_sec": self.rel(t),
                    "input_speed_mps": speed,
                }
                self.motion_events.append(event)
                self.open_arm_events[arm].append((t, event))
                self.last_hand_event_time[arm] = t

    def on_pose(self, kind, source, label, msg):
        if not self.trace_window_open():
            return
        t = time.monotonic()
        pose = pose_tuple(msg)
        frame_id = msg.header.frame_id
        original_frame_id = msg.header.frame_id
        if kind == "ee" and source == "sim":
            pose = transform_base_pose_to_world(pose, self.robot_spawn_transforms.get(label))
            frame_id = "world"
        stream = f"{kind}:{source}:{label}"
        self.record_topic_time(stream, t)
        prev = self.previous_pose.get(stream)
        prev_t = self.previous_time.get(stream)
        speed = pose_speed(prev, pose[:3], t - prev_t if prev_t is not None else 0.0)
        self.previous_pose[stream] = pose[:3]
        self.previous_time[stream] = t
        self.latest[stream] = (t, pose, frame_id, stamp_to_sec(msg.header.stamp), speed)
        if len(self.pose_samples) < pose_samples_limit:
            self.pose_samples.append({
                "time_sec": self.rel(t), "stream": stream, "source": source, "label": label,
                "frame_id": frame_id, "original_frame_id": original_frame_id,
                "source_stamp_sec": stamp_to_sec(msg.header.stamp), "speed_mps": speed,
                "x": pose[0], "y": pose[1], "z": pose[2],
                "qx": pose[3], "qy": pose[4], "qz": pose[5], "qw": pose[6],
            })
        self.maybe_record_alignment(kind, label)
        self.maybe_record_motion_event(kind, source, label, t, speed)

    def maybe_record_alignment(self, kind, label):
        if len(self.alignment_samples) >= alignment_samples_limit:
            return
        if kind == "object":
            source_key = f"object:gazebo:{label}"
            visual_key = f"object:unity:{label}"
            pair_id = f"{label}_gazebo_to_unity_visual"
            self.record_alignment_pair(kind, label, source_key, visual_key, pair_id)
            return

        source_key = f"ee:sim:{label}"
        visual_pairs = (
            ("unity", "tool0"),
            ("unity_wrist_3", "wrist_3"),
            ("unity_hande_end", "hande_end"),
        )
        for visual_source, visual_label in visual_pairs:
            visual_key = f"ee:{visual_source}:{label}"
            pair_id = f"{label}_ee_sim_to_unity_{visual_label}_visual"
            self.record_alignment_pair(kind, label, source_key, visual_key, pair_id)

    def record_alignment_pair(self, kind, label, source_key, visual_key, pair_id):
        if len(self.alignment_samples) >= alignment_samples_limit:
            return
        if source_key not in self.latest or visual_key not in self.latest:
            return
        source_t, source_pose, source_frame, source_stamp, source_speed = self.latest[source_key]
        visual_t, visual_pose, visual_frame, visual_stamp, visual_speed = self.latest[visual_key]
        pair_age = visual_t - source_t
        if abs(pair_age) > max_pair_age_sec:
            return
        dx = visual_pose[0] - source_pose[0]
        dy = visual_pose[1] - source_pose[1]
        dz = visual_pose[2] - source_pose[2]
        self.alignment_samples.append({
            "time_sec": self.rel(max(source_t, visual_t)), "pair_id": pair_id, "label": label,
            "kind": kind, "source_frame_id": source_frame, "visual_frame_id": visual_frame,
            "source_stamp_sec": source_stamp, "visual_unity_stamp_sec": visual_stamp,
            "source_speed_mps": source_speed, "visual_speed_mps": visual_speed,
            "pair_age_ms": pair_age * 1000.0,
            "position_error_m": vec_norm([dx, dy, dz]),
            "position_error_x_m": dx, "position_error_y_m": dy, "position_error_z_m": dz,
            "orientation_error_deg": quat_angle_deg(source_pose, visual_pose),
            "source_x": source_pose[0], "source_y": source_pose[1], "source_z": source_pose[2],
            "visual_x": visual_pose[0], "visual_y": visual_pose[1], "visual_z": visual_pose[2],
        })

    def maybe_record_motion_event(self, kind, source, label, t, speed):
        if kind == "object":
            if source == "gazebo" and speed >= object_motion_threshold_mps and t - self.last_object_event_time >= event_cooldown_sec:
                event = {
                    "event_id": f"object_{len(self.motion_events) + 1}",
                    "event_type": "recording_gazebo_object_to_mr_visual",
                    "object_id": label,
                    "gazebo_object_start_sec": self.rel(t),
                    "gazebo_object_speed_mps": speed,
                }
                self.motion_events.append(event)
                self.open_object_events.append((t, event))
                self.last_object_event_time = t
            elif source == "unity" and speed >= object_motion_threshold_mps:
                self.record_object_visual_stage(t, speed)
            return
        if kind != "ee":
            return
        arm = label
        while self.open_arm_events[arm] and t - self.open_arm_events[arm][0][0] > event_expire_sec:
            self.open_arm_events[arm].popleft()
        if source == "sim" and speed >= ee_motion_threshold_mps:
            self.record_arm_stage(arm, "sim_ee", t, speed)
        elif source == "unity" and speed >= ee_motion_threshold_mps:
            self.record_arm_stage(arm, "unity_ee_visual", t, speed)

    def record_arm_stage(self, arm, stage, t, speed):
        for input_t, event in self.open_arm_events[arm]:
            key = f"{stage}_start_sec"
            if key in event:
                continue
            event[key] = self.rel(t)
            event[f"{stage}_speed_mps"] = speed
            event[f"input_to_{stage}_latency_ms"] = (t - input_t) * 1000.0
            if stage == "unity_ee_visual" and "sim_ee_start_sec" in event:
                event["sim_ee_to_unity_visual_latency_ms"] = (
                    event["unity_ee_visual_start_sec"] - event["sim_ee_start_sec"]
                ) * 1000.0
            break

    def record_object_visual_stage(self, t, speed):
        while self.open_object_events and t - self.open_object_events[0][0] > event_expire_sec:
            self.open_object_events.popleft()
        for object_t, event in self.open_object_events:
            if "unity_object_visual_start_sec" in event:
                continue
            event["unity_object_visual_start_sec"] = self.rel(t)
            event["unity_object_speed_mps"] = speed
            event["gazebo_object_to_unity_visual_latency_ms"] = (t - object_t) * 1000.0
            break

    def topic_rate_rows(self):
        rows = []
        for stream, times in sorted(self.topic_times.items()):
            gaps = [b - a for a, b in zip(times, times[1:])]
            hz = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 and times[-1] > times[0] else 0.0
            rows.append({
                "stream": stream,
                "messages": len(times),
                "mean_hz": hz,
                "min_gap_sec": min(gaps) if gaps else None,
                "max_gap_sec": max(gaps) if gaps else None,
            })
        return rows

    def fps_summary(self):
        values = []
        for row in self.fps_samples:
            try:
                values.append(float(row.get("fps")))
            except Exception:
                pass
        if not values:
            return {"samples": 0, "mean_fps": None, "min_fps": None, "max_fps": None}
        return {"samples": len(values), "mean_fps": sum(values) / len(values), "min_fps": min(values), "max_fps": max(values)}

    def alignment_summary(self):
        buckets = {}
        for row in self.alignment_samples:
            bucket = buckets.setdefault(row["pair_id"], {
                "pair_id": row["pair_id"],
                "samples": 0,
                "pos": [],
                "rot": [],
                "age": [],
                "source_speed": [],
                "visual_speed": [],
                "still_pos": [],
                "moving_pos": [],
            })
            bucket["samples"] += 1
            position_error = float(row["position_error_m"])
            source_speed = float(row.get("source_speed_mps", 0.0) or 0.0)
            visual_speed = float(row.get("visual_speed_mps", 0.0) or 0.0)
            max_speed = max(source_speed, visual_speed)
            bucket["pos"].append(position_error)
            bucket["rot"].append(float(row["orientation_error_deg"]))
            bucket["age"].append(float(row["pair_age_ms"]))
            bucket["source_speed"].append(source_speed)
            bucket["visual_speed"].append(visual_speed)
            if max_speed <= 0.001:
                bucket["still_pos"].append(position_error)
            if max_speed >= 0.01:
                bucket["moving_pos"].append(position_error)
        rows = []
        for bucket in buckets.values():
            def stats(values):
                if not values:
                    return (None, None, None)
                return (sum(values) / len(values), min(values), max(values))
            pos_mean, pos_min, pos_max = stats(bucket["pos"])
            rot_mean, rot_min, rot_max = stats(bucket["rot"])
            age_mean, age_min, age_max = stats(bucket["age"])
            source_speed_mean, _, source_speed_max = stats(bucket["source_speed"])
            visual_speed_mean, _, visual_speed_max = stats(bucket["visual_speed"])
            still_mean, still_min, still_max = stats(bucket["still_pos"])
            moving_mean, moving_min, moving_max = stats(bucket["moving_pos"])
            rows.append({
                "pair_id": bucket["pair_id"], "samples": bucket["samples"],
                "mean_position_error_m": pos_mean, "min_position_error_m": pos_min, "max_position_error_m": pos_max,
                "still_samples": len(bucket["still_pos"]),
                "mean_still_position_error_m": still_mean,
                "min_still_position_error_m": still_min,
                "max_still_position_error_m": still_max,
                "moving_samples": len(bucket["moving_pos"]),
                "mean_moving_position_error_m": moving_mean,
                "min_moving_position_error_m": moving_min,
                "max_moving_position_error_m": moving_max,
                "mean_orientation_error_deg": rot_mean, "min_orientation_error_deg": rot_min, "max_orientation_error_deg": rot_max,
                "mean_pair_age_ms": age_mean, "min_pair_age_ms": age_min, "max_pair_age_ms": age_max,
                "mean_source_speed_mps": source_speed_mean,
                "max_source_speed_mps": source_speed_max,
                "mean_visual_speed_mps": visual_speed_mean,
                "max_visual_speed_mps": visual_speed_max,
            })
        return rows

    def summary(self):
        diagnostic = None
        if self.first_recording_time is None and not self.recording_events and not self.fps_samples:
            diagnostic = (
                "No messages were received on /unity_eval/recording_state or /unity_eval/fps_sample. "
                "Verify the deployed Quest build includes RecordingPerformanceTraceLogger and that ROS-TCP is connected."
            )
        return {
            "done_reason": self.done_reason,
            "max_wait_sec": max_wait_sec,
            "max_duration_sec": max_duration_sec,
            "post_recording_tail_sec": post_recording_tail_sec,
            "object_id": object_id,
            "diagnostic": diagnostic,
            "recording_started_sec": self.rel(self.first_recording_time) if self.first_recording_time is not None else None,
            "last_recording_seen_sec": self.rel(self.last_recording_time) if self.last_recording_time is not None else None,
            "fps_summary": self.fps_summary(),
            "topic_rates": self.topic_rate_rows(),
            "alignment_summary": self.alignment_summary(),
            "motion_events": self.motion_events,
            "note": "Run this before pressing record. It logs Unity FPS samples from /unity_eval/fps_sample and sync/latency from ROS arrival times until 15 seconds after recording stops by default.",
        }


rclpy.init()
node = RecordingFpsSyncTrace()
while rclpy.ok() and node.done_reason is None:
    rclpy.spin_once(node, timeout_sec=0.01)
    node.maybe_finish()

result = node.summary()
result["recording_events"] = node.recording_events
result["fps_samples"] = node.fps_samples
result["pose_samples"] = node.pose_samples
result["alignment_samples"] = node.alignment_samples
print(json.dumps(result))
node.destroy_node()
rclpy.shutdown()
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-wait-sec", type=float, default=120.0, help="How long to wait for recording to start.")
    parser.add_argument("--max-duration-sec", type=float, default=240.0, help="Hard cap for the whole script.")
    parser.add_argument("--post-recording-tail-sec", type=float, default=15.0, help="Continue logging this long after recording stops.")
    parser.add_argument("--object-id", default="Sync_RedCube")
    parser.add_argument("--robot-profile", default="/workspace/profiles/robots/ur5e_hande_dual/robot.yaml")
    parser.add_argument("--max-pair-age-sec", type=float, default=0.10)
    parser.add_argument("--hand-motion-threshold-mps", type=float, default=0.08)
    parser.add_argument("--ee-motion-threshold-mps", type=float, default=0.015)
    parser.add_argument("--object-motion-threshold-mps", type=float, default=0.010)
    parser.add_argument("--event-cooldown-sec", type=float, default=0.50)
    parser.add_argument("--event-expire-sec", type=float, default=2.0)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()

    require_container()
    out = results_dir("recording_fps_sync_latency_trace", args.output_root)
    env = (
        f"EVAL_MAX_WAIT_SEC={args.max_wait_sec} "
        f"EVAL_MAX_DURATION_SEC={args.max_duration_sec} "
        f"EVAL_POST_RECORDING_TAIL_SEC={args.post_recording_tail_sec} "
        f"EVAL_OBJECT_ID={args.object_id} "
        f"EVAL_ROBOT_PROFILE={args.robot_profile} "
        f"EVAL_MAX_PAIR_AGE_SEC={args.max_pair_age_sec} "
        f"EVAL_HAND_MOTION_THRESHOLD_MPS={args.hand_motion_threshold_mps} "
        f"EVAL_EE_MOTION_THRESHOLD_MPS={args.ee_motion_threshold_mps} "
        f"EVAL_OBJECT_MOTION_THRESHOLD_MPS={args.object_motion_threshold_mps} "
        f"EVAL_EVENT_COOLDOWN_SEC={args.event_cooldown_sec} "
        f"EVAL_EVENT_EXPIRE_SEC={args.event_expire_sec} "
        "python3 -u -"
    )
    proc = docker_ros_popen(env, input_text=TRACE_SCRIPT)
    timeout = max(30.0, args.max_duration_sec + 20.0)
    stdout, stderr = proc.communicate(timeout=timeout)
    (out / "stderr.txt").write_text(stderr or "", encoding="utf-8")
    if proc.returncode != 0:
        (out / "stdout.txt").write_text(stdout or "", encoding="utf-8")
        raise SystemExit(f"Recording FPS/sync trace failed with code {proc.returncode}; see {out}")

    data = json.loads((stdout or "{}").strip().splitlines()[-1])
    write_json(out / "summary.json", {k: v for k, v in data.items() if k not in ("fps_samples", "pose_samples", "alignment_samples", "recording_events")})
    write_csv(out / "recording_events.csv", data.get("recording_events", []))
    write_csv(out / "recording_fps_samples.csv", data.get("fps_samples", []))
    write_csv(out / "topic_rates.csv", data.get("topic_rates", []))
    write_csv(out / "motion_events.csv", data.get("motion_events", []))
    write_csv(out / "pose_samples.csv", data.get("pose_samples", []))
    write_csv(out / "pose_alignment_samples.csv", data.get("alignment_samples", []))
    write_csv(out / "pose_alignment_summary.csv", data.get("alignment_summary", []))
    print(json.dumps({k: v for k, v in data.items() if k not in ("fps_samples", "pose_samples", "alignment_samples", "recording_events")}, indent=2))
    print_done(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
