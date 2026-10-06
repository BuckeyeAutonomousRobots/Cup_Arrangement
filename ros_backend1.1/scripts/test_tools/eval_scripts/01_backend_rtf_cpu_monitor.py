#!/usr/bin/env python3
"""Measure Gazebo real-time factor from /clock plus Docker CPU/memory."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from eval_common import add_common_args, docker_ros_popen, print_done, require_container, results_dir, sample_docker_stats, summarize, write_csv, write_json

CLOCK_MONITOR = r'''
import json, os, time
import rclpy
from rclpy.node import Node
from rosgraph_msgs.msg import Clock

duration = float(os.environ.get("EVAL_DURATION", "30"))

class ClockMonitor(Node):
    def __init__(self):
        super().__init__("eval_clock_monitor")
        self.samples = []
        self.first_wall = None
        self.first_sim = None
        self.last_wall = None
        self.last_sim = None
        self.create_subscription(Clock, "/clock", self.on_clock, 50)
        self.deadline = time.monotonic() + duration

    def on_clock(self, msg):
        wall = time.monotonic()
        sim = msg.clock.sec + msg.clock.nanosec * 1e-9
        if self.first_wall is None:
            self.first_wall = wall
            self.first_sim = sim
            self.last_wall = wall
            self.last_sim = sim
            return
        dt_wall = wall - self.last_wall if self.last_wall is not None else 0.0
        dt_sim = sim - self.last_sim if self.last_sim is not None else 0.0
        inst = dt_sim / dt_wall if dt_wall > 1e-9 else None
        total = (sim - self.first_sim) / (wall - self.first_wall) if wall > self.first_wall else None
        self.samples.append({"wall_mono": wall, "sim_time": sim, "rtf_instant": inst, "rtf_total": total})
        self.last_wall = wall
        self.last_sim = sim

rclpy.init()
node = ClockMonitor()
try:
    while rclpy.ok() and time.monotonic() < node.deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
finally:
    print(json.dumps({"samples": node.samples}))
    node.destroy_node()
    rclpy.shutdown()
'''


def window_rtf_rows(clock_rows: list[dict], window_sec: float) -> list[dict]:
    """Aggregate noisy per-clock intervals into stable windowed RTF samples."""
    if not clock_rows:
        return []
    rows = sorted(
        (row for row in clock_rows if isinstance(row.get("wall_mono"), (int, float)) and isinstance(row.get("sim_time"), (int, float))),
        key=lambda row: row["wall_mono"],
    )
    if len(rows) < 2:
        return []
    window_sec = max(0.1, window_sec)
    out: list[dict] = []
    first_wall = rows[0]["wall_mono"]
    bucket: list[dict] = []
    bucket_idx = 0

    def flush_bucket(samples: list[dict], idx: int) -> None:
        if len(samples) < 2:
            return
        start = samples[0]
        end = samples[-1]
        wall_dt = end["wall_mono"] - start["wall_mono"]
        sim_dt = end["sim_time"] - start["sim_time"]
        if wall_dt <= 0:
            return
        out.append({
            "window_index": idx,
            "wall_start_mono": start["wall_mono"],
            "wall_end_mono": end["wall_mono"],
            "wall_dt": wall_dt,
            "sim_dt": sim_dt,
            "rtf_window": sim_dt / wall_dt,
            "clock_messages": len(samples),
            "clock_hz": len(samples) / wall_dt,
        })

    for row in rows:
        idx = int((row["wall_mono"] - first_wall) // window_sec)
        if bucket and idx != bucket_idx:
            flush_bucket(bucket, bucket_idx)
            bucket = []
            bucket_idx = idx
        if not bucket:
            bucket_idx = idx
        bucket.append(row)
    flush_bucket(bucket, bucket_idx)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--rtf-window-sec", type=float, default=1.0, help="Window size for stable RTF min/max summaries.")
    args = parser.parse_args()
    require_container()
    out = results_dir("backend_rtf_cpu", args.output_root)

    proc = docker_ros_popen(f"EVAL_DURATION={args.duration} python3 -u -", input_text=CLOCK_MONITOR)
    cpu_rows = sample_docker_stats(args.duration, args.interval)
    stdout, stderr = proc.communicate(timeout=max(10.0, args.duration + 10.0))
    if proc.returncode != 0:
        (out / "clock_monitor.stderr.txt").write_text(stderr or "")
        raise SystemExit(f"clock monitor failed with code {proc.returncode}; see {out}")

    clock_data = json.loads((stdout or "{}").strip().splitlines()[-1])
    clock_rows = clock_data.get("samples", [])
    window_rows = window_rtf_rows(clock_rows, args.rtf_window_sec)
    write_csv(out / "clock_rtf.csv", clock_rows)
    write_csv(out / "clock_rtf_windowed.csv", window_rows)
    write_csv(out / "docker_cpu_memory.csv", cpu_rows)

    rtf_values = [row["rtf_instant"] for row in clock_rows if isinstance(row.get("rtf_instant"), (int, float))]
    rtf_window_values = [row["rtf_window"] for row in window_rows if isinstance(row.get("rtf_window"), (int, float))]
    cpu_values = [row["cpu_percent"] for row in cpu_rows if isinstance(row.get("cpu_percent"), (int, float))]
    summary = {
        "duration_sec": args.duration,
        "clock_messages": len(clock_rows),
        "rtf_instant": summarize(rtf_values),
        "rtf_window_sec": args.rtf_window_sec,
        "rtf_window": summarize(rtf_window_values),
        "cpu_percent": summarize(cpu_values),
        "cpu_samples": len(cpu_rows),
    }
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    print_done(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
