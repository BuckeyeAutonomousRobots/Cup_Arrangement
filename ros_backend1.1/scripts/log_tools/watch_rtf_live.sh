#!/usr/bin/env bash
set -euo pipefail

CONTAINER="${CONTAINER:-motion_planner_11}"
ROS_ENV="${ROS_ENV:-source /opt/ros/humble/setup.bash && source /workspace/install/setup.bash}"
WINDOW_SEC="${WINDOW_SEC:-1.0}"

if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER}"; then
  echo "[error] Container ${CONTAINER} is not running." >&2
  echo "Start it first, for example:" >&2
  echo "  cd ros_backend1.1 && ./scripts/backend11_lifecycle.sh bringup_dual" >&2
  exit 1
fi

docker exec -i "${CONTAINER}" bash -lc "${ROS_ENV} && WINDOW_SEC='${WINDOW_SEC}' python3 -u -" <<'PY'
import os
import time

import rclpy
from rclpy.node import Node
from rosgraph_msgs.msg import Clock

window_sec = max(0.1, float(os.environ.get("WINDOW_SEC", "1.0")))

class RtfMonitor(Node):
    def __init__(self):
        super().__init__("live_rtf_monitor")
        self.first_wall = None
        self.first_sim = None
        self.window_wall = None
        self.window_sim = None
        self.window_count = 0
        self.total_count = 0
        self.latest_sim = None
        self.rtf_min = None
        self.rtf_max = None
        self.create_subscription(Clock, "/clock", self.on_clock, 100)
        self.last_print = time.monotonic()

    def on_clock(self, msg):
        wall = time.monotonic()
        sim = msg.clock.sec + msg.clock.nanosec * 1e-9
        if self.first_wall is None:
            self.first_wall = wall
            self.first_sim = sim
            self.window_wall = wall
            self.window_sim = sim
            self.latest_sim = sim
            return
        self.latest_sim = sim
        self.window_count += 1
        self.total_count += 1

    def maybe_print(self):
        now = time.monotonic()
        if self.first_wall is None or now - self.last_print < window_sec:
            return
        wall_dt = now - self.window_wall
        sim_dt = self.latest_sim - self.window_sim
        total_wall = now - self.first_wall
        total_sim = self.latest_sim - self.first_sim
        inst_rtf = sim_dt / wall_dt if wall_dt > 0 else 0.0
        total_rtf = total_sim / total_wall if total_wall > 0 else 0.0
        hz = self.window_count / wall_dt if wall_dt > 0 else 0.0
        if self.rtf_min is None or inst_rtf < self.rtf_min:
            self.rtf_min = inst_rtf
        if self.rtf_max is None or inst_rtf > self.rtf_max:
            self.rtf_max = inst_rtf
        print(
            f"wall_dt={wall_dt:5.2f}s sim_dt={sim_dt:6.3f}s "
            f"rtf={inst_rtf:5.3f} min={self.rtf_min:5.3f} max={self.rtf_max:5.3f} "
            f"total_rtf={total_rtf:5.3f} clock_hz={hz:7.1f}",
            flush=True,
        )
        self.window_wall = now
        self.window_sim = self.latest_sim
        self.window_count = 0
        self.last_print = now

rclpy.init()
node = RtfMonitor()
try:
    print("Live RTF monitor for /clock. Press Ctrl-C to stop.", flush=True)
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.05)
        node.maybe_print()
except KeyboardInterrupt:
    pass
finally:
    node.destroy_node()
    rclpy.shutdown()
PY
