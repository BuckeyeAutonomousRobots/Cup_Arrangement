#!/usr/bin/env python3
"""Run synthetic hand input and audit receiver/mapper rates without the headset."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from eval_common import BACKEND_ROOT, results_dir, run, write_json, print_done


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--arm", default="both", choices=["left", "right", "both"])
    parser.add_argument("--pattern", default="line_x", choices=["line_x", "line_y", "line_z", "circle_xy", "circle_xz"])
    parser.add_argument("--amplitude", type=float, default=0.10)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    out = results_dir("synthetic_hand_receiver", args.output_root)

    env = {
        "DEBUG_ARM": args.arm,
        "DEBUG_PATTERN": args.pattern,
        "DEBUG_DURATION_SEC": str(args.duration),
        "DEBUG_AMPLITUDE_X": str(args.amplitude),
    }
    print("[info] starting debug hand generator; this stops quest_controller_receiver to avoid competing input")
    proc = run(["./scripts/backend11_lifecycle.sh", "debug_hand_start"], cwd=BACKEND_ROOT, env=env, check=False)
    (out / "debug_hand_start.stdout.txt").write_text(proc.stdout or "")
    (out / "debug_hand_start.stderr.txt").write_text(proc.stderr or "")
    if proc.returncode != 0:
        raise SystemExit(f"debug_hand_start failed; see {out}")

    topics = ",".join([
        "/left_arm/received_pose_states",
        "/right_arm/received_pose_states",
        "/left_arm/target_twist_states",
        "/right_arm/target_twist_states",
    ])
    audit = subprocess.run([
        sys.executable,
        str(Path(__file__).with_name("03_ros_topic_rate_audit.py")),
        "--duration", str(args.duration),
        "--topics", topics,
        "--output-root", str(out),
    ], cwd=str(BACKEND_ROOT), text=True, capture_output=True, check=False)
    (out / "topic_audit.stdout.txt").write_text(audit.stdout or "")
    (out / "topic_audit.stderr.txt").write_text(audit.stderr or "")

    stop = run(["./scripts/backend11_lifecycle.sh", "debug_hand_stop"], cwd=BACKEND_ROOT, check=False)
    (out / "debug_hand_stop.stdout.txt").write_text(stop.stdout or "")
    summary = {"duration_sec": args.duration, "arm": args.arm, "pattern": args.pattern, "topic_audit_returncode": audit.returncode}
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    print_done(out)
    print("[note] Restore live headset input with: ./scripts/backend11_lifecycle.sh start_receiver")
    return audit.returncode


if __name__ == "__main__":
    raise SystemExit(main())
