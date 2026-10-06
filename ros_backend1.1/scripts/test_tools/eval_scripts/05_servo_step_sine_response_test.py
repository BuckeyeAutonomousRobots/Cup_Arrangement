#!/usr/bin/env python3
"""Use synthetic hand motion plus Servo sampler to quantify backend response."""

from __future__ import annotations

import argparse
import json

from eval_common import BACKEND_ROOT, results_dir, run, write_json, print_done


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--arm", default="both", choices=["left", "right", "both"])
    parser.add_argument("--pattern", default="line_x", choices=["line_x", "line_y", "line_z", "circle_xy", "circle_xz"])
    parser.add_argument("--amplitude", type=float, default=0.12)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    out = results_dir("servo_response", args.output_root)
    env = {
        "DEBUG_ARM": args.arm,
        "DEBUG_PATTERN": args.pattern,
        "DEBUG_DURATION_SEC": str(args.duration + 2.0),
        "DEBUG_SAMPLE_DURATION_SEC": str(args.duration),
        "DEBUG_AMPLITUDE_X": str(args.amplitude),
    }
    proc = run(["./scripts/backend11_lifecycle.sh", "debug_servo_motion"], cwd=BACKEND_ROOT, env=env, check=False, timeout=args.duration + 30.0)
    (out / "debug_servo_motion.stdout.txt").write_text(proc.stdout or "")
    (out / "debug_servo_motion.stderr.txt").write_text(proc.stderr or "")
    summary = {"duration_sec": args.duration, "arm": args.arm, "pattern": args.pattern, "returncode": proc.returncode}
    write_json(out / "summary.json", summary)
    print(proc.stdout or "")
    print_done(out)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
