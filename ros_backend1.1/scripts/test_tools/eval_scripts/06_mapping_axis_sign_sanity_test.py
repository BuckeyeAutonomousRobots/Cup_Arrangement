#!/usr/bin/env python3
"""Send synthetic +X/+Y/+Z motions and summarize whether backend output responds."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from eval_common import BACKEND_ROOT, results_dir, run, write_json, print_done

AXES = {
    "x": {"pattern": "line_x", "amp_x": "0.10", "amp_y": "0.0", "amp_z": "0.0"},
    "y": {"pattern": "line_y", "amp_x": "0.10", "amp_y": "0.10", "amp_z": "0.0"},
    "z": {"pattern": "line_z", "amp_x": "0.10", "amp_y": "0.0", "amp_z": "0.10"},
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-per-axis", type=float, default=8.0)
    parser.add_argument("--arm", default="both", choices=["left", "right", "both"])
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    out = results_dir("mapping_axis_sign", args.output_root)
    results = []
    for axis, cfg in AXES.items():
        axis_dir = out / axis
        axis_dir.mkdir(parents=True, exist_ok=True)
        env = {
            "DEBUG_ARM": args.arm,
            "DEBUG_PATTERN": cfg["pattern"],
            "DEBUG_DURATION_SEC": str(args.duration_per_axis + 2.0),
            "DEBUG_SAMPLE_DURATION_SEC": str(args.duration_per_axis),
            "DEBUG_AMPLITUDE_X": cfg["amp_x"],
            "DEBUG_AMPLITUDE_Y": cfg["amp_y"],
            "DEBUG_AMPLITUDE_Z": cfg["amp_z"],
        }
        proc = run(["./scripts/backend11_lifecycle.sh", "debug_servo_motion"], cwd=BACKEND_ROOT, env=env, check=False, timeout=args.duration_per_axis + 35.0)
        (axis_dir / "stdout.txt").write_text(proc.stdout or "")
        (axis_dir / "stderr.txt").write_text(proc.stderr or "")
        results.append({"axis": axis, "pattern": cfg["pattern"], "returncode": proc.returncode})
    write_json(out / "summary.json", {"arm": args.arm, "results": results, "note": "Inspect stdout files for max target/cmd response. This is an automated sanity test, not a human-perceived direction test."})
    print(json.dumps(results, indent=2))
    print_done(out)
    return 0 if all(r["returncode"] == 0 for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
