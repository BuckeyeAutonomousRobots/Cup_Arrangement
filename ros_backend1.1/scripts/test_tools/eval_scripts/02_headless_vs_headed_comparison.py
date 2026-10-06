#!/usr/bin/env python3
"""Compare RTF/CPU for headless and headed Gazebo modes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from eval_common import BACKEND_ROOT, results_dir, run, write_json, print_done


def run_condition(mode: str, duration: float, interval: float, out_root: Path, warmup: float) -> dict:
    headless = "1" if mode == "headless" else "0"
    print(f"[info] starting dual sim mode={mode} SIM_HEADLESS={headless}")
    run(["env", f"SIM_HEADLESS={headless}", "./scripts/backend11_lifecycle.sh", "start_dual_sim"], cwd=BACKEND_ROOT, check=True, capture=False)
    if warmup > 0:
        run(["sleep", str(warmup)], cwd=BACKEND_ROOT, check=True, capture=False)
    cmd = [sys.executable, str(Path(__file__).with_name("01_backend_rtf_cpu_monitor.py")), "--duration", str(duration), "--interval", str(interval), "--output-root", str(out_root)]
    proc = subprocess.run(cmd, cwd=str(BACKEND_ROOT), text=True, capture_output=True, check=False)
    (out_root / f"{mode}_monitor.stdout.txt").write_text(proc.stdout or "")
    (out_root / f"{mode}_monitor.stderr.txt").write_text(proc.stderr or "")
    if proc.returncode != 0:
        return {"mode": mode, "ok": False, "returncode": proc.returncode}
    summaries = sorted(out_root.glob("*_backend_rtf_cpu/summary.json"), key=lambda p: p.stat().st_mtime)
    summary = json.loads(summaries[-1].read_text()) if summaries else {}
    summary.update({"mode": mode, "ok": True})
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--warmup", type=float, default=8.0)
    parser.add_argument("--modes", default="headless,headed", help="Comma list: headless,headed. noVNC-connected is a manual headed subcase.")
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    out = results_dir("headless_vs_headed", args.output_root)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    results = [run_condition(mode, args.duration, args.interval, out, args.warmup) for mode in modes]
    write_json(out / "comparison_summary.json", {"conditions": results, "note": "For a noVNC-connected condition, run headed mode while a browser is connected to noVNC and label it manually."})
    print(json.dumps(results, indent=2))
    print_done(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
