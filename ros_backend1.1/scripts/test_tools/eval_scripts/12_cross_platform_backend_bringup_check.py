#!/usr/bin/env python3
"""Collect host/container/backend facts for portability checks across macOS, Ubuntu, and Windows/WSL."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess

from eval_common import BACKEND_ROOT, CONTAINER, print_done, results_dir, run, write_json


def run_text(cmd: list[str], timeout: float = 20.0, cwd=BACKEND_ROOT) -> dict:
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True, timeout=timeout, check=False)
    return {"cmd": cmd, "returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bringup", action="store_true", help="Run bringup_dual before collecting status. Default is status-only.")
    parser.add_argument("--topic-audit", action="store_true", help="Also run a short topic-rate audit after status.")
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    out = results_dir("cross_platform_bringup_check", args.output_root)

    report = {
        "host": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "docker_on_path": shutil.which("docker"),
            "adb_on_path": shutil.which("adb"),
        },
        "commands": {},
        "container": CONTAINER,
    }
    if args.bringup:
        report["commands"]["bringup_dual"] = run_text(["./scripts/backend11_lifecycle.sh", "bringup_dual"], timeout=180.0)
    report["commands"]["docker_version"] = run_text(["docker", "version"], timeout=30.0)
    report["commands"]["docker_ps"] = run_text(["docker", "ps"], timeout=20.0)
    report["commands"]["lifecycle_status"] = run_text(["./scripts/backend11_lifecycle.sh", "status"], timeout=45.0)
    report["commands"]["adb_devices"] = run_text(["adb", "devices"], timeout=20.0) if shutil.which("adb") else {"skipped": "adb not on PATH"}
    if args.topic_audit:
        report["commands"]["topic_audit"] = run_text([
            "python3", "scripts/test_tools/eval_scripts/03_ros_topic_rate_audit.py", "--duration", "8", "--output-root", str(out)
        ], timeout=30.0)

    ok = report["commands"].get("lifecycle_status", {}).get("returncode") == 0
    report["ok"] = bool(ok)
    write_json(out / "platform_bringup_report.json", report)
    print(json.dumps({"ok": ok, "platform": report["host"], "output": str(out)}, indent=2))
    print_done(out)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
