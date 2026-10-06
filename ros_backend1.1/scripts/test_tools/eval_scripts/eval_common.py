#!/usr/bin/env python3
"""Shared helpers for backend evaluation scripts."""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_ROOT = SCRIPT_DIR.parents[2]
REPO_ROOT = BACKEND_ROOT.parent
CONTAINER = os.environ.get("CONTAINER", "motion_planner_11")
ROS_ENV = "source /opt/ros/humble/setup.bash && source /workspace/install/setup.bash"


def timestamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def results_dir(test_name: str, root: str | None = None) -> Path:
    base = Path(root or os.environ.get("EVAL_RESULTS_DIR", BACKEND_ROOT / "eval_results"))
    path = base / f"{timestamp()}_{safe_name(test_name)}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("_") or "eval"


def run(cmd: list[str], *, cwd: Path | None = None, check: bool = True, timeout: float | None = None,
        input_text: str | None = None, capture: bool = True, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.run(
        cmd,
        cwd=str(cwd or BACKEND_ROOT),
        input=input_text,
        text=True,
        capture_output=capture,
        timeout=timeout,
        check=check,
        env=merged_env,
    )


def docker_exec(command: str, *, check: bool = True, timeout: float | None = None,
                input_text: str | None = None, capture: bool = True) -> subprocess.CompletedProcess:
    return run(["docker", "exec", CONTAINER, "bash", "-lc", command], check=check, timeout=timeout, input_text=input_text, capture=capture)


def ros_exec(command: str, *, check: bool = True, timeout: float | None = None,
             input_text: str | None = None, capture: bool = True) -> subprocess.CompletedProcess:
    return docker_exec(f"{ROS_ENV} && {command}", check=check, timeout=timeout, input_text=input_text, capture=capture)


def docker_ros_popen(command: str, *, input_text: str | None = None) -> subprocess.Popen:
    proc = subprocess.Popen(
        ["docker", "exec", "-i", CONTAINER, "bash", "-lc", f"{ROS_ENV} && {command}"],
        stdin=subprocess.PIPE if input_text is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if input_text is not None and proc.stdin is not None:
        proc.stdin.write(input_text)
        proc.stdin.close()
        # Python 3.14's communicate() flushes stdin even after caller-side close.
        # Mark it consumed so eval scripts can still call communicate(timeout=...).
        proc.stdin = None
    return proc


def require_container() -> None:
    proc = run(["docker", "ps", "--format", "{{.Names}}"], check=False)
    names = set((proc.stdout or "").splitlines())
    if CONTAINER not in names:
        raise SystemExit(f"Container {CONTAINER!r} is not running. Start with: cd {BACKEND_ROOT} && ./scripts/backend11_lifecycle.sh bringup_dual")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_percent(value: str) -> float | None:
    try:
        return float(str(value).strip().rstrip("%"))
    except Exception:
        return None


def parse_mem_usage(value: str) -> tuple[float | None, str]:
    # Docker usually returns "123.4MiB / 7.7GiB".
    raw = str(value)
    first = raw.split("/")[0].strip()
    match = re.match(r"([0-9.]+)\s*([A-Za-z]+)", first)
    if not match:
        return None, raw
    number = float(match.group(1))
    unit = match.group(2).lower()
    scale = {
        "b": 1,
        "kib": 1024,
        "kb": 1000,
        "mib": 1024 ** 2,
        "mb": 1000 ** 2,
        "gib": 1024 ** 3,
        "gb": 1000 ** 3,
    }.get(unit, 1)
    return number * scale, raw


def sample_docker_stats(duration: float, interval: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    deadline = time.monotonic() + duration
    while time.monotonic() < deadline:
        wall = time.time()
        proc = run(
            ["docker", "stats", "--no-stream", "--format", "{{json .}}", CONTAINER],
            check=False,
            capture=True,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            try:
                data = json.loads(proc.stdout.strip().splitlines()[-1])
            except Exception:
                data = {"raw": proc.stdout.strip()}
            cpu = parse_percent(data.get("CPUPerc", ""))
            mem_bytes, mem_raw = parse_mem_usage(data.get("MemUsage", ""))
            rows.append({
                "wall_time": wall,
                "cpu_percent": cpu,
                "mem_bytes": mem_bytes,
                "mem_usage_raw": mem_raw,
                "net_io": data.get("NetIO", ""),
                "block_io": data.get("BlockIO", ""),
                "pids": data.get("PIDs", ""),
            })
        time.sleep(max(0.05, interval))
    return rows


def summarize(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None, "std": None, "p05": None}
    ordered = sorted(values)
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    p05 = ordered[max(0, min(len(ordered) - 1, int(0.05 * (len(ordered) - 1))))]
    return {"count": len(values), "mean": mean, "min": ordered[0], "max": ordered[-1], "std": var ** 0.5, "p05": p05}


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--duration", type=float, default=30.0, help="Sampling duration in seconds.")
    parser.add_argument("--interval", type=float, default=1.0, help="Sampling interval in seconds.")
    parser.add_argument("--output-root", default=None, help="Override eval output root. Defaults to ros_backend1.1/eval_results.")


def print_done(out_dir: Path) -> None:
    print(f"[ok] wrote results to {out_dir}")
