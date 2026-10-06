#!/usr/bin/env python3
"""Compare dynamic RTF/CPU under plain, noVNC, and noVNC+headed modes.

This runner intentionally uses the existing backend lifecycle and monitor
script so the test matches normal bringup behavior. During each sample window
it starts the synthetic hand generator with a large vertical motion to create
dynamic Servo/Gazebo pressure without requiring the Quest headset.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from eval_common import BACKEND_ROOT, REPO_ROOT, results_dir, run, safe_name, summarize, write_csv, write_json


CONDITIONS = [
    {
        "id": "plain_backend",
        "label": "Plain backend",
        "enable_desktop": "0",
        "enable_novnc": "0",
        "sim_headless": "1",
        "gazebo_mode": "headless",
        "novnc_state": "disabled",
    },
    {
        "id": "novnc_only",
        "label": "noVNC only",
        "enable_desktop": "1",
        "enable_novnc": "1",
        "sim_headless": "1",
        "gazebo_mode": "headless",
        "novnc_state": "enabled; no browser connection required",
    },
    {
        "id": "novnc_headed",
        "label": "noVNC + headed Gazebo",
        "enable_desktop": "1",
        "enable_novnc": "1",
        "sim_headless": "0",
        "gazebo_mode": "headed",
        "novnc_state": "enabled",
    },
]


def lifecycle(command: str, env: dict[str, str] | None = None, timeout: float | None = None) -> subprocess.CompletedProcess:
    return run(
        ["./scripts/backend11_lifecycle.sh", command],
        cwd=BACKEND_ROOT,
        env=env,
        check=False,
        timeout=timeout,
    )


def append_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(text)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_float(value: Any) -> float | None:
    try:
        if value in ("", None):
            return None
        return float(value)
    except Exception:
        return None


def latest_monitor_dir(condition_dir: Path) -> Path | None:
    dirs = sorted(condition_dir.glob("*_backend_rtf_cpu"), key=lambda p: p.stat().st_mtime)
    return dirs[-1] if dirs else None


def run_monitor(duration: float, interval: float, output_root: Path) -> tuple[subprocess.CompletedProcess, Path | None]:
    monitor = Path(__file__).with_name("01_backend_rtf_cpu_monitor.py")
    proc = subprocess.run(
        [
            sys.executable,
            str(monitor),
            "--duration",
            str(duration),
            "--interval",
            str(interval),
            "--rtf-window-sec",
            "1.0",
            "--output-root",
            str(output_root),
        ],
        cwd=str(BACKEND_ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    return proc, latest_monitor_dir(output_root)


def load_summary(monitor_dir: Path | None) -> dict[str, Any]:
    if monitor_dir is None:
        return {}
    path = monitor_dir / "summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def start_dynamic_hand(duration: float, args: argparse.Namespace, condition_dir: Path) -> subprocess.CompletedProcess:
    env = {
        "DEBUG_ARM": args.fake_arm,
        "DEBUG_PATTERN": args.fake_pattern,
        "DEBUG_DURATION_SEC": str(duration),
        "DEBUG_PERIOD_SEC": str(args.fake_period),
        "DEBUG_AMPLITUDE_X": str(args.fake_amplitude_x),
        "DEBUG_AMPLITUDE_Y": str(args.fake_amplitude_y),
        "DEBUG_AMPLITUDE_Z": str(args.fake_amplitude_z),
    }
    proc = lifecycle("debug_hand_start", env=env, timeout=30.0)
    (condition_dir / "debug_hand_start.stdout.txt").write_text(proc.stdout or "", encoding="utf-8")
    (condition_dir / "debug_hand_start.stderr.txt").write_text(proc.stderr or "", encoding="utf-8")
    return proc


def stop_dynamic_hand(condition_dir: Path) -> None:
    proc = lifecycle("debug_hand_stop", timeout=20.0)
    (condition_dir / "debug_hand_stop.stdout.txt").write_text(proc.stdout or "", encoding="utf-8")
    (condition_dir / "debug_hand_stop.stderr.txt").write_text(proc.stderr or "", encoding="utf-8")


def run_condition(condition: dict[str, str], args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    condition_dir = run_dir / condition["id"]
    condition_dir.mkdir(parents=True, exist_ok=True)
    env = {
        "ENABLE_DESKTOP": condition["enable_desktop"],
        "ENABLE_NOVNC": condition["enable_novnc"],
        "SIM_HEADLESS": condition["sim_headless"],
    }

    append_text(run_dir / "run.log", f"\n[{dt.datetime.now().isoformat()}] condition {condition['id']} starting\n")

    down = lifecycle("safe_down", timeout=60.0)
    (condition_dir / "pre_safe_down.stdout.txt").write_text(down.stdout or "", encoding="utf-8")
    (condition_dir / "pre_safe_down.stderr.txt").write_text(down.stderr or "", encoding="utf-8")

    bringup = lifecycle("bringup_dual", env=env, timeout=args.bringup_timeout)
    (condition_dir / "bringup.stdout.txt").write_text(bringup.stdout or "", encoding="utf-8")
    (condition_dir / "bringup.stderr.txt").write_text(bringup.stderr or "", encoding="utf-8")
    if bringup.returncode != 0:
        return {
            **condition,
            "ok": False,
            "returncode": bringup.returncode,
            "error": "bringup_dual failed",
            "raw_dir": str(condition_dir.relative_to(REPO_ROOT)),
        }

    if args.warmup > 0:
        time.sleep(args.warmup)

    fake_duration = args.duration + max(2.0, args.warmup * 0.25)
    fake = start_dynamic_hand(fake_duration, args, condition_dir)
    if fake.returncode != 0:
        return {
            **condition,
            "ok": False,
            "returncode": fake.returncode,
            "error": "debug_hand_start failed",
            "raw_dir": str(condition_dir.relative_to(REPO_ROOT)),
        }

    proc, monitor_dir = run_monitor(args.duration, args.interval, condition_dir)
    (condition_dir / "monitor.stdout.txt").write_text(proc.stdout or "", encoding="utf-8")
    (condition_dir / "monitor.stderr.txt").write_text(proc.stderr or "", encoding="utf-8")
    stop_dynamic_hand(condition_dir)

    summary = load_summary(monitor_dir)
    rtf_summary = summary.get("rtf_window") or summary.get("rtf_instant") or {}
    cpu_summary = summary.get("cpu_percent") or {}
    result = {
        **condition,
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "duration_sec": args.duration,
        "fake_motion": (
            f"{args.fake_arm}:{args.fake_pattern}, "
            f"amp=({args.fake_amplitude_x},{args.fake_amplitude_y},{args.fake_amplitude_z}), "
            f"period={args.fake_period}s"
        ),
        "rtf_mean": rtf_summary.get("mean"),
        "rtf_min": rtf_summary.get("min"),
        "rtf_max": rtf_summary.get("max"),
        "rtf_std": rtf_summary.get("std"),
        "rtf_count": rtf_summary.get("count"),
        "cpu_mean_percent": cpu_summary.get("mean"),
        "cpu_min_percent": cpu_summary.get("min"),
        "cpu_max_percent": cpu_summary.get("max"),
        "cpu_std_percent": cpu_summary.get("std"),
        "cpu_count": cpu_summary.get("count"),
        "raw_dir": str((monitor_dir or condition_dir).relative_to(REPO_ROOT)),
    }
    append_text(run_dir / "run.log", f"[{dt.datetime.now().isoformat()}] condition {condition['id']} complete: {result}\n")
    return result


def collect_timeseries(run_dir: Path, results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cpu_rows: list[dict[str, Any]] = []
    rtf_rows: list[dict[str, Any]] = []
    by_id = {condition["id"]: condition for condition in CONDITIONS}
    for result in results:
        condition_id = result["id"]
        condition = by_id.get(condition_id, result)
        raw_dir = REPO_ROOT / str(result.get("raw_dir", ""))
        if not raw_dir.exists():
            continue

        raw_cpu = read_csv(raw_dir / "docker_cpu_memory.csv")
        cpu_times = [parse_float(row.get("wall_time")) for row in raw_cpu]
        cpu_times = [value for value in cpu_times if value is not None]
        cpu_t0 = min(cpu_times) if cpu_times else None
        for row in raw_cpu:
            wall = parse_float(row.get("wall_time"))
            cpu = parse_float(row.get("cpu_percent"))
            if wall is None or cpu is None or cpu_t0 is None:
                continue
            cpu_rows.append({
                "condition_id": condition_id,
                "condition_label": condition.get("label", condition_id),
                "time_sec": wall - cpu_t0,
                "cpu_percent": cpu,
            })

        raw_rtf = read_csv(raw_dir / "clock_rtf_windowed.csv")
        starts = [parse_float(row.get("wall_start_mono")) for row in raw_rtf]
        starts = [value for value in starts if value is not None]
        rtf_t0 = min(starts) if starts else None
        for row in raw_rtf:
            start = parse_float(row.get("wall_start_mono"))
            end = parse_float(row.get("wall_end_mono"))
            rtf = parse_float(row.get("rtf_window"))
            if start is None or end is None or rtf is None or rtf_t0 is None:
                continue
            rtf_rows.append({
                "condition_id": condition_id,
                "condition_label": condition.get("label", condition_id),
                "time_sec": end - rtf_t0,
                "rtf": rtf,
            })
    write_csv(run_dir / "combined_cpu_timeseries.csv", cpu_rows)
    write_csv(run_dir / "combined_rtf_timeseries.csv", rtf_rows)
    return cpu_rows, rtf_rows


def svg_escape(value: Any) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def make_svg_plot(
    rows: list[dict[str, Any]],
    value_key: str,
    title: str,
    y_label: str,
    output: Path,
    y_min: float | None = None,
    y_max: float | None = None,
) -> None:
    width, height = 980, 560
    left, right, top, bottom = 82, 32, 58, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    colors = {
        "plain_backend": "#2563eb",
        "novnc_only": "#f97316",
        "novnc_headed": "#16a34a",
    }
    grouped: dict[str, list[dict[str, Any]]] = {}
    labels: dict[str, str] = {}
    for row in rows:
        cid = str(row.get("condition_id", "unknown"))
        grouped.setdefault(cid, []).append(row)
        labels[cid] = str(row.get("condition_label", cid))

    x_values = [float(row["time_sec"]) for row in rows if row.get("time_sec") is not None]
    y_values = [float(row[value_key]) for row in rows if row.get(value_key) is not None and math.isfinite(float(row[value_key]))]
    if not x_values or not y_values:
        output.write_text(f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'><text x='20' y='40'>No data for {svg_escape(title)}</text></svg>\n", encoding="utf-8")
        return

    x0, x1 = 0.0, max(max(x_values), 1.0)
    if y_min is None:
        y_min = min(0.0, min(y_values))
    if y_max is None:
        y_max = max(y_values)
    if abs(y_max - y_min) < 1e-9:
        y_max = y_min + 1.0
    pad = (y_max - y_min) * 0.08
    y_min -= pad
    y_max += pad

    def sx(x: float) -> float:
        return left + (x - x0) / (x1 - x0) * plot_w

    def sy(y: float) -> float:
        return top + (y_max - y) / (y_max - y_min) * plot_h

    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='white'/>",
        f"<text x='{left}' y='32' font-family='Arial' font-size='22' font-weight='700'>{svg_escape(title)}</text>",
        f"<rect x='{left}' y='{top}' width='{plot_w}' height='{plot_h}' fill='#f8fafc' stroke='#cbd5e1'/>",
    ]
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = top + frac * plot_h
        val = y_max - frac * (y_max - y_min)
        parts.append(f"<line x1='{left}' y1='{y:.1f}' x2='{left+plot_w}' y2='{y:.1f}' stroke='#e2e8f0'/>")
        parts.append(f"<text x='{left-10}' y='{y+4:.1f}' text-anchor='end' font-family='Arial' font-size='12' fill='#475569'>{val:.2f}</text>")
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        x = left + frac * plot_w
        val = x0 + frac * (x1 - x0)
        parts.append(f"<line x1='{x:.1f}' y1='{top}' x2='{x:.1f}' y2='{top+plot_h}' stroke='#e2e8f0'/>")
        parts.append(f"<text x='{x:.1f}' y='{top+plot_h+24}' text-anchor='middle' font-family='Arial' font-size='12' fill='#475569'>{val:.0f}</text>")

    for cid, series in grouped.items():
        series = sorted(series, key=lambda row: float(row["time_sec"]))
        points = []
        for row in series:
            x = float(row["time_sec"])
            y = float(row[value_key])
            if math.isfinite(y):
                points.append(f"{sx(x):.2f},{sy(y):.2f}")
        if len(points) >= 2:
            color = colors.get(cid, "#334155")
            parts.append(f"<polyline points='{' '.join(points)}' fill='none' stroke='{color}' stroke-width='3' stroke-linejoin='round' stroke-linecap='round'/>")

    legend_x = left + 12
    legend_y = top + 20
    for idx, cid in enumerate(grouped.keys()):
        y = legend_y + idx * 24
        color = colors.get(cid, "#334155")
        parts.append(f"<line x1='{legend_x}' y1='{y}' x2='{legend_x+28}' y2='{y}' stroke='{color}' stroke-width='4'/>")
        parts.append(f"<text x='{legend_x+36}' y='{y+5}' font-family='Arial' font-size='13' fill='#0f172a'>{svg_escape(labels.get(cid, cid))}</text>")

    parts.append(f"<text x='{left + plot_w / 2}' y='{height - 20}' text-anchor='middle' font-family='Arial' font-size='14' fill='#334155'>time (s)</text>")
    parts.append(f"<text transform='translate(22,{top + plot_h / 2}) rotate(-90)' text-anchor='middle' font-family='Arial' font-size='14' fill='#334155'>{svg_escape(y_label)}</text>")
    parts.append("</svg>")
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def append_performance_trials(results: list[dict[str, Any]], run_dir: Path, trial_csv: Path) -> None:
    fieldnames = [
        "trial_id",
        "host_os",
        "host_hardware",
        "gpu_mode",
        "gazebo_mode",
        "novnc_state",
        "duration_s",
        "rtf_mean",
        "rtf_min",
        "rtf_max",
        "rtf_std",
        "cpu_mean_percent",
        "cpu_min_percent",
        "cpu_max_percent",
        "memory_mean_mb",
        "notes",
        "raw_log_refs",
    ]
    trial_csv.parent.mkdir(parents=True, exist_ok=True)
    exists = trial_csv.exists()
    with trial_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for result in results:
            if not result.get("ok"):
                continue
            writer.writerow({
                "trial_id": f"perf_{dt.datetime.now().strftime('%Y%m%d')}_{result['id']}_dynamic",
                "host_os": f"{platform.system()}-{platform.release()}-{platform.machine()}",
                "host_hardware": "Mac / Docker Desktop ARM64",
                "gpu_mode": "Mac Docker CPU simulation; Gazebo GPU acceleration not used",
                "gazebo_mode": result.get("gazebo_mode"),
                "novnc_state": result.get("novnc_state"),
                "duration_s": result.get("duration_sec"),
                "rtf_mean": result.get("rtf_mean"),
                "rtf_min": result.get("rtf_min"),
                "rtf_max": result.get("rtf_max"),
                "rtf_std": result.get("rtf_std"),
                "cpu_mean_percent": result.get("cpu_mean_percent"),
                "cpu_min_percent": result.get("cpu_min_percent"),
                "cpu_max_percent": result.get("cpu_max_percent"),
                "memory_mean_mb": "",
                "notes": f"Dynamic fake-hand pressure test: {result.get('fake_motion')}. RTF uses 1-second windowed /clock.",
                "raw_log_refs": result.get("raw_dir"),
            })


def write_markdown_section(results: list[dict[str, Any]], output: Path, run_dir: Path) -> None:
    lines = [
        "# Dynamic Backend Performance: noVNC and Headed Gazebo",
        "",
        "## Why This Test Was Run",
        "",
        "This test isolates backend simulation/rendering overhead from headset usability. The teleoperation system depends on Gazebo running close to real time while MoveIt Servo, object synchronization, haptics, and the TCP/ROS bridge are active. If desktop services or Gazebo GUI rendering consume too much CPU, the robot can feel delayed even when the Unity controller stream is healthy.",
        "",
        "The synthetic hand generator replaces the headset for this test. It publishes repeatable Quest-like hand poses while teleop is engaged, using a large vertical oscillation intended to push the robot into the workspace/table region. That makes the measurement closer to a loaded manipulation case than an idle simulation measurement.",
        "",
        "## Test Conditions",
        "",
        "| Condition | Gazebo | Desktop/noVNC | Dynamic input |",
        "|---|---:|---:|---|",
    ]
    for result in results:
        lines.append(
            f"| {result.get('label', result.get('id'))} | {result.get('gazebo_mode')} | {result.get('novnc_state')} | {result.get('fake_motion')} |"
        )

    lines += [
        "",
        "## Summary Table",
        "",
        "| Condition | Mean RTF | Min RTF | Max RTF | Mean CPU % | Min CPU % | Max CPU % |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        def fmt(key: str) -> str:
            value = result.get(key)
            return "" if value is None else f"{float(value):.3f}"
        lines.append(
            f"| {result.get('label', result.get('id'))} | {fmt('rtf_mean')} | {fmt('rtf_min')} | {fmt('rtf_max')} | {fmt('cpu_mean_percent')} | {fmt('cpu_min_percent')} | {fmt('cpu_max_percent')} |"
        )

    valid = [r for r in results if r.get("rtf_mean") is not None]
    best = max(valid, key=lambda r: float(r["rtf_mean"])) if valid else None
    worst = min(valid, key=lambda r: float(r["rtf_mean"])) if valid else None
    lines += [
        "",
        "## Interpretation",
        "",
    ]
    if best and worst:
        lines.append(
            f"In this run, `{best['label']}` had the highest mean RTF and `{worst['label']}` had the lowest mean RTF. The useful interpretation is not only the mean value, but also how stable the RTF trace is under dynamic robot motion. A lower or more variable RTF means Gazebo is falling behind wall time, which directly increases perceived robot sluggishness."
        )
    else:
        lines.append(
            "This run did not produce enough valid RTF samples for a condition-level conclusion. Re-run with a longer duration or inspect the Gazebo/clock bridge logs."
        )
    lines += [
        "",
        "CPU percentage is Docker container CPU from `docker stats`, so values can exceed 100% on multi-core systems. This is still useful for comparing relative backend load across conditions on the same host.",
        "",
        "## Plots",
        "",
        "- `dynamic_cpu_load_timeseries.svg`: time versus Docker CPU load.",
        "- `dynamic_rtf_timeseries.svg`: time versus 1-second-windowed real-time factor.",
        "",
        f"Raw run directory: `{run_dir.relative_to(REPO_ROOT)}`",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--warmup", type=float, default=8.0)
    parser.add_argument("--bringup-timeout", type=float, default=240.0)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--fake-arm", default="both", choices=["left", "right", "both"])
    parser.add_argument("--fake-pattern", default="line_y", choices=["line_x", "line_y", "line_z", "circle_xy", "circle_xz"])
    parser.add_argument("--fake-period", type=float, default=8.0)
    parser.add_argument("--fake-amplitude-x", type=float, default=0.0)
    parser.add_argument("--fake-amplitude-y", type=float, default=0.55)
    parser.add_argument("--fake-amplitude-z", type=float, default=0.0)
    parser.add_argument("--keep-running", action="store_true", help="Do not safe_down at the end. Default is to safe_down.")
    args = parser.parse_args()

    date_folder = dt.datetime.now().strftime("%m_%d")
    default_output_root = REPO_ROOT / "thesis_eval_raw" / date_folder / "logs" / "backend_eval_runs"
    run_dir = results_dir("dynamic_novnc_headed_performance", args.output_root or str(default_output_root))
    runtime_dir = REPO_ROOT / "thesis_eval_raw" / date_folder / "runtime_performance"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    try:
        for condition in CONDITIONS:
            result = run_condition(condition, args, run_dir)
            results.append(result)
            write_json(run_dir / "partial_summary.json", {"conditions": results})
    finally:
        if not args.keep_running:
            final_down = lifecycle("safe_down", timeout=90.0)
            (run_dir / "final_safe_down.stdout.txt").write_text(final_down.stdout or "", encoding="utf-8")
            (run_dir / "final_safe_down.stderr.txt").write_text(final_down.stderr or "", encoding="utf-8")

    cpu_rows, rtf_rows = collect_timeseries(run_dir, results)
    write_csv(run_dir / "dynamic_performance_summary.csv", results)
    write_json(run_dir / "dynamic_performance_summary.json", {"conditions": results})
    make_svg_plot(cpu_rows, "cpu_percent", "Dynamic Backend CPU Load", "Docker CPU (%)", run_dir / "dynamic_cpu_load_timeseries.svg")
    make_svg_plot(rtf_rows, "rtf", "Dynamic Gazebo Real-Time Factor", "RTF", run_dir / "dynamic_rtf_timeseries.svg", y_min=0.0)

    shutil.copy2(run_dir / "dynamic_cpu_load_timeseries.svg", runtime_dir / "dynamic_cpu_load_timeseries.svg")
    shutil.copy2(run_dir / "dynamic_rtf_timeseries.svg", runtime_dir / "dynamic_rtf_timeseries.svg")
    shutil.copy2(run_dir / "dynamic_performance_summary.csv", runtime_dir / "dynamic_performance_summary.csv")
    append_performance_trials(results, run_dir, runtime_dir / "performance_trials.csv")
    write_markdown_section(results, runtime_dir / "dynamic_backend_performance_section.md", run_dir)

    print(json.dumps({"run_dir": str(run_dir), "conditions": results}, indent=2))
    print(f"[ok] wrote runtime performance plots/table to {runtime_dir}")
    if not args.keep_running:
        print("[ok] safe_down completed after performance run")
    return 0 if all(r.get("ok") for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
