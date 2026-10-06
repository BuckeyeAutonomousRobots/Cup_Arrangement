#!/usr/bin/env python3
"""Merge per-platform dynamic performance summaries into one CSV."""

from __future__ import annotations

import argparse
import csv
import platform
from pathlib import Path


FIELDNAMES = [
    "platform_label",
    "condition_id",
    "condition_label",
    "gazebo_mode",
    "novnc_state",
    "duration_sec",
    "rtf_mean",
    "rtf_min",
    "rtf_max",
    "rtf_std",
    "cpu_mean_percent",
    "cpu_min_percent",
    "cpu_max_percent",
    "cpu_std_percent",
    "raw_dir",
    "source_csv",
]


def infer_platform_label(path: Path, fallback: str) -> str:
    text = path.name.lower()
    if "ubuntu" in text or "linux" in text:
        return "Ubuntu/Linux"
    if "windows" in text or "wsl" in text:
        return "Windows/WSL2"
    if "mac" in text or "darwin" in text or "macos" in text:
        return "macOS"
    return fallback


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, help="dynamic_performance_summary.csv from a platform run.")
    parser.add_argument("--output", required=True, help="Merged output CSV path.")
    parser.add_argument("--platform-label", action="append", default=[], help="Optional label for each input, in the same order.")
    args = parser.parse_args()

    rows = []
    fallback = f"{platform.system()}-{platform.release()}-{platform.machine()}"
    for index, raw_path in enumerate(args.input):
        path = Path(raw_path)
        label = args.platform_label[index] if index < len(args.platform_label) else infer_platform_label(path, fallback)
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append({
                    "platform_label": label,
                    "condition_id": row.get("id", ""),
                    "condition_label": row.get("label", ""),
                    "gazebo_mode": row.get("gazebo_mode", ""),
                    "novnc_state": row.get("novnc_state", ""),
                    "duration_sec": row.get("duration_sec", ""),
                    "rtf_mean": row.get("rtf_mean", ""),
                    "rtf_min": row.get("rtf_min", ""),
                    "rtf_max": row.get("rtf_max", ""),
                    "rtf_std": row.get("rtf_std", ""),
                    "cpu_mean_percent": row.get("cpu_mean_percent", ""),
                    "cpu_min_percent": row.get("cpu_min_percent", ""),
                    "cpu_max_percent": row.get("cpu_max_percent", ""),
                    "cpu_std_percent": row.get("cpu_std_percent", ""),
                    "raw_dir": row.get("raw_dir", ""),
                    "source_csv": str(path),
                })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
