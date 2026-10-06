#!/usr/bin/env python3
"""Summarize stage timings; simulation and monotonic wall clocks stay separate."""
import argparse
import json
from pathlib import Path
import statistics


def describe(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return {'samples': 0}
    return dict(samples=len(values), median=statistics.median(values),
                p95=values[min(len(values)-1, int(.95*(len(values)-1)))], max=max(values))


def analyze(path):
    def rows(name):
        file = path/name
        if not file.exists():
            return []
        with file.open() as stream:
            return [json.loads(line) for line in stream]
    frames = rows('camera_latency.jsonl')
    actions = rows('policy_actions.jsonl')
    result = {'episode': str(path), 'live_instrumentation_present': bool(frames),
        'interpretation': 'Source-to-callback simulation age combines rendering, transport and executor delay. These cannot be separated by ROS endpoint timestamps alone. Negative ages can occur because the reference clock callback lags. Wall stages use monotonic time; never subtract wall time from simulation time.'}
    for key in ('age_at_callback_joint_sim_s', 'age_at_callback_clock_sim_s',
                'clock_receipt_age_wall_s', 'callback_copy_publish_wall_s',
                'writer_queue_wall_s', 'png_encode_write_wall_s', 'callback_to_png_done_wall_s'):
        result[key] = describe([r.get(key) for r in frames])
    for key in ('image_age_at_selection_sim_s', 'callback_to_selection_wall_s',
                'preprocess_wall_s', 'rpc_wall_s', 'inference_wall_s'):
        result[key] = describe([r.get(key) for r in actions])
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('episode', type=Path)
    print(json.dumps(analyze(p.parse_args().episode), indent=2))
