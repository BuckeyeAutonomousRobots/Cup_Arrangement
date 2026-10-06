#!/usr/bin/env python3
"""Bounded camera/clock validation. Rates explicitly distinguish wall/sim time."""
import argparse
import json
import pathlib
import time

import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, Image


def seconds(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seconds', type=float, default=60)
    parser.add_argument('--output', required=True)
    parser.add_argument('--clock-only', action='store_true', help='Baseline with no camera enabled')
    args = parser.parse_args()
    if not 1 <= args.seconds <= 600:
        parser.error('duration must be between 1 and 600 seconds')
    dest = pathlib.Path(args.output)
    dest.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = rclpy.create_node('gripper_camera_validator')
    stamps, arrivals, clocks, infos = [], [], [], []
    last_image = [None]

    def image_cb(msg):
        stamps.append(seconds(msg.header.stamp))
        arrivals.append(time.monotonic())
        last_image[0] = msg

    subscriptions = [
        node.create_subscription(Image, '/right_gripper_camera/image_raw', image_cb, qos_profile_sensor_data),
        node.create_subscription(CameraInfo, '/right_gripper_camera/camera_info', lambda m: infos.append(m), qos_profile_sensor_data),
        node.create_subscription(Clock, '/clock', lambda m: clocks.append((time.monotonic(), seconds(m.clock))), qos_profile_sensor_data),
    ]
    start = time.monotonic()
    while time.monotonic() - start < args.seconds:
        rclpy.spin_once(node, timeout_sec=0.1)
    result = {'duration_wall_s': time.monotonic() - start, 'frames': len(stamps), 'camera_info_count': len(infos)}
    if len(clocks) > 1:
        result['rtf_clock_delta'] = (clocks[-1][1] - clocks[0][1]) / (clocks[-1][0] - clocks[0][0])
    if len(stamps) > 1:
        ds = np.diff(stamps)
        dw = np.diff(arrivals)
        result.update(fps_wall=(len(stamps)-1)/(arrivals[-1]-arrivals[0]),
                      fps_sim=(len(stamps)-1)/(stamps[-1]-stamps[0]),
                      nonmonotonic_stamps=int(np.sum(ds <= 0)),
                      sim_gap_max_s=float(ds.max()), wall_gap_max_s=float(dw.max()),
                      sim_gaps_over_50ms=int(np.sum(ds > 0.050)),
                      wall_gap_p95_s=float(np.percentile(dw, 95)))
    msg = last_image[0]
    if msg is not None:
        result.update(width=msg.width, height=msg.height, encoding=msg.encoding, frame_id=msg.header.frame_id)
        pixels = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(msg.height, msg.step)[:, :msg.width*3].reshape(msg.height, msg.width, 3)
        result.update(pixel_std=float(pixels.std()), pixel_min=int(pixels.min()), pixel_max=int(pixels.max()))
        if msg.encoding == 'rgb8':
            (dest / 'camera.ppm').write_bytes(f'P6\n{msg.width} {msg.height}\n255\n'.encode() + pixels.tobytes())
    if infos:
        info = infos[-1]
        result['camera_info'] = dict(width=info.width, height=info.height, frame_id=info.header.frame_id, k=list(info.k), p=list(info.p))
    result['passed_stream_check'] = bool(len(stamps) > 2 and infos and result.get('nonmonotonic_stamps') == 0 and result.get('pixel_std', 0) > 1)
    result['passed_requested_check'] = bool(len(clocks) > 1) if args.clock_only else result['passed_stream_check']
    (dest / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    node.destroy_node()
    rclpy.shutdown()
    return 0 if result['passed_requested_check'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
