#!/usr/bin/env python3
"""One bounded simulated ACT rollout, wrist RGB recording and raw ROS data.

Run inside cup_arrangement_backend. The imported cheat helpers provide ONLY
ownership, state parsing, FK safety and final zero commands; no IK, waypoints,
object repositioning or expert action is used. Object poses are evaluation only.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import signal
import socket
import sys
import time

import cv2
import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, HistoryPolicy
from rosidl_runtime_py.convert import message_to_ordereddict
from sensor_msgs.msg import Image, JointState, CameraInfo
from rosgraph_msgs.msg import Clock
from std_msgs.msg import Float64MultiArray
from tf2_msgs.msg import TFMessage
from camera_pipeline import BoundedWorker, LatestFrame
from observation_recovery import ObservationRecovery

ROOT = Path('/workspace')
sys.path.insert(0, str(ROOT/'scripts'))
import cup_task as helpers


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--socket', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--policy-label', default='ACT best step 4500')
    p.add_argument('--sim-seconds', type=float, default=90)
    p.add_argument('--wall-seconds', type=float, default=500)
    a = p.parse_args()
    if not 1 <= a.sim_seconds <= 90 or not 10 <= a.wall_seconds <= 600:
        p.error('Rollout exceeds bounded duration')
    if (ROOT/'profiles/active_scene_profile.txt').read_text().strip() != 'profiles/scenes/baseline_v1/scene.yaml':
        raise RuntimeError('Expected baseline_v1 scene')
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output/'frames').mkdir()
    cv2.setNumThreads(1)
    signal.signal(signal.SIGTERM, helpers.interrupted)
    signal.signal(signal.SIGINT, helpers.interrupted)
    result = {'status': 'startup_failed', 'success': False, 'policy': a.policy_label,
              'seed': 8, 'inputs': 'wrist RGB + 8 joints only', 'actions': '6 joint velocities + common finger target',
              'max_sim_s': a.sim_seconds, 'max_wall_s': a.wall_seconds,
              'controller_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'camera_pipeline_sha256': hashlib.sha256(Path(__file__).with_name('camera_pipeline.py').read_bytes()).hexdigest(),
              'helpers_sha256': hashlib.sha256((ROOT/'scripts/cup_task.py').read_bytes()).hexdigest()}
    start_wall = time.monotonic()
    frames_log = (a.output/'frames.jsonl').open('w')
    raw_log = (a.output/'robot_data.jsonl').open('w')
    actions_log = (a.output/'policy_actions.jsonl').open('w')
    latest_slot = LatestFrame()
    recovery = ObservationRecovery()
    joint_stamps = {}; frame_times = []; c = None
    clock_sample = {}
    accepting_records = [True]
    latency_log = (a.output/'camera_latency.jsonl').open('w')
    def write_frame(item, enqueued, started):
        metadata, packet = item
        rgb = np.frombuffer(packet['data'], np.uint8).reshape(packet['height'], packet['step'])[:, :packet['width']*3].reshape(packet['height'], packet['width'], 3)
        if not cv2.imwrite(str(a.output/metadata['file']), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
            raise RuntimeError('Failed to record frame')
        done = time.monotonic()
        frames_log.write(json.dumps(metadata)+'\n')
        latency_log.write(json.dumps(dict(metadata, writer_queue_wall_s=started-enqueued,
            png_encode_write_wall_s=done-started, callback_to_png_done_wall_s=done-packet['wall']))+'\n')
    def write_data(item, enqueued, started):
        kind, row, msg = item
        if kind == 'raw':
            row['message'] = message_to_ordereddict(msg)
            raw_log.write(json.dumps(row)+'\n')
        else:
            actions_log.write(json.dumps(row)+'\n')
    image_writer = BoundedWorker(write_frame, 32, 'policy-png-writer')
    data_writer = BoundedWorker(write_data, 2048, 'policy-data-writer')
    pool = ThreadPoolExecutor(max_workers=1)
    sock = socket.socket(socket.AF_UNIX)
    sock.settimeout(1.)
    try:
        sock.connect(a.socket)
        wire = sock.makefile('rwb')

        def inference(obs, packet):
            prepare_start = time.monotonic()
            rgb = np.frombuffer(packet['data'], np.uint8).reshape(packet['height'], packet['step'])[:, :packet['width']*3].reshape(packet['height'], packet['width'], 3)
            rgb = cv2.resize(rgb, (320, 240), interpolation=cv2.INTER_AREA)
            obs['rgb'] = base64.b64encode(rgb.tobytes()).decode()
            send_start = time.monotonic()
            wire.write((json.dumps(obs)+'\n').encode()); wire.flush()
            response = wire.readline()
            if not response:
                raise RuntimeError('Inference server disconnected')
            decoded = json.loads(response)
            decoded.update(preprocess_wall_s=send_start-prepare_start,
                           rpc_wall_s=time.monotonic()-send_start)
            return decoded

        with helpers.task_lock(), helpers.exclusive_commands():
            c = helpers.GroundTruthController(helpers.active_task())
            try:
                names = c.names+['right_robotiq_hande_'+side+'_finger_joint' for side in ('left', 'right')]

                def frame(msg):
                    if not accepting_records[0]:
                        return
                    callback_start = time.monotonic()
                    if msg.encoding != 'rgb8':
                        raise RuntimeError('Expected RGB8 camera')
                    stamp = msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
                    index = len(frame_times); filename = f'frames/{index:06d}.png'
                    packet = dict(data=bytes(msg.data), height=msg.height, width=msg.width,
                                  step=msg.step, stamp=stamp, wall=callback_start, index=index)
                    latest_slot.publish(packet)
                    frame_times.append((stamp, callback_start))
                    metadata = dict(index=index, file=filename, stamp_sim_s=stamp, wall=callback_start,
                        age_at_callback_joint_sim_s=c.sim_time-stamp,
                        age_at_callback_clock_sim_s=None if not clock_sample else clock_sample['sim']-stamp,
                        clock_receipt_age_wall_s=None if not clock_sample else callback_start-clock_sample['wall'],
                        callback_copy_publish_wall_s=time.monotonic()-callback_start)
                    image_writer.submit((metadata, packet))

                def record(topic, msg):
                    if not accepting_records[0]:
                        return
                    wall = time.monotonic()
                    data_writer.submit(('raw', {'topic': topic, 'receipt_sim_s': c.sim_time,
                        'receipt_monotonic_s': wall}, msg))
                    if isinstance(msg, Clock):
                        clock_sample.update(sim=msg.clock.sec+msg.clock.nanosec*1e-9, wall=wall)
                    if isinstance(msg, JointState):
                        t = msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
                        for name in msg.name:
                            joint_stamps[name] = t

                image_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                                       reliability=ReliabilityPolicy.BEST_EFFORT)
                subs = [c.node.create_subscription(Image, '/right_gripper_camera/image_raw', frame, image_qos)]
                for cls, topic in [(JointState, '/joint_states'), (Clock, '/clock'),
                        (CameraInfo, '/right_gripper_camera/camera_info'),
                        (Float64MultiArray, '/right_joint_group_velocity_controller/commands'),
                        (Float64MultiArray, '/right_hande_position_controller/commands'),
                        (TFMessage, f'/world/{helpers.WORLD}/dynamic_pose/info')]:
                    subs.append(c.node.create_subscription(cls, topic, lambda m,t=topic: record(t,m), qos_profile_sensor_data))
                c.wait(lambda: len(frame_times) >= 5 and all(n in joint_stamps for n in names), 25)
                c.command([0.]*6)
                initial_cup = c.poses[helpers.CUP].copy()
                initial_q = [c.q[n] for n in names]
                if max(abs(x) for x in initial_q[6:]) > .002:
                    raise RuntimeError('Gripper must start open')
                start_sim = c.sim_time; next_tick = start_sim
                pending = None; submitted = None; action = np.zeros(7); action[6] = 0.
                reset_policy = False
                applied_wall = time.monotonic(); applied_sim = start_sim
                count = 0; clamped = 0; stable_since = None; max_lift = 0.; last_report = -10.
                limits = [(float(j.find('limit').get('lower'))+.02, float(j.find('limit').get('upper'))-.02)
                          for j in c.chain if j.get('type') == 'revolute']
                print(json.dumps({'event': 'policy_started', 'initial_q': initial_q, 'start_sim': start_sim}), flush=True)
                result.update(status='running', initial_cup=initial_cup, initial_q=initial_q)
                while c.sim_time-start_sim < a.sim_seconds:
                    rclpy.spin_once(c.node, timeout_sec=.005)
                    helpers.check_cancelled()
                    image_writer.check(); data_writer.check()
                    latest = latest_slot.peek()
                    wall = time.monotonic()
                    if wall-start_wall > a.wall_seconds:
                        raise TimeoutError('Wall-time limit')
                    if wall-c.last_joint > 1 or wall-latest['wall'] > 1:
                        raise RuntimeError('Stale camera or joint data')
                    q = np.array([c.q[n] for n in c.names]); tool = c.fk(q)
                    # Conservative workspace envelope, not an expert controller.
                    if not (.04 < tool[0,3] < .48 and -.28 < tool[1,3] < .28 and .278 < tool[2,3] < .65):
                        raise RuntimeError('Tool left safe workspace envelope')
                    tilt = math.acos(float(np.clip(-tool[2,2], -1, 1)))
                    if tilt > .35:
                        raise RuntimeError('Gripper tilt exceeded 20 degrees')
                    if q[1] > -.1 or q[2] < .1:
                        raise RuntimeError('Unsafe shoulder/elbow branch')
                    if any(not lo < v < hi for v,(lo,hi) in zip(q,limits)):
                        raise RuntimeError('Joint limit margin')
                    # Check freshness before applying even an already-completed prediction.
                    was_paused = recovery.active is not None
                    mode = recovery.update(wall, c.sim_time, latest['stamp'],
                                           [joint_stamps[n] for n in names], latest['index'],
                                           pending is not None)
                    if mode != 'normal':
                        action[:6] = 0.
                        c.command([0.]*6)
                        c.grip.publish(c.Msg(data=[float(action[6])]*2))
                        stable_since = None
                        if not was_paused:
                            print(json.dumps({'event': 'observation_pause', **recovery.events[-1]}), flush=True)
                        # Drain the single RPC before sending anything else on its socket.
                        # Its action is NEVER applied; reset ACT's temporal ensemble on resume.
                        if pending is not None and pending.done():
                            response = pending.result()
                            if response['id'] != count:
                                raise RuntimeError('Mismatched discarded policy response')
                            pending = None
                        if mode == 'wait':
                            # Retain object safety checks while stationary too.
                            if any(wall-c.pose_received.get(n, 0) >= 1 for n in (helpers.CUP, helpers.SAUCER)):
                                raise RuntimeError('Stale object state during observation pause')
                            if c.poses[helpers.CUP][2] < .07 or c.tilt[helpers.CUP] > .6:
                                raise RuntimeError('Cup fell/tipped during observation pause')
                            continue
                        print(json.dumps({'event': 'observation_resume', **recovery.events[-1]}), flush=True)
                        next_tick = c.sim_time  # Skip missed ticks; never replay/catch up.
                        applied_wall = wall; applied_sim = c.sim_time
                        reset_policy = True
                    if pending is not None and pending.done():
                        response = pending.result()
                        if response['id'] != count or wall-submitted > .75 or c.sim_time-obs_sim > .2:
                            raise RuntimeError('Late/mismatched policy response')
                        predicted = np.array(response['action'], dtype=float)
                        if predicted.shape != (7,) or not np.isfinite(predicted).all():
                            raise RuntimeError('Invalid action')
                        action = np.clip(predicted, [-.245]*6+[-.01], [.245]*6+[0.])
                        clamped += int(not np.array_equal(action, predicted))
                        data_writer.submit(('action', {'index': count, 'observation_sim_s': obs_sim,
                            'applied_sim_s': c.sim_time, 'state': obs_state, 'raw': predicted.tolist(),
                            'applied': action.tolist(), 'inference_wall_s': response['inference_wall_s'],
                            'preprocess_wall_s': response['preprocess_wall_s'], 'rpc_wall_s': response['rpc_wall_s'],
                            **obs_timing}, None))
                        count += 1; pending = None; applied_wall = wall; applied_sim = c.sim_time
                    if pending is not None and wall-submitted > .75:
                        raise RuntimeError('Inference watchdog timeout')
                    if pending is None and c.sim_time >= next_tick:
                        latest = latest_slot.take_latest()
                        if abs(c.sim_time-latest['stamp']) > .1 or any(abs(c.sim_time-joint_stamps[n]) > .1 for n in names):
                            result['timing_stop'] = {
                                'sim_s': c.sim_time, 'elapsed_sim_s': c.sim_time-start_sim,
                                'threshold_sim_s': .1,
                                'image_stamp_sim_s': latest['stamp'],
                                'image_age_sim_s': c.sim_time-latest['stamp'],
                                'image_age_wall_s': wall-latest['wall'],
                                'joint_age_sim_s': {n: c.sim_time-joint_stamps[n] for n in names},
                                'right_joint_age_wall_s': wall-c.last_joint,
                                'last_frame_intervals_sim_s': [b[0]-a[0] for a,b in zip(frame_times[-6:],frame_times[-5:])],
                            }
                            print(json.dumps({'timing_stop': result['timing_stop']}), flush=True)
                            raise RuntimeError('Unsynchronized observation')
                        obs_sim = c.sim_time; obs_state = [c.q[n] for n in names]
                        obs_timing = dict(image_index=latest['index'], image_stamp_sim_s=latest['stamp'],
                            image_age_at_selection_sim_s=obs_sim-latest['stamp'],
                            callback_to_selection_wall_s=time.monotonic()-latest['wall'])
                        pending = pool.submit(inference, {'id': count, 'state': obs_state,
                                                        'reset_policy': reset_policy}, latest)
                        reset_policy = False
                        submitted = wall; next_tick += .1
                        if next_tick < c.sim_time:
                            raise RuntimeError('Missed 10 Hz simulation control tick')
                    if count and (wall-applied_wall > 1 or c.sim_time-applied_sim > .22):
                        raise RuntimeError('Action watchdog expired')
                    if any(not lo < v+.15*d < hi for v,d,(lo,hi) in zip(q,action,limits)):
                        raise RuntimeError('Predicted joint limit violation')
                    c.command(action[:6])
                    # Output already models actual recorded finger commands. No extra
                    # expert closure/ramping or object-dependent feedback is applied.
                    c.grip.publish(c.Msg(data=[float(action[6])]*2))
                    cup, plate = c.poses[helpers.CUP], c.poses[helpers.SAUCER]
                    fresh = all(wall-c.pose_received.get(n,0) < 1 for n in (helpers.CUP,helpers.SAUCER))
                    error = math.hypot(cup[0]-plate[0], cup[1]-plate[1])
                    max_lift = max(max_lift, cup[2]-initial_cup[2])
                    success = fresh and error < .02 and abs(cup[2]-plate[2]-.029) < .01 and c.tilt[helpers.CUP] < .2 and min(c.finger_positions()) > -.002
                    stable_since = (stable_since if stable_since is not None else c.sim_time) if success else None
                    result.update(steps=count, clamped_steps=clamped, elapsed_sim_s=c.sim_time-start_sim,
                        max_cup_lift_m=max_lift, final_cup=cup, final_plate=plate, xy_error_m=error,
                        cup_tilt_rad=c.tilt[helpers.CUP], gripper_tilt_rad=tilt)
                    if c.sim_time-last_report > 5:
                        print(json.dumps(result), flush=True); last_report=c.sim_time
                    if stable_since is not None and c.sim_time-stable_since > 1:
                        result.update(status='success', success=True); break
                    if fresh and (cup[2] < .07 or c.tilt[helpers.CUP] > .6):
                        raise RuntimeError('Cup fell/tipped; ending trial')
                else:
                    result.update(status='task_not_completed', success=False)
            finally:
                if c is not None:
                    try:
                        c.command([0.]*6)
                        # Keep the stationary final scene for two wall seconds.
                        until = time.monotonic()+2
                        while time.monotonic() < until:
                            c.command([0.]*6)
                            rclpy.spin_once(c.node, timeout_sec=.02)
                    finally:
                        # Even a recording callback error must not bypass stop/cleanup.
                        accepting_records[0] = False
                        c.close(); c=None
    except BaseException as exc:
        result.update(status='stopped', success=False, reason=f'{type(exc).__name__}: {exc}')
    finally:
        sock.close()
        pool.shutdown(wait=True, cancel_futures=True)
        for worker in (image_writer, data_writer):
            try:
                worker.close()
            except Exception as exc:
                result.update(status='recording_failed', success=False, recording_error=str(exc))
        result.update(image_writer=image_writer.stats(), data_writer=data_writer.stats(),
                      latest_frame_overwrites=latest_slot.overwritten_unselected,
                      timing_check_changed=True, freshness_threshold_sim_s=.1,
                      observation_pause_events=recovery.events,
                      observation_recovery_sha256=hashlib.sha256(Path(__file__).with_name('observation_recovery.py').read_bytes()).hexdigest(),
                      recovery_timeout_wall_s=.5, max_observation_pauses=3,
                      minimum_pause_spacing_wall_s=10, image_qos_depth=1)
        # Do not close files underneath a worker if filesystem I/O is hung.
        if not image_writer.thread.is_alive():
            frames_log.close(); latency_log.close()
        if not data_writer.thread.is_alive():
            raw_log.close(); actions_log.close()
        result.update(frames=len(frame_times), wall_duration_s=time.monotonic()-start_wall)
        if len(frame_times)>1:
            result.update(recorded_sim_s=frame_times[-1][0]-frame_times[0][0],
                          rtf=(frame_times[-1][0]-frame_times[0][0])/(frame_times[-1][1]-frame_times[0][1]))
        (a.output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result), flush=True)
    return 0 if result['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
