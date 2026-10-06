#!/usr/bin/env python3
"""One bounded privileged-cheat episode; PNG observations and timestamped ROS data.

Never labels receipt-clock command timing as exact application timing. No topic
publishing here; the existing cheat controller exclusively owns robot commands.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import yaml

import cv2
import numpy as np
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, qos_profile_sensor_data
from rosidl_runtime_py.convert import message_to_ordereddict
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image, CameraInfo, JointState
from std_msgs.msg import Float64MultiArray
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import PoseStamped


def stamp(t):
    return t.sec + t.nanosec * 1e-9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    ap.add_argument('--max-seconds', type=float, default=600)
    ap.add_argument('--action', choices=['debug','debug_flip','approach_close','finger_clear'], default='debug')
    args = ap.parse_args()
    if not 10 <= args.max_seconds <= 600:
        ap.error('max-seconds must be 10..600')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'frames').mkdir()
    root = Path('/workspace')
    scene_path = root / (root/'profiles/active_scene_profile.txt').read_text().strip()
    scene = yaml.safe_load(scene_path.read_text())
    profiles = [scene_path] + [(scene_path.parent/scene[key]).resolve()
        for key in ('robot_profile', 'workspace_profile', 'task_profile')]
    files = ['scripts/cup_task.py', 'src/ur_hande_description/urdf/ur_hande.urdf.xacro',
             'profiles/active_scene_profile.txt', 'simulation/config/ur5e_gz_controllers_right.yaml']
    files += [str(p.relative_to(root)) for p in profiles]
    hashes = {}
    for name in files:
        data = (root / name).read_bytes()
        hashes[name] = hashlib.sha256(data).hexdigest()
        dest = out / 'configuration' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    for name in ('right_ur5e_hande_dual.urdf', 'left_ur5e_hande_dual.urdf'):
        (out / 'configuration' / name).write_bytes((Path('/tmp') / name).read_bytes())
    rclpy.init()
    node = rclpy.create_node('collection_smoke_recorder')
    stream = (out / 'robot_data.jsonl').open('w')
    frames = (out / 'frames.jsonl').open('w')
    events = (out / 'events.jsonl').open('w')
    clock = [None]
    counts = {}
    image_stamps, image_wall = [], []
    joint_last = {}
    interrupted = [False]
    signal.signal(signal.SIGTERM, lambda *_: interrupted.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: interrupted.__setitem__(0, True))

    def timing():
        return dict(receipt_sim_s=clock[0], receipt_monotonic_s=time.monotonic(), receipt_unix_ns=time.time_ns())

    def event(kind, **kwargs):
        events.write(json.dumps(dict(event=kind, **timing(), **kwargs))+'\n')
        events.flush()

    def record(topic, msg):
        if isinstance(msg, Clock):
            clock[0] = stamp(msg.clock)
        counts[topic] = counts.get(topic, 0)+1
        if isinstance(msg, JointState):
            for i, name in enumerate(msg.name):
                joint_last[name] = dict(stamp_s=stamp(msg.header.stamp),
                    position=msg.position[i] if i < len(msg.position) else None,
                    velocity=msg.velocity[i] if i < len(msg.velocity) else None)
        stream.write(json.dumps(dict(topic=topic, **timing(), message=message_to_ordereddict(msg)))+'\n')

    def frame(msg):
        if msg.encoding != 'rgb8':
            raise RuntimeError(f'Unexpected image encoding {msg.encoding}')
        rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)[:, :msg.width*3].reshape(msg.height,msg.width,3)
        index = len(image_stamps)
        filename = f'frames/{index:06d}.png'
        if not cv2.imwrite(str(out / filename), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
            raise RuntimeError('Image write failed')
        t = stamp(msg.header.stamp)
        image_stamps.append(t)
        image_wall.append(time.monotonic())
        # Convenience snapshot only; raw joint samples are retained for proper
        # per-joint interpolation offline. Ages expose stale/future samples.
        snap = {n: dict(v, age_s=t-v['stamp_s']) for n,v in joint_last.items()}
        frames.write(json.dumps(dict(index=index, file=filename, stamp_sim_s=t,
            frame_id=msg.header.frame_id, width=msg.width, height=msg.height,
            **timing(), latest_joint_snapshot=snap))+'\n')

    topics = [(Clock,'/clock'), (JointState,'/joint_states'),
        (CameraInfo,'/right_gripper_camera/camera_info'), (TFMessage,'/tf'),
        (Float64MultiArray,'/right_joint_group_velocity_controller/commands'),
        (Float64MultiArray,'/right_hande_position_controller/commands'),
        (Float64MultiArray,'/left_joint_group_velocity_controller/commands'),
        (Float64MultiArray,'/left_hande_position_controller/commands'),
        (PoseStamped,'/unity_sync/Sync_EspressoCup_pose'),
        (PoseStamped,'/unity_sync/Sync_Saucer_pose')]
    subs = [node.create_subscription(cls,topic,lambda m,t=topic:record(t,m),qos_profile_sensor_data) for cls,topic in topics]
    subs.append(node.create_subscription(TFMessage,'/tf_static',lambda m:record('/tf_static',m),
        QoSProfile(depth=100,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.TRANSIENT_LOCAL)))
    subs.append(node.create_subscription(Image,'/right_gripper_camera/image_raw',frame,qos_profile_sensor_data))
    proc = None
    log = (out / 'cheat.log').open('w')
    start = time.monotonic()
    status = 'startup_failed'
    try:
        event('recording_started')
        while time.monotonic()-start < 25 and not interrupted[0]:
            rclpy.spin_once(node,timeout_sec=0.05)
            if len(image_stamps)>=5 and len(joint_last)>=12 and counts.get('/tf_static') and counts.get('/right_gripper_camera/camera_info'):
                break
        else:
            raise RuntimeError('Camera/joints/calibration/static TF readiness failed')
        event('cheat_started', privileged_idealized_transport=False, control='ground_truth_physical_contact')
        proc = subprocess.Popen(['python3',str(root/'scripts/cup_task.py'),args.action],
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        deadline = time.monotonic()+args.max_seconds
        while proc.poll() is None and time.monotonic()<deadline and not interrupted[0]:
            rclpy.spin_once(node,timeout_sec=0.03)
        if proc.poll() is None:
            event('cheat_cancel_requested')
            os.killpg(proc.pid,signal.SIGTERM)
            cleanup_end=time.monotonic()+15
            while proc.poll() is None and time.monotonic()<cleanup_end:
                rclpy.spin_once(node,timeout_sec=0.03)
            if proc.poll() is None:
                raise RuntimeError('Cheat failed to exit after SIGTERM: inspect robot ownership immediately')
            status='interrupted' if interrupted[0] else 'timeout'
        else:
            prefix='controller' if args.action in ('debug','debug_flip') else 'diagnostic'
            status=prefix+('_pass' if proc.returncode==0 else '_failed')
        event('cheat_finished',returncode=proc.returncode,status=status)
        until=time.monotonic()+5
        while time.monotonic()<until:
            rclpy.spin_once(node,timeout_sec=0.03)
    finally:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM)
        event('recording_finished',status=status)
        result=dict(status=status,cheat_returncode=None if proc is None else proc.poll(),
            wall_duration_s=time.monotonic()-start,frames=len(image_stamps),topic_counts=counts,
            source_sha256=hashes,command_timing='Unstamped commands: receipt simulation-clock approximation, NOT exact application time',
            baseline='GROUND_TRUTH_PHYSICAL_CONTACT',action=args.action,scene_profile=scene.get('scene_profile'),
            posture_optimized=False)
        if len(image_stamps)>1:
            ds=np.diff(image_stamps)
            result.update(sim_duration_s=image_stamps[-1]-image_stamps[0],
                wall_fps=(len(image_stamps)-1)/(image_wall[-1]-image_wall[0]),
                sim_fps=(len(image_stamps)-1)/(image_stamps[-1]-image_stamps[0]),
                rtf=(image_stamps[-1]-image_stamps[0])/(image_wall[-1]-image_wall[0]),
                nonmonotonic_image_stamps=int(np.sum(ds<=0)),gaps_over_50ms=int(np.sum(ds>0.05)),max_gap_sim_s=float(ds.max()))
        (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2),flush=True)
        stream.close();frames.close();events.close();log.close()
        node.destroy_node();rclpy.shutdown()
    return 0 if status in ('controller_pass','diagnostic_pass') else 1


if __name__=='__main__':
    raise SystemExit(main())
