#!/usr/bin/env python3
"""Live scene-only reset verification; records observed Gazebo poses and arm drift."""
import json
import math
import time
from pathlib import Path
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_msgs.msg import TFMessage
from cup_task import active_task, randomize, task_lock, WORLD, CUP, SAUCER

rclpy.init()
node = Node('verify_cup_resets')
poses, joints = {}, {}

def on_joints(msg):
    joints.update(zip(msg.name, msg.position))

def on_poses(msg):
    for tf in msg.transforms:
        if tf.child_frame_id in (CUP, SAUCER, 'left_ur5e_hande', 'right_ur5e_hande'):
            t, q = tf.transform.translation, tf.transform.rotation
            poses[tf.child_frame_id] = [t.x, t.y, t.z, math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))]

node.create_subscription(JointState, '/joint_states', on_joints, 20)
node.create_subscription(TFMessage, f'/world/{WORLD}/dynamic_pose/info', on_poses, 20)

def spin(seconds):
    end = time.monotonic()+seconds
    while time.monotonic() < end: rclpy.spin_once(node, timeout_sec=0.05)

try:
    spin(5)
    assert CUP in poses and SAUCER in poses, poses
    assert 'left_shoulder_pan_joint' in joints and 'right_shoulder_pan_joint' in joints
    task = active_task()
    results = []
    with task_lock():
        for seed in (1,2,3):
            before = dict(joints)
            expected = randomize(task, seed)
            spin(3)
            c, s = poses[CUP][:], poses[SAUCER][:]
            cfg = task['randomization']
            for p in (c, s):
                assert cfg['x_bounds'][0] <= p[0] <= cfg['x_bounds'][1], p
                assert cfg['y_bounds'][0] <= p[1] <= cfg['y_bounds'][1], p
            sep = math.hypot(c[0]-s[0],c[1]-s[1])
            assert sep >= cfg['minimum_separation'], (c,s,sep)
            assert abs(math.atan2(math.sin(c[3]-expected[CUP][3]), math.cos(c[3]-expected[CUP][3]))) < 0.05
            drift = max(abs(joints[n]-v) for n,v in before.items() if n.endswith('_joint') and 'finger' not in n)
            assert drift < 0.002, drift
            record = {'seed':seed,'observed_cup':c,'observed_saucer':s,'separation_m':sep,'max_arm_joint_drift_rad':drift,'passed':True}
            results.append(record)
            print(json.dumps(record),flush=True)
    print(json.dumps({'all_passed':True,'trials':results}),flush=True)
finally:
    node.destroy_node()
    rclpy.shutdown()
