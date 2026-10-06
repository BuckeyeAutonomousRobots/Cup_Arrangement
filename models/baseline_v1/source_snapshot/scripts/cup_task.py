#!/usr/bin/env python3
"""Scene reset and privileged right-arm CHEAT baseline, for simulation only.

The baseline uses ground-truth poses and numerical URDF IK, but carries objects
only through simulated finger contact/friction. set_pose is for scene reset only.
"""
import argparse
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'simulation/tools'))
from cup_layout import sample_layout
WORLD = 'ur_hande_dual_arm_tabletop'
CUP, SAUCER = 'Sync_EspressoCup', 'Sync_Saucer'
CANCELLED = False


def active_task():
    scene_path = ROOT / (ROOT / 'profiles/active_scene_profile.txt').read_text().strip()
    scene = yaml.safe_load(scene_path.read_text())
    task = yaml.safe_load((scene_path.parent / scene['task_profile']).read_text())
    if task['task_profile'] != 'cup_arrangement':
        raise RuntimeError('Select cup_arrangement before running this helper')
    return task


def ign_service(endpoint, reqtype, request, reptype='ignition.msgs.Boolean'):
    result = subprocess.run(['ign', 'service', '-s', f'/world/{WORLD}/{endpoint}',
        '--reqtype', reqtype, '--reptype', reptype, '--timeout', '5000', '--req', request],
        capture_output=True, text=True, timeout=8, check=True)
    if reptype == 'ignition.msgs.Boolean' and not re.search(r'data:\s*true', result.stdout):
        raise RuntimeError(f'{endpoint} failed: {result.stdout} {result.stderr}')
    return result.stdout


def set_pose(name, pose):
    check_cancelled()
    x, y, z, yaw = pose
    ign_service('set_pose', 'ignition.msgs.Pose',
        f'name: "{name}" position: {{x: {x} y: {y} z: {z}}} '
        f'orientation: {{z: {math.sin(yaw/2)} w: {math.cos(yaw/2)}}}')


@contextmanager
def task_lock():
    with open('/tmp/cup_arrangement.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def randomize(task, seed):
    layout = sample_layout(task, seed)
    # Only these two models are moved. No world reset or robot commands.
    for name, pose in layout.items():
        set_pose(name, pose)
    print(json.dumps({'action': 'scene_only_reset', 'seed': seed, 'requested_poses': layout}), flush=True)
    return layout


def interrupted(signum, frame):
    # Never raise through rclpy's pybind callbacks; cancel at Python boundaries.
    global CANCELLED
    CANCELLED = True


def check_cancelled():
    if CANCELLED:
        raise KeyboardInterrupt('Cancellation requested')


@contextmanager
def exclusive_commands():
    """Suspend known input/reset publishers and right Servo inside our container.

    Preserve exact process state, including teleop configuration; no process is
    restarted. The left Servo remains alive and times out to a stationary hold.
    """
    stopped = []
    tokens = {'hand_pose_mapper', 'servo_command_bridge', 'simple_gripper_command_bridge',
              'coupled_gripper_controller', 'reset_manager', 'keyboard_servo_override',
              'debug_hand_generator', 'quest_controller_receiver', 'runtime_task_manager'}
    try:
        for path in Path('/proc').glob('[0-9]*/cmdline'):
            try:
                args = path.read_bytes().decode().split('\0')
                # Select actual executables, never shells containing command text.
                is_node = any(Path(arg).name in tokens for arg in args[:2])
                is_right_servo = any(Path(arg).name == 'servo_node_main' for arg in args[:2]) and '__ns:=/right_arm' in args
                if is_node or is_right_servo:
                    pid = int(path.parent.name)
                    state = (path.parent / 'status').read_text()
                    if re.search(r'^State:\s+T', state, re.M):
                        continue  # Do not resume a process that was already stopped.
                    os.kill(pid, signal.SIGSTOP)
                    stopped.append(pid)
            except (FileNotFoundError, ProcessLookupError):
                continue
        if not stopped:
            raise RuntimeError('No teleoperation processes found; run bringup_dual first')
        print(json.dumps({'ownership': 'acquired', 'paused_pids': stopped}), flush=True)
        time.sleep(0.5)
        check_cancelled()
        yield
    finally:
        for pid in stopped:
            try:
                os.kill(pid, signal.SIGCONT)
            except ProcessLookupError:
                pass
        print(json.dumps({'ownership': 'released', 'resumed_pids': stopped}), flush=True)


class GroundTruthController:
    def __init__(self, task):
        import numpy as np
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
        from tf2_msgs.msg import TFMessage
        from std_msgs.msg import Float64MultiArray
        self.np, self.ros, self.Msg = np, rclpy, Float64MultiArray
        from rclpy.signals import SignalHandlerOptions
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        self.node = Node('cup_cheat_baseline', namespace='/right_arm')
        self.q = {}
        self.poses = {}
        self.pose_received = {}
        self.tilt = {}
        self.sim_time = 0.0
        self.grasp_relative = None
        self.last_joint = 0
        self.task = task
        self.gripper_target = 0.0
        self.node.create_subscription(JointState, '/joint_states', self.on_joints, 20)
        self.node.create_subscription(TFMessage, f'/world/{WORLD}/dynamic_pose/info', self.on_poses, 20)
        self.vel = self.node.create_publisher(Float64MultiArray, '/right_joint_group_velocity_controller/commands', 10)
        self.grip = self.node.create_publisher(Float64MultiArray, '/right_hande_position_controller/commands', 10)
        urdf = ET.parse('/tmp/right_ur5e_hande_dual.urdf')
        # Derive open inner-face spacing from the actual collision proxies.
        faces = []
        for side, sign in (('left', 1), ('right', -1)):
            joint = urdf.find(f"joint[@name='right_robotiq_hande_{side}_finger_joint']")
            collision = urdf.find(f"link[@name='right_robotiq_hande_{side}_finger']/collision")
            x = float(joint.find('origin').get('xyz').split()[0])
            x += float(collision.find('origin').get('xyz').split()[0])
            x += sign*float(collision.find('geometry/box').get('size').split()[0])/2
            faces.append(x)
        self.open_finger_gap = faces[1]-faces[0]
        self.cup_diameter = 2*float(next(o for o in task['objects'] if o['id']==CUP)['radius'])
        by_child = {j.find('child').get('link'): j for j in urdf.findall('joint')}
        self.chain = []
        child = 'right_tool0'
        while child in by_child:
            j = by_child[child]
            self.chain.insert(0, j)
            child = j.find('parent').get('link')
        self.names = [j.get('name') for j in self.chain if j.get('type') == 'revolute']
        if len(self.names) != 6 or not all(n.startswith('right_') for n in self.names):
            raise RuntimeError('Expected exactly six right-arm revolute joints')
        scene_path = ROOT / (ROOT / 'profiles/active_scene_profile.txt').read_text().strip()
        scene = yaml.safe_load(scene_path.read_text())
        robots = yaml.safe_load((scene_path.parent / scene['robot_profile']).read_text())['robots']
        self.spawn = next(r['spawn_pose_xyz_rpy'] for r in robots if r['arm_id'] == 'right_arm')
        self.check_publishers()
        self.wait(lambda: all(n in self.q for n in self.names) and CUP in self.poses and SAUCER in self.poses, 20)

    def check_publishers(self):
        allowed = {'servo_node', 'reset_manager', 'cup_cheat_baseline',
                   'coupled_gripper_controller', 'simple_gripper_command_bridge'}
        for topic in ('/right_joint_group_velocity_controller/commands', '/right_hande_position_controller/commands'):
            for info in self.node.get_publishers_info_by_topic(topic):
                if info.node_name not in allowed or info.node_namespace != '/right_arm':
                    raise RuntimeError(f'Unexpected competing publisher on {topic}: {info.node_namespace}/{info.node_name}')

    def wait(self, predicate, timeout):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            self.ros.spin_once(self.node, timeout_sec=0.03)
            check_cancelled()
            if predicate(): return
        raise TimeoutError('Timed out waiting for joint/object state')

    def on_joints(self, msg):
        self.q.update(zip(msg.name, msg.position))
        self.sim_time = msg.header.stamp.sec + msg.header.stamp.nanosec*1e-9
        if 'right_shoulder_pan_joint' in msg.name: self.last_joint = time.monotonic()

    def on_poses(self, msg):
        for tf in msg.transforms:
            if tf.child_frame_id in (CUP, SAUCER):
                t, q = tf.transform.translation, tf.transform.rotation
                self.poses[tf.child_frame_id] = [t.x, t.y, t.z, math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))]
                self.pose_received[tf.child_frame_id] = time.monotonic()
                self.tilt[tf.child_frame_id] = math.acos(max(-1.,min(1.,1-2*(q.x*q.x+q.y*q.y))))

    def fk(self, q):
        from scipy.spatial.transform import Rotation
        np = self.np
        result = np.eye(4)
        result[:3, :3] = Rotation.from_euler('xyz', self.spawn[3:]).as_matrix()
        result[:3, 3] = self.spawn[:3]
        index = 0
        for joint in self.chain:
            origin = joint.find('origin')
            transform = np.eye(4)
            if origin is not None:
                transform[:3, 3] = [float(v) for v in origin.get('xyz', '0 0 0').split()]
                transform[:3, :3] = Rotation.from_euler('xyz', [float(v) for v in origin.get('rpy', '0 0 0').split()]).as_matrix()
            result = result @ transform
            if joint.get('type') == 'revolute':
                axis = np.array([float(v) for v in joint.find('axis').get('xyz').split()])
                transform = np.eye(4)
                transform[:3, :3] = Rotation.from_rotvec(axis*q[index]).as_matrix()
                result = result @ transform
                index += 1
        return result

    def command(self, values):
        self.vel.publish(self.Msg(data=[float(v) for v in values]))

    def finger_positions(self):
        return [self.q['right_robotiq_hande_'+side+'_finger_joint'] for side in ('left', 'right')]

    def hold_gripper(self):
        fingers = self.finger_positions()
        target = self.gripper_target
        if target < 0:
            # A common aperture must stop closing when either jaw is blocked.
            # Do not drive the unblocked jaw deep into the cup or across its rail.
            target = max(target, max(fingers)-0.0005)
        self.grip.publish(self.Msg(data=[target, target]))

    @staticmethod
    def aperture_evidence(fingers, diameter, open_gap):
        if len(fingers)!=2 or not all(math.isfinite(x) for x in [*fingers,diameter,open_gap]):
            raise ValueError('Invalid grasp geometry/state')
        if not 0 < diameter < open_gap:
            raise ValueError('Cup must fit within the open collision gap')
        gap = open_gap+sum(fingers)
        expected_travel = (open_gap-diameter)/2
        minimum_travel = max(0., expected_travel-.0015)
        centered = abs(fingers[0]-fingers[1]) < .001
        aperture_ok = abs(gap-diameter) <= .0015
        both_moved = all(-x >= minimum_travel for x in fingers)
        return dict(ready=centered and aperture_ok and both_moved,
            finger_positions=list(fingers),gap_m=gap,cup_diameter_m=diameter,
            expected_each_travel_m=expected_travel,centered=centered,
            aperture_ok=aperture_ok,both_fingers_moved=both_moved)

    @staticmethod
    def closing_target(initial, target, elapsed, speed):
        if not math.isfinite(speed) or speed < 0:
            raise ValueError('Gripper closing speed must be finite and nonnegative')
        return max(target, initial-speed*max(0., elapsed)) if speed > 0 else target

    def set_gripper(self, target, verify_grasp=None):
        if verify_grasp is None:verify_grasp=target<0
        cfg = self.task['debug']
        closing_speed = float(cfg.get('gripper_closing_speed_m_s', 0.0)) if target < 0 else 0.0
        initial = max(self.finger_positions())
        start_sim = self.sim_time
        self.gripper_target = target
        start_wall = time.monotonic()
        deadline = start_wall+float(cfg.get('gripper_timeout_sec', 20))
        stable_since = None
        last_report = start_wall
        while time.monotonic()<deadline:
            check_cancelled()
            self.ros.spin_once(self.node, timeout_sec=0.02)
            if time.monotonic()-self.last_joint > 2:
                raise RuntimeError('Stale joint state during gripper action')
            self.gripper_target = self.closing_target(initial, target, self.sim_time-start_sim, closing_speed)
            self.command([0]*6)
            self.hold_gripper()
            fingers = self.finger_positions()
            centered = abs(fingers[0]-fingers[1])<0.001
            reached = max(abs(x-target) for x in fingers)<(.0001 if target<0 and not verify_grasp else .0005)
            # Contact may stop closure short of its target, but elapsed time
            # and symmetry alone are NOT evidence of a grasp.
            ramp_done = abs(self.gripper_target-target) < 1e-9
            evidence = self.aperture_evidence(fingers,self.cup_diameter,self.open_finger_gap) if verify_grasp else None
            acceptable = (ramp_done and evidence['ready']) if verify_grasp else (centered and reached)
            if evidence is not None and time.monotonic()-last_report >= 2:
                print(json.dumps({'grasp_aperture':evidence}),flush=True)
                last_report = time.monotonic()
            stable_since = (stable_since or time.monotonic()) if acceptable else None
            if stable_since is not None and time.monotonic()-stable_since>.5:
                print(json.dumps({'gripper_target':target, 'finger_positions':fingers, 'centered':True}),flush=True)
                return
        raise RuntimeError('Gripper failed measured aperture/centering gate: '+json.dumps(
            self.aperture_evidence(self.finger_positions(),self.cup_diameter,self.open_finger_gap)
            if target<0 else {'fingers':self.finger_positions(),'target':target}))

    def solve_ik(self, xyz, yaw, initial):
        from scipy.optimize import least_squares
        from scipy.spatial.transform import Rotation
        np = self.np
        # tool +Z points down at the platform.
        desired = Rotation.from_euler('xyz', [math.pi, 0, yaw]).as_matrix()
        joints = [j for j in self.chain if j.get('type') == 'revolute']
        lower = np.array([float(j.find('limit').get('lower')) for j in joints])
        upper = np.array([float(j.find('limit').get('upper')) for j in joints])
        # Keep the upper arm raised and select the positive-elbow branch.
        # These are task posture constraints, not changes to physical limits.
        upper[1] = min(upper[1], -0.15)
        lower[2] = max(lower[2], 0.15)
        def residual(q):
            actual = self.fk(q)
            return np.r_[actual[:3, 3]-xyz, 0.25*Rotation.from_matrix(desired @ actual[:3, :3].T).as_rotvec()]
        seeds = [np.array(initial), np.array([initial[0], -1.1, 1.2, -1.7, -math.pi/2, initial[5]])]
        candidates = []
        for seed in seeds:
            solution = least_squares(residual, np.clip(seed, lower+1e-6, upper-1e-6), bounds=(lower, upper), max_nfev=400)
            if np.linalg.norm(residual(solution.x)) < 0.001:
                candidates.append(solution.x)
        if not candidates:
            raise RuntimeError(f'Unreachable debug waypoint: {xyz}')
        return min(candidates, key=lambda q: np.linalg.norm(q-initial))

    def arm_speed_scale(self):
        scale = float(self.task['debug'].get('arm_speed_scale', 1.0))
        if not math.isfinite(scale) or not 0 < scale <= 1:
            raise ValueError('Global arm speed scale must be in (0, 1]')
        return scale

    def move(self, xyz, yaw=0.0, carry=False, speed_scale=1.0, position_tolerance=None, orientation_tolerance=None):
        if not math.isfinite(speed_scale) or not 0 < speed_scale <= 1:
            raise ValueError('Motion speed scale must be in (0, 1]')
        if carry:
            return self.move_level(xyz, yaw, speed_scale, position_tolerance)
        np = self.np
        global_scale = self.arm_speed_scale()
        target = self.solve_ik(xyz, yaw, np.array([self.q[n] for n in self.names]))
        print(json.dumps({'waypoint_xyz': list(xyz), 'target_joints': target.tolist(), 'posture': 'raised_shoulder_positive_elbow', 'speed_scale': speed_scale}), flush=True)
        deadline = time.monotonic()+self.task['debug']['waypoint_timeout_sec']/global_scale
        progress_time = time.monotonic()
        best_error = float('inf')
        while time.monotonic() < deadline:
            self.ros.spin_once(self.node, timeout_sec=0.02)
            check_cancelled()
            self.check_publishers()
            self.hold_gripper()
            if time.monotonic()-self.last_joint > 2:
                raise RuntimeError('Stale right-arm state; refusing motion')
            q = np.array([self.q[n] for n in self.names])
            error = target-q
            worst_error = float(np.max(np.abs(error)))
            if worst_error < best_error-0.005:
                best_error, progress_time = worst_error, time.monotonic()
            elif time.monotonic()-progress_time > 20/global_scale and worst_error > 0.025:
                raise RuntimeError('Joint tracking stalled: '+json.dumps(dict(zip(self.names, error.tolist()))))
            self.command(global_scale*speed_scale*np.clip(1.8*error, -self.task['debug']['max_joint_speed'], self.task['debug']['max_joint_speed']))
            if carry:
                self.check_physical_hold(q)
            position_ok = position_tolerance is None or np.linalg.norm(self.fk(q)[:3,3]-xyz) <= position_tolerance
            orientation_ok = orientation_tolerance is None or self.orientation_error(q, yaw)[1] <= orientation_tolerance
            if np.max(np.abs(error)) < 0.012 and position_ok and orientation_ok:
                self.command([0]*6)
                return
        raise TimeoutError(f'Right-arm waypoint timeout: {xyz}')

    def orientation_error(self, q, yaw):
        from scipy.spatial.transform import Rotation
        desired = Rotation.from_euler('xyz', [math.pi, 0, yaw]).as_matrix()
        error = Rotation.from_matrix(desired @ self.fk(q)[:3,:3].T).as_rotvec()
        return error, float(self.np.linalg.norm(error))

    def cartesian_jacobian(self, q):
        """World-frame geometric Jacobian of tool0, from the active URDF."""
        from scipy.spatial.transform import Rotation
        np = self.np
        current = self.fk(q)
        jac = np.zeros((6,6))
        for i in range(6):
            shifted = q.copy(); shifted[i] += 1e-5
            tf = self.fk(shifted)
            jac[:3,i] = (tf[:3,3]-current[:3,3])/1e-5
            jac[3:,i] = Rotation.from_matrix(tf[:3,:3] @ current[:3,:3].T).as_rotvec()/1e-5
        return jac

    def descend_centered(self, xyz, yaw):
        """Straight Cartesian approach; never use a joint-space arc at the rim."""
        np = self.np
        xyz = np.asarray(xyz, dtype=float)
        initial_cup = np.array(self.poses[CUP][:3])
        cfg = self.task['debug']
        global_scale = self.arm_speed_scale()
        deadline = time.monotonic()+float(cfg['waypoint_timeout_sec'])/global_scale
        progress_time = time.monotonic(); best_error = float('inf')
        last_report = -float('inf'); max_near_error = 0.
        cup_height = float(next(o for o in self.task['objects'] if o['id']==CUP)['height'])
        print(json.dumps({'phase':'centered_descent_start','sim_s':self.sim_time}),flush=True)
        while time.monotonic()<deadline:
            self.ros.spin_once(self.node,timeout_sec=.02)
            check_cancelled(); self.check_publishers(); self.hold_gripper()
            if time.monotonic()-self.last_joint>2 or time.monotonic()-self.pose_received.get(CUP,0)>2:
                raise RuntimeError('Stale state during centered descent')
            q = np.array([self.q[n] for n in self.names]); tf = self.fk(q)
            delta = xyz-tf[:3,3]; rotation, angle = self.orientation_error(q,yaw)
            relative = self.cup_in_tool(q)
            drift = float(np.linalg.norm(np.array(self.poses[CUP][:3])-initial_cup))
            if drift>.001 or self.tilt[CUP]>.03:
                raise RuntimeError('Cup disturbed before closure during centered descent')
            # Conservative approach corridor whenever tool0 is within 70 mm
            # of its final height (well before the finger tips meet the rim).
            near = tf[2,3]-xyz[2]<.070
            lateral = float(np.linalg.norm(relative[:2]))
            margin = (self.open_finger_gap+sum(self.finger_positions())-self.cup_diameter)/2
            if near:
                max_near_error = max(max_near_error,lateral)
                if lateral>min(.0015,margin-.001) or angle>.005:
                    raise RuntimeError('Unsafe rim approach alignment: '+str((lateral,angle,margin)))
            if self.sim_time-last_report>=.25:
                print(json.dumps({'phase':'centered_descent','sim_s':self.sim_time,
                    'tool_xyz':tf[:3,3].tolist(),'cup_in_tool':relative.tolist(),
                    'orientation_error_rad':angle,'cup_drift_m':drift,
                    'rim_world_z':initial_cup[2]+cup_height/2}),flush=True)
                last_report=self.sim_time
            distance=float(np.linalg.norm(delta))
            if distance<.0004 and angle<.0015:
                self.command([0]*6)
                print(json.dumps({'phase':'centered_descent_done','sim_s':self.sim_time,
                    'max_near_lateral_error_m':max_near_error,'position_error_m':distance,
                    'orientation_error_rad':angle,'cup_drift_m':drift}),flush=True)
                return
            if distance<best_error-.001:
                best_error=distance;progress_time=time.monotonic()
            elif time.monotonic()-progress_time>20/global_scale and distance>.001:
                raise RuntimeError('Centered descent stalled; refusing to push through contact')
            # Correct XY/orientation continuously, descend at <=20 mm/s sim.
            linear=2.*delta
            linear[:2]=np.clip(linear[:2],-.012,.012)
            linear[2]=np.clip(linear[2],-.020,.020)
            if np.linalg.norm(delta[:2])>.0008 or angle>.003:
                linear[2]=0.
            twist=np.r_[linear,np.clip(2.*rotation,-.08,.08)]
            jac=self.cartesian_jacobian(q)
            velocity=jac.T @ np.linalg.solve(jac@jac.T+1e-6*np.eye(6),twist)
            limit=float(cfg['max_joint_speed'])*float(cfg.get('approach_speed_scale',1.))
            velocity/=max(1.,float(np.max(np.abs(velocity)))/limit)
            joints=[j for j in self.chain if j.get('type')=='revolute']
            for i,j in enumerate(joints):
                lo=float(j.find('limit').get('lower'))+.01
                hi=float(j.find('limit').get('upper'))-.01
                if not lo<q[i]+.1*velocity[i]<hi:
                    raise RuntimeError('Centered descent would exceed joint limit')
            self.command(global_scale*velocity)
        raise TimeoutError('Centered descent timed out')

    def move_level(self, xyz, yaw, speed_scale=1.0, position_tolerance=None):
        """Carry along a Cartesian segment, holding tool +Z vertically down.

        Endpoint IK alone cannot prevent intermediate pitch/roll. Correct the
        full orientation continuously and stop if its error exceeds one degree.
        """
        np = self.np
        cfg = self.task['debug']; scale = self.arm_speed_scale()
        xyz = np.asarray(xyz,dtype=float)
        tolerance = .0005 if position_tolerance is None else min(.0005,position_tolerance)
        q = np.array([self.q[n] for n in self.names])
        origin = self.fk(q)[:3,3].copy()
        # Validate the whole segment on the same raised-shoulder/elbow branch.
        seed = q.copy()
        for alpha in np.linspace(0.,1.,max(2,int(np.linalg.norm(xyz-origin)/.02)+2)):
            seed = self.solve_ik(origin+alpha*(xyz-origin),yaw,seed)
        deadline = time.monotonic()+float(cfg['waypoint_timeout_sec'])/scale
        best_error = float('inf'); progress_time=time.monotonic()
        max_tilt=0.;max_angle=0.;last_report=-float('inf')
        print(json.dumps({'phase':'level_carry_start','sim_s':self.sim_time,
            'target_xyz':xyz.tolist(),'global_speed_scale':scale}),flush=True)
        while time.monotonic()<deadline:
            self.ros.spin_once(self.node,timeout_sec=.02)
            check_cancelled();self.check_publishers();self.hold_gripper()
            if time.monotonic()-self.last_joint>2:
                raise RuntimeError('Stale joint state during level carry')
            q=np.array([self.q[n] for n in self.names]);tf=self.fk(q)
            rotation,angle=self.orientation_error(q,yaw)
            tilt=math.acos(float(np.clip(-tf[2,2],-1.,1.)))
            max_tilt=max(max_tilt,tilt);max_angle=max(max_angle,angle)
            if angle>math.radians(1.):
                raise RuntimeError('Level carry orientation exceeded one degree')
            self.check_physical_hold(q)
            delta=xyz-tf[:3,3];distance=float(np.linalg.norm(delta))
            if self.sim_time-last_report>=.25:
                print(json.dumps({'phase':'level_carry','sim_s':self.sim_time,
                    'tool_xyz':tf[:3,3].tolist(),'tilt_rad':tilt,
                    'orientation_error_rad':angle}),flush=True)
                last_report=self.sim_time
            if distance<tolerance and angle<.0015:
                self.command([0]*6)
                print(json.dumps({'phase':'level_carry_done','sim_s':self.sim_time,
                    'max_tilt_rad':max_tilt,'max_orientation_error_rad':max_angle,
                    'position_error_m':distance}),flush=True)
                return
            if distance<best_error-.001:
                best_error=distance;progress_time=time.monotonic()
            elif time.monotonic()-progress_time>20/scale and distance>.001:
                raise RuntimeError('Level carry stalled')
            linear=2.*delta
            linear/=max(1.,float(np.linalg.norm(linear))/(.08*speed_scale))
            if angle>.003:linear[:]=0.
            twist=np.r_[linear,np.clip(2.*rotation,-.08,.08)]
            jac=self.cartesian_jacobian(q)
            velocity=jac.T @ np.linalg.solve(jac@jac.T+1e-6*np.eye(6),twist)
            limit=float(cfg['max_joint_speed'])*speed_scale
            velocity/=max(1.,float(np.max(np.abs(velocity)))/limit)
            joints=[j for j in self.chain if j.get('type')=='revolute']
            for i,j in enumerate(joints):
                lo=float(j.find('limit').get('lower'))+.01
                hi=float(j.find('limit').get('upper'))-.01
                if i==1:hi=min(hi,-.15)
                if i==2:lo=max(lo,.15)
                if not lo<q[i]+.1*scale*velocity[i]<hi:
                    raise RuntimeError('Level carry would exceed joint/posture limit')
            self.command(scale*velocity)
        raise TimeoutError('Level carry timed out')

    def cup_in_tool(self, q):
        tf = self.fk(q)
        return tf[:3,:3].T @ (self.np.array(self.poses[CUP][:3])-tf[:3,3])

    def check_physical_hold(self, q):
        if time.monotonic()-self.pose_received.get(CUP,0)>2:
            raise RuntimeError('Stale cup pose during physical carry')
        slip = float(self.np.linalg.norm(self.cup_in_tool(q)-self.grasp_relative))
        if slip > .035:
            raise RuntimeError(f'Physical grasp slipped: relative displacement {slip:.4f} m')

    def settle(self, simulation_seconds=1.0):
        start = self.sim_time
        deadline = time.monotonic()+20
        while self.sim_time-start < simulation_seconds:
            if time.monotonic()>deadline:
                raise RuntimeError('Simulation did not advance during settle')
            check_cancelled()
            self.command([0]*6)
            self.hold_gripper()
            self.ros.spin_once(self.node,timeout_sec=.02)

    def run(self, approach_only=False, flip_grasp=False):
        cfg = self.task['debug']
        release_gap = float(cfg.get('release_clearance_m', 0.0))
        if not math.isfinite(release_gap) or not 0 <= release_gap <= .01:
            raise ValueError('Release clearance must be between 0 and 10 mm')
        cup, saucer = self.poses[CUP][:], self.poses[SAUCER][:]
        # Handle extends along cup +X; jaw separation is tool X. Keep them
        # perpendicular; no cup pose commands are used during manipulation.
        grasp_yaw = math.remainder(cup[3]+math.pi/2, 2*math.pi)
        if flip_grasp:grasp_yaw=math.remainder(grasp_yaw+math.pi,2*math.pi)
        print(json.dumps({'cup_yaw':cup[3], 'grasp_yaw':grasp_yaw, 'handle_offset_deg':90}),flush=True)
        offset, clearance = cfg['tool_to_cup_m'], cfg['clearance_m']
        np = self.np
        start = np.array(cup[:3]); start[2] += offset
        finish = np.array(saucer[:3]); finish[2] += 0.004+0.025+offset+release_gap
        # Reject infeasible task poses before publishing any movement commands.
        seed = np.array([self.q[n] for n in self.names])
        for waypoint in (start+[0,0,clearance], start, start+[0,0,.01], finish+[0,0,clearance], finish):
            seed = self.solve_ik(waypoint, grasp_yaw, seed)
        self.command([0]*6)
        open_position=cfg.get('open_gripper_position_m',0.0)
        if open_position<0:self.set_gripper(open_position,verify_grasp=False)
        else:self.set_gripper(0.0)
        self.move(start+[0,0,clearance], grasp_yaw, position_tolerance=.0004, orientation_tolerance=.0015)
        self.descend_centered(start, grasp_yaw)
        relative = self.cup_in_tool(np.array([self.q[n] for n in self.names]))
        if np.linalg.norm(relative[:2])>.003 or abs(relative[2]-offset)>.005:
            raise RuntimeError('Cup moved or grasp alignment is invalid before closure: '+str(relative))
        if approach_only:
            print(json.dumps({'diagnostic':'approach_close','cup_in_tool':relative.tolist(),
                'cup_pose':self.poses[CUP], 'finger_positions':self.finger_positions(),
                'tool_xyz':self.fk(np.array([self.q[n] for n in self.names]))[:3,3].tolist()}),flush=True)
            try:
                self.set_gripper(-0.01)
                print(json.dumps({'diagnostic':'approach_close','closure_pass':True,'lift_performed':False}),flush=True)
            finally:
                self.command([0]*6)
                self.set_gripper(0.0)
            return
        self.set_gripper(-0.01)
        self.grasp_relative = self.cup_in_tool(np.array([self.q[n] for n in self.names]))
        # Aperture is only a prerequisite. Verify physical following with a
        # small lift before attempting the full transport height.
        before_lift = self.poses[CUP][2]
        self.move(start+[0,0,.01], grasp_yaw, carry=True,
                  speed_scale=cfg.get('initial_lift_speed_scale',1.0), position_tolerance=.002)
        self.settle(.2)
        self.check_physical_hold(np.array([self.q[n] for n in self.names]))
        validation_lift = self.poses[CUP][2]-before_lift
        print(json.dumps({'grasp_validation_lift_m':validation_lift}),flush=True)
        if validation_lift < .005:
            raise RuntimeError('Grasp validation failed: cup did not follow the 10 mm test lift')
        self.move(start+[0,0,clearance], grasp_yaw, carry=True, speed_scale=cfg.get('initial_lift_speed_scale', 1.0))
        lifted = self.poses[CUP][2]-cup[2]
        print(json.dumps({'physical_lift_m':lifted}),flush=True)
        if lifted < .08:
            raise RuntimeError('Physical lift failed: cup did not rise at least 80 mm')
        self.move(finish+[0,0,clearance], grasp_yaw, carry=True)
        self.move(finish, grasp_yaw, carry=True)
        if release_gap>0:
            # Hold stationary before opening; verify actual cup-bottom gap,
            # not just the nominal tool target (a grasp may settle slightly).
            self.settle(.2)
            self.check_physical_hold(np.array([self.q[n] for n in self.names]))
            c,s=self.poses[CUP],self.poses[SAUCER]
            actual_gap=c[2]-s[2]-.029
            if abs(actual_gap-release_gap)>.002 or math.hypot(c[0]-s[0],c[1]-s[1])>.005 or self.tilt[CUP]>.03:
                raise RuntimeError('Unsafe pose for small-drop release: '+str((actual_gap,c,s)))
            print(json.dumps({'phase':'small_drop_release','sim_s':self.sim_time,
                'requested_gap_m':release_gap,'actual_gap_m':actual_gap,
                'cup_pose':c,'cup_tilt_rad':self.tilt[CUP]}),flush=True)
        self.set_gripper(0.0)
        if release_gap>0:self.settle(.5)
        self.move(finish+[0,0,clearance], grasp_yaw)
        self.settle()
        c, s = self.poses[CUP], self.poses[SAUCER]
        error = math.hypot(c[0]-s[0],c[1]-s[1])
        fresh = all(time.monotonic()-self.pose_received.get(n,0)<2 for n in (CUP,SAUCER))
        success = fresh and error < 0.02 and abs((c[2]-s[2])-0.029) < 0.01 and self.tilt[CUP]<.2
        print(json.dumps({'baseline':'GROUND_TRUTH_PHYSICAL_CONTACT','success':success,'cup':c,'saucer':s,'xy_error':error,'cup_tilt_rad':self.tilt[CUP]}),flush=True)
        if not success: raise RuntimeError('Cup did not settle on saucer')

    def close(self):
        # Keep the ROS context alive until the final stop has been delivered.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        for _ in range(5):
            self.command([0]*6)
            self.ros.spin_once(self.node, timeout_sec=0.03)
        self.node.destroy_node()
        self.ros.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['randomize', 'sample', 'debug', 'debug_flip', 'approach_close', 'finger_clear'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--run-id', default='', help=argparse.SUPPRESS)
    args = parser.parse_args()
    task = active_task()
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    if args.action == 'sample':
        print(json.dumps(sample_layout(task, args.seed)))
        return
    with task_lock():
        if args.action == 'randomize':
            randomize(task, args.seed)
        else:
            with exclusive_commands():
                controller = GroundTruthController(task)
                try:
                    if args.action=='finger_clear':
                        # Diagnostic scene reset only, before any closure; no carrying.
                        set_pose(CUP,[.15,-.12,.126,0.])
                        controller.set_gripper(0.)
                        initial=[controller.q[n] for n in controller.names]
                        try:
                            for mode in ('limited','raw'):
                                controller.set_gripper(0.)
                                controller.gripper_target=-.005
                                deadline=time.monotonic()+12
                                while time.monotonic()<deadline:
                                    check_cancelled()
                                    controller.ros.spin_once(controller.node,timeout_sec=.02)
                                    if time.monotonic()-controller.last_joint>2:raise RuntimeError('Stale state')
                                    if max(abs(controller.q[n]-q) for n,q in zip(controller.names,initial))>.01:raise RuntimeError('Arm drift')
                                    controller.command([0.]*6)
                                    if mode=='limited':controller.hold_gripper()
                                    else:controller.grip.publish(controller.Msg(data=[-.005,-.005]))
                                print(json.dumps({'mode':mode,'fingers':controller.finger_positions(),'arm_stationary':True}),flush=True)
                        finally:controller.set_gripper(0.)
                    else:
                        controller.run(approach_only=args.action=='approach_close',flip_grasp=args.action=='debug_flip')
                finally:
                    controller.close()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(json.dumps({'interrupted': True, 'cleanup': 'completed'}), flush=True)
        sys.exit(130)
