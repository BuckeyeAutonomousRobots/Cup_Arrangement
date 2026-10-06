import time
from dataclasses import dataclass, field
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32, Int8
from teleop_bridge_msgs.msg import TargetTwistStates
from tf2_msgs.msg import TFMessage


@dataclass
class ArmHapticState:
    name: str
    joint_prefix: str
    base_xyz: np.ndarray
    base_xyzw: np.ndarray
    output_topic: str
    ee_base_pos: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    ee_base_xyzw: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0], dtype=float))
    last_ee_time: float = 0.0
    target_ee_base_pos: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    last_target_ee_time: float = 0.0
    servo_status: int = 0
    collision_gap_above_since: float = -1.0
    collision_gap_active: bool = False
    gripper_pos: float = 0.0
    gripper_cmd: int = 0
    tracked: bool = False
    last_cmd_time: float = 0.0
    pulse_stage: int = 0
    pulse_until: float = 0.0
    contact_armed: bool = True
    last_contact: bool = False


@dataclass(frozen=True)
class ObjectContactSpec:
    name: str
    shape: str
    half_extents: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    radius: float = 0.0
    length: float = 0.0
    padding: float = 0.0


@dataclass
class ObjectPose:
    pos: np.ndarray
    xyzw: np.ndarray


class GazeboContactHapticPublisher(Node):
    """Publishes Quest haptic amplitudes from Gazebo-authoritative object and robot state.

    This deliberately avoids Unity collision callbacks and EE error/gap haptics. The signal is
    event-like: when a closing gripper is near a dynamic task object in the Gazebo world, publish
    a short two-tap pulse on that arm's haptic topic.
    """

    def __init__(self):
        super().__init__("contact_haptic_publisher")

        self.declare_parameter("publish_rate_hz", 60.0)
        self.declare_parameter("gz_dynamic_pose_topic", "/world/ur_hande_dual_arm_tabletop/dynamic_pose/info")
        self.declare_parameter(
            "object_names",
            [
                "Sync_RedCube",
                "Sync_GreenCube",
                "Sync_RedCylinder",
                "Sync_GreenCylinder",
                "Sync_CableReceiverBox",
                "Sync_CableReceiverBox_Fixed",
                "Sync_CableRod",
                "Sync_CableRod_B",
                "Sync_Rubik2x2",
            ],
        )
        self.declare_parameter("scene_layout_sdf_path", "/workspace/simulation/worlds/ur_hande_dual_arm_tabletop.sdf")
        self.declare_parameter("pose_timeout_sec", 0.40)
        self.declare_parameter("contact_radius_m", 0.095)
        self.declare_parameter("contact_skin_m", 0.014)
        self.declare_parameter("near_surface_m", 0.055)
        self.declare_parameter("object_geometry_padding_m", 0.004)
        self.declare_parameter("rearm_gripper_open_m", -0.003)
        self.declare_parameter("contact_gripper_max_m", -0.004)
        self.declare_parameter("pulse_amplitude", 0.55)
        self.declare_parameter("pulse_on_sec", 0.075)
        self.declare_parameter("pulse_gap_sec", 0.055)
        self.declare_parameter("continuous_contact_amplitude", 0.0)
        self.declare_parameter("proximity_amplitude", 0.0)
        self.declare_parameter("enable_servo_collision_gap_haptics", True)
        self.declare_parameter("collision_gap_activation_threshold_m", 0.08)
        self.declare_parameter("collision_gap_release_threshold_m", 0.045)
        self.declare_parameter("collision_gap_hold_sec", 0.25)
        self.declare_parameter("collision_gap_max_m", 0.16)
        self.declare_parameter("collision_gap_amplitude", 0.35)
        self.declare_parameter("require_both_finger_contact_for_pulse", True)
        self.declare_parameter("enable_empty_close_contact_pulse", True)
        self.declare_parameter(
            "finger_probe_offsets_xyz",
            [
                0.0, 0.0, 0.035,
                0.0, 0.0, 0.070,
                0.018, 0.0, 0.080,
                -0.018, 0.0, 0.080,
                0.0, 0.018, 0.080,
                0.0, -0.018, 0.080,
            ],
        )
        self.declare_parameter(
            "left_finger_probe_offsets_xyz",
            [
                0.018, 0.0, 0.080,
                0.0, 0.018, 0.080,
            ],
        )
        self.declare_parameter(
            "right_finger_probe_offsets_xyz",
            [
                -0.018, 0.0, 0.080,
                0.0, -0.018, 0.080,
            ],
        )

        self.declare_parameter("left_base_xyz", [0.0, 0.60, 0.0])
        self.declare_parameter("left_base_xyzw", [0.0, 0.0, -0.7071068, 0.7071068])
        self.declare_parameter("right_base_xyz", [0.0, -0.60, 0.0])
        self.declare_parameter("right_base_xyzw", [0.0, 0.0, 0.7071068, 0.7071068])

        publish_rate_hz = max(1.0, float(self.get_parameter("publish_rate_hz").value))
        self.gz_dynamic_pose_topic = str(self.get_parameter("gz_dynamic_pose_topic").value)
        self.object_names = [str(v) for v in self.get_parameter("object_names").value]
        self.scene_layout_sdf_path = str(self.get_parameter("scene_layout_sdf_path").value)
        self.pose_timeout_sec = max(0.05, float(self.get_parameter("pose_timeout_sec").value))
        self.contact_radius_m = max(0.001, float(self.get_parameter("contact_radius_m").value))
        self.contact_skin_m = max(0.0, float(self.get_parameter("contact_skin_m").value))
        self.near_surface_m = max(self.contact_skin_m, float(self.get_parameter("near_surface_m").value))
        self.object_geometry_padding_m = max(0.0, float(self.get_parameter("object_geometry_padding_m").value))
        self.rearm_gripper_open_m = max(0.0, float(self.get_parameter("rearm_gripper_open_m").value))
        self.contact_gripper_max_m = max(0.0, float(self.get_parameter("contact_gripper_max_m").value))
        self.pulse_amplitude = self._clamp01(float(self.get_parameter("pulse_amplitude").value))
        self.pulse_on_sec = max(0.01, float(self.get_parameter("pulse_on_sec").value))
        self.pulse_gap_sec = max(0.01, float(self.get_parameter("pulse_gap_sec").value))
        self.continuous_contact_amplitude = self._clamp01(
            float(self.get_parameter("continuous_contact_amplitude").value)
        )
        self.proximity_amplitude = self._clamp01(float(self.get_parameter("proximity_amplitude").value))
        self.enable_servo_collision_gap_haptics = bool(
            self.get_parameter("enable_servo_collision_gap_haptics").value
        )
        self.collision_gap_activation_threshold_m = max(
            0.0, float(self.get_parameter("collision_gap_activation_threshold_m").value)
        )
        self.collision_gap_release_threshold_m = max(
            0.0, float(self.get_parameter("collision_gap_release_threshold_m").value)
        )
        self.collision_gap_hold_sec = max(0.0, float(self.get_parameter("collision_gap_hold_sec").value))
        self.collision_gap_max_m = max(
            self.collision_gap_activation_threshold_m + 0.001,
            float(self.get_parameter("collision_gap_max_m").value),
        )
        self.collision_gap_amplitude = self._clamp01(
            float(self.get_parameter("collision_gap_amplitude").value)
        )
        self.require_both_finger_contact_for_pulse = bool(
            self.get_parameter("require_both_finger_contact_for_pulse").value
        )
        self.enable_empty_close_contact_pulse = bool(
            self.get_parameter("enable_empty_close_contact_pulse").value
        )
        self.finger_probe_offsets = self._parse_probe_offsets(
            self.get_parameter("finger_probe_offsets_xyz").value
        )
        self.left_finger_probe_offsets = self._parse_probe_offsets(
            self.get_parameter("left_finger_probe_offsets_xyz").value
        )
        self.right_finger_probe_offsets = self._parse_probe_offsets(
            self.get_parameter("right_finger_probe_offsets_xyz").value
        )

        self.left = ArmHapticState(
            name="left",
            joint_prefix="left_",
            base_xyz=self._parse_vec3(self.get_parameter("left_base_xyz").value),
            base_xyzw=self._normalize_quat(self._parse_vec4(self.get_parameter("left_base_xyzw").value)),
            output_topic="/left_arm/haptics/contact_amplitude",
        )
        self.right = ArmHapticState(
            name="right",
            joint_prefix="right_",
            base_xyz=self._parse_vec3(self.get_parameter("right_base_xyz").value),
            base_xyzw=self._normalize_quat(self._parse_vec4(self.get_parameter("right_base_xyzw").value)),
            output_topic="/right_arm/haptics/contact_amplitude",
        )

        self._object_specs = self._load_object_specs_from_sdf(self.scene_layout_sdf_path)
        if self._object_specs:
            spec_names = set(self._object_specs.keys())
            self.object_names = [
                name
                for name in self.object_names
                if name in spec_names or any(spec_name.startswith(f"{name}_cubie_") for spec_name in spec_names)
            ]
        self._object_world_poses: dict[str, ObjectPose] = {}
        self._last_object_pose_time = 0.0
        self._last_log_time = 0.0

        self.left_pub = self.create_publisher(Float32, self.left.output_topic, 20)
        self.right_pub = self.create_publisher(Float32, self.right.output_topic, 20)

        self.create_subscription(TFMessage, self.gz_dynamic_pose_topic, self._on_gz_dynamic_pose, 50)
        self.create_subscription(JointState, "/joint_states", self._on_joint_states, 50)
        self.create_subscription(TargetTwistStates, "/left_arm/target_twist_states", self._on_left_target, 20)
        self.create_subscription(TargetTwistStates, "/right_arm/target_twist_states", self._on_right_target, 20)
        self.create_subscription(PoseStamped, "/left_arm/teleop/target_ee_pose", self._on_left_target_ee_pose, 20)
        self.create_subscription(PoseStamped, "/right_arm/teleop/target_ee_pose", self._on_right_target_ee_pose, 20)
        self.create_subscription(PoseStamped, "/left_arm/teleop/actual_ee_pose", self._on_left_ee_pose, 20)
        self.create_subscription(PoseStamped, "/right_arm/teleop/actual_ee_pose", self._on_right_ee_pose, 20)
        self.create_subscription(Int8, "/left_arm/servo_node/status", self._on_left_servo_status, 20)
        self.create_subscription(Int8, "/right_arm/servo_node/status", self._on_right_servo_status, 20)
        for object_name in self.object_names:
            self.create_subscription(
                PoseStamped,
                f"/unity_sync/{object_name}_pose",
                lambda msg, name=object_name: self._on_sync_object_pose(name, msg),
                10,
            )
        self.create_timer(1.0 / publish_rate_hz, self._publish_loop)

        self.get_logger().info(
            "GazeboContactHapticPublisher started: "
            f"objects={self.object_names}, specs={len(self._object_specs)}, "
            f"skin={self.contact_skin_m:.3f}, near={self.near_surface_m:.3f}, "
            f"both_finger_pulse={self.require_both_finger_contact_for_pulse}, "
            f"empty_close_pulse={self.enable_empty_close_contact_pulse}, "
            f"servo_collision_gap={self.enable_servo_collision_gap_haptics}, "
            f"outputs=({self.left.output_topic}, {self.right.output_topic})"
        )

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))

    @staticmethod
    def _parse_vec3(value) -> np.ndarray:
        arr = np.array(value, dtype=float)
        if arr.shape != (3,):
            return np.zeros(3, dtype=float)
        return arr

    @staticmethod
    def _parse_vec4(value) -> np.ndarray:
        arr = np.array(value, dtype=float)
        if arr.shape != (4,):
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        return arr

    @staticmethod
    def _parse_probe_offsets(value) -> list[np.ndarray]:
        arr = np.array(value, dtype=float).reshape(-1)
        if arr.size < 3 or arr.size % 3 != 0:
            return [np.zeros(3, dtype=float)]
        return [arr[i : i + 3].copy() for i in range(0, arr.size, 3)]

    @staticmethod
    def _normalize_quat(q: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(q))
        if norm < 1e-9:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        return q / norm

    @staticmethod
    def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return np.array(
            [
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ],
            dtype=float,
        )

    @staticmethod
    def _quat_conjugate(q: np.ndarray) -> np.ndarray:
        return np.array([-q[0], -q[1], -q[2], q[3]], dtype=float)

    def _quat_rotate_vector(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        qv = np.array([v[0], v[1], v[2], 0.0], dtype=float)
        return self._quat_multiply(self._quat_multiply(q, qv), self._quat_conjugate(q))[:3]

    def _on_gz_dynamic_pose(self, msg: TFMessage):
        latest = {}
        wanted = set(self.object_names)
        for tf in msg.transforms:
            child = str(tf.child_frame_id).strip()
            model_name = self._model_name_from_child(child)
            if not self._object_name_wanted(model_name, wanted):
                continue
            latest[model_name] = ObjectPose(
                pos=np.array(
                    [
                        float(tf.transform.translation.x),
                        float(tf.transform.translation.y),
                        float(tf.transform.translation.z),
                    ],
                    dtype=float,
                ),
                xyzw=self._normalize_quat(
                    np.array(
                        [
                            float(tf.transform.rotation.x),
                            float(tf.transform.rotation.y),
                            float(tf.transform.rotation.z),
                            float(tf.transform.rotation.w),
                        ],
                        dtype=float,
                    )
                ),
            )

        if latest:
            self._object_world_poses.update(latest)
            self._last_object_pose_time = time.monotonic()

    @staticmethod
    def _model_name_from_child(child: str) -> str:
        if "::" in child:
            return child.split("::", 1)[0]
        if "/" in child:
            return child.split("/", 1)[0]
        return child

    def _on_sync_object_pose(self, object_name: str, msg: PoseStamped):
        self._object_world_poses[object_name] = ObjectPose(
            pos=np.array(
                [
                    float(msg.pose.position.x),
                    float(msg.pose.position.y),
                    float(msg.pose.position.z),
                ],
                dtype=float,
            ),
            xyzw=self._normalize_quat(
                np.array(
                    [
                        float(msg.pose.orientation.x),
                        float(msg.pose.orientation.y),
                        float(msg.pose.orientation.z),
                        float(msg.pose.orientation.w),
                    ],
                    dtype=float,
                )
            ),
        )
        self._last_object_pose_time = time.monotonic()

    def _on_joint_states(self, msg: JointState):
        if msg.name is None or msg.position is None:
            return
        n = min(len(msg.name), len(msg.position))
        for i in range(n):
            name = str(msg.name[i])
            if name == "left_robotiq_hande_left_finger_joint":
                self.left.gripper_pos = float(msg.position[i])
            elif name == "right_robotiq_hande_left_finger_joint":
                self.right.gripper_pos = float(msg.position[i])

    def _on_left_target(self, msg: TargetTwistStates):
        self._on_target(self.left, msg)

    def _on_right_target(self, msg: TargetTwistStates):
        self._on_target(self.right, msg)

    @staticmethod
    def _on_target(arm: ArmHapticState, msg: TargetTwistStates):
        arm.gripper_cmd = int(msg.gripper_cmd)
        arm.tracked = bool(msg.tracked)
        arm.last_cmd_time = time.monotonic()

    def _on_left_ee_pose(self, msg: PoseStamped):
        self._on_ee_pose(self.left, msg)

    def _on_right_ee_pose(self, msg: PoseStamped):
        self._on_ee_pose(self.right, msg)

    def _on_left_target_ee_pose(self, msg: PoseStamped):
        self._on_target_ee_pose(self.left, msg)

    def _on_right_target_ee_pose(self, msg: PoseStamped):
        self._on_target_ee_pose(self.right, msg)

    def _on_left_servo_status(self, msg: Int8):
        self.left.servo_status = int(msg.data)

    def _on_right_servo_status(self, msg: Int8):
        self.right.servo_status = int(msg.data)

    @staticmethod
    def _on_ee_pose(arm: ArmHapticState, msg: PoseStamped):
        arm.ee_base_pos = np.array(
            [
                float(msg.pose.position.x),
                float(msg.pose.position.y),
                float(msg.pose.position.z),
            ],
            dtype=float,
        )
        arm.ee_base_xyzw = GazeboContactHapticPublisher._normalize_quat(
            np.array(
                [
                    float(msg.pose.orientation.x),
                    float(msg.pose.orientation.y),
                    float(msg.pose.orientation.z),
                    float(msg.pose.orientation.w),
                ],
                dtype=float,
            )
        )
        arm.last_ee_time = time.monotonic()

    @staticmethod
    def _on_target_ee_pose(arm: ArmHapticState, msg: PoseStamped):
        arm.target_ee_base_pos = np.array(
            [
                float(msg.pose.position.x),
                float(msg.pose.position.y),
                float(msg.pose.position.z),
            ],
            dtype=float,
        )
        arm.last_target_ee_time = time.monotonic()

    def _publish_loop(self):
        now = time.monotonic()
        left_amp, left_dist = self._compute_arm_amplitude(self.left, now)
        right_amp, right_dist = self._compute_arm_amplitude(self.right, now)
        self.left_pub.publish(Float32(data=float(left_amp)))
        self.right_pub.publish(Float32(data=float(right_amp)))

        if now - self._last_log_time > 2.0:
            self._last_log_time = now
            self.get_logger().info(
                f"haptics left={left_amp:.2f} d={left_dist:.3f} grip={self.left.gripper_pos:.3f} "
                f"right={right_amp:.2f} d={right_dist:.3f} grip={self.right.gripper_pos:.3f} "
                f"objects={len(self._object_world_poses)}"
            )

    def _compute_arm_amplitude(self, arm: ArmHapticState, now: float) -> tuple[float, float]:
        objects_fresh = (now - self._last_object_pose_time) <= self.pose_timeout_sec
        ee_fresh = (now - arm.last_ee_time) <= self.pose_timeout_sec
        cmd_fresh = (now - arm.last_cmd_time) <= self.pose_timeout_sec
        nearest_dist = float("inf")
        contact = False
        near = False
        finger_touch_contact = arm.gripper_pos <= self.contact_gripper_max_m

        if objects_fresh and ee_fresh and cmd_fresh and arm.tracked and self._object_world_poses:
            ee_world = arm.base_xyz + self._quat_rotate_vector(arm.base_xyzw, arm.ee_base_pos)
            ee_world_rot = self._normalize_quat(self._quat_multiply(arm.base_xyzw, arm.ee_base_xyzw))
            probe_points = [
                ee_world + self._quat_rotate_vector(ee_world_rot, offset)
                for offset in self.finger_probe_offsets
            ]
            left_probe_points = [
                ee_world + self._quat_rotate_vector(ee_world_rot, offset)
                for offset in self.left_finger_probe_offsets
            ]
            right_probe_points = [
                ee_world + self._quat_rotate_vector(ee_world_rot, offset)
                for offset in self.right_finger_probe_offsets
            ]
            nearest_dist = min(self._distance_to_task_objects(point) for point in probe_points)
            left_dist = min(self._distance_to_task_objects(point) for point in left_probe_points)
            right_dist = min(self._distance_to_task_objects(point) for point in right_probe_points)
            nearest_dist = min(nearest_dist, left_dist, right_dist)
            both_fingers_on_object = left_dist <= self.contact_skin_m and right_dist <= self.contact_skin_m
            near = nearest_dist <= self.near_surface_m
            object_contact = nearest_dist <= self.contact_skin_m
            contact = (
                both_fingers_on_object
                if self.require_both_finger_contact_for_pulse
                else object_contact
            )
            if self.enable_empty_close_contact_pulse and finger_touch_contact and not near:
                contact = True

            # Geometry parsing is preferred, but keep a center-distance fallback for hand-authored worlds.
            if not self._object_specs and not contact:
                center_dist = min(
                    float(np.linalg.norm(pose.pos - ee_world))
                    for pose in self._object_world_poses.values()
                )
                nearest_dist = min(nearest_dist, center_dist)
                near = nearest_dist <= self.contact_radius_m
                contact = nearest_dist <= self.contact_radius_m and arm.gripper_pos <= self.contact_gripper_max_m
        elif cmd_fresh and self.enable_empty_close_contact_pulse and finger_touch_contact:
            contact = True
            nearest_dist = 0.0

        if arm.gripper_cmd < 0 or arm.gripper_pos >= self.rearm_gripper_open_m:
            arm.contact_armed = True

        if contact and arm.gripper_cmd > 0 and arm.contact_armed and not arm.last_contact:
            arm.pulse_stage = 1
            arm.pulse_until = now + self.pulse_on_sec
            arm.contact_armed = False
            self.get_logger().warn(f"{arm.name} gripper contact pulse triggered at d={nearest_dist:.3f}m")

        arm.last_contact = contact
        pulse = self._pulse_amplitude(arm, now)
        if pulse > 0.0:
            return pulse, nearest_dist

        if contact and self.continuous_contact_amplitude > 0.0:
            return self.continuous_contact_amplitude, nearest_dist
        if near and arm.gripper_cmd > 0 and self.proximity_amplitude > 0.0:
            t = 1.0 - min(1.0, max(0.0, nearest_dist - self.contact_skin_m) / max(1e-6, self.near_surface_m - self.contact_skin_m))
            return self.proximity_amplitude * t, nearest_dist
        collision_gap_amp = self._servo_collision_gap_amplitude(arm, now)
        if collision_gap_amp > 0.0:
            return collision_gap_amp, nearest_dist
        return 0.0, nearest_dist

    def _servo_collision_gap_amplitude(self, arm: ArmHapticState, now: float) -> float:
        if not self.enable_servo_collision_gap_haptics:
            arm.collision_gap_active = False
            arm.collision_gap_above_since = -1.0
            return 0.0

        # MoveIt Servo status codes: 3=decelerate for collision, 4=halt for collision.
        collision_status = arm.servo_status in (3, 4)
        target_fresh = (now - arm.last_target_ee_time) <= self.pose_timeout_sec
        actual_fresh = (now - arm.last_ee_time) <= self.pose_timeout_sec
        if not collision_status or not target_fresh or not actual_fresh:
            arm.collision_gap_active = False
            arm.collision_gap_above_since = -1.0
            return 0.0

        gap = float(np.linalg.norm(arm.target_ee_base_pos - arm.ee_base_pos))
        activation = self.collision_gap_activation_threshold_m
        release = min(self.collision_gap_release_threshold_m, activation)
        if gap >= activation:
            if arm.collision_gap_above_since < 0.0:
                arm.collision_gap_above_since = now
            if now - arm.collision_gap_above_since >= self.collision_gap_hold_sec:
                arm.collision_gap_active = True
        elif gap <= release:
            arm.collision_gap_active = False
            arm.collision_gap_above_since = -1.0

        if not arm.collision_gap_active:
            return 0.0

        t = min(1.0, max(0.0, (gap - activation) / max(1e-6, self.collision_gap_max_m - activation)))
        return self.collision_gap_amplitude * t

    def _distance_to_task_objects(self, point_world: np.ndarray) -> float:
        nearest = float("inf")
        for object_name, pose in self._object_world_poses.items():
            spec = self._object_specs.get(object_name)
            if spec is None:
                nearest = min(nearest, float(np.linalg.norm(point_world - pose.pos)))
                continue

            local = self._quat_rotate_vector(self._quat_conjugate(pose.xyzw), point_world - pose.pos)
            if spec.shape == "box":
                dist = self._distance_to_box_surface(local, spec.half_extents + spec.padding)
            elif spec.shape == "cylinder":
                dist = self._distance_to_cylinder_surface(local, spec.radius + spec.padding, spec.length + (2.0 * spec.padding))
            else:
                dist = float(np.linalg.norm(point_world - pose.pos))
            nearest = min(nearest, dist)
        return nearest

    @staticmethod
    def _distance_to_box_surface(local: np.ndarray, half_extents: np.ndarray) -> float:
        outside = np.maximum(np.abs(local) - half_extents, 0.0)
        return float(np.linalg.norm(outside))

    @staticmethod
    def _distance_to_cylinder_surface(local: np.ndarray, radius: float, length: float) -> float:
        radial = float(np.linalg.norm(local[:2]))
        radial_outside = max(radial - radius, 0.0)
        z_outside = max(abs(float(local[2])) - (length * 0.5), 0.0)
        return float(np.hypot(radial_outside, z_outside))

    def _load_object_specs_from_sdf(self, sdf_path: str) -> dict[str, ObjectContactSpec]:
        path = Path(sdf_path)
        if not path.exists():
            self.get_logger().warn(f"Haptic object geometry SDF not found: {sdf_path}")
            return {}

        wanted = set(self.object_names)
        specs: dict[str, ObjectContactSpec] = {}
        try:
            root = ET.parse(path).getroot()
        except Exception as exc:
            self.get_logger().warn(f"Failed to parse haptic object geometry from {sdf_path}: {exc}")
            return {}

        world = root.find("world") or root.find(".//world")
        if world is None:
            return {}

        for model in world.findall("model"):
            name = str(model.get("name", "")).strip()
            if not name or not self._object_name_wanted(name, wanted):
                continue

            geometry = model.find(".//collision/geometry") or model.find(".//visual/geometry")
            if geometry is None:
                continue

            box = geometry.find("box")
            cylinder = geometry.find("cylinder")
            if box is not None:
                size_text = box.findtext("size", default="")
                try:
                    size = np.array([float(v) for v in size_text.split()], dtype=float)
                    if size.shape == (3,):
                        specs[name] = ObjectContactSpec(
                            name=name,
                            shape="box",
                            half_extents=size * 0.5,
                            padding=self.object_geometry_padding_m,
                        )
                except ValueError:
                    continue
            elif cylinder is not None:
                try:
                    specs[name] = ObjectContactSpec(
                        name=name,
                        shape="cylinder",
                        radius=float(cylinder.findtext("radius", default="0.0")),
                        length=float(cylinder.findtext("length", default="0.0")),
                        padding=self.object_geometry_padding_m,
                    )
                except ValueError:
                    continue

        return specs

    @staticmethod
    def _object_name_wanted(model_name: str, wanted: set[str]) -> bool:
        if model_name in wanted:
            return True
        return any(model_name.startswith(f"{root}_cubie_") for root in wanted)

    def _pulse_amplitude(self, arm: ArmHapticState, now: float) -> float:
        if arm.pulse_stage == 0:
            return 0.0
        if now < arm.pulse_until:
            return self.pulse_amplitude if arm.pulse_stage in (1, 3) else 0.0
        if arm.pulse_stage == 1:
            arm.pulse_stage = 2
            arm.pulse_until = now + self.pulse_gap_sec
            return 0.0
        if arm.pulse_stage == 2:
            arm.pulse_stage = 3
            arm.pulse_until = now + self.pulse_on_sec
            return self.pulse_amplitude
        arm.pulse_stage = 0
        return 0.0


def main(args=None):
    rclpy.init(args=args)
    node = GazeboContactHapticPublisher()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    main()
