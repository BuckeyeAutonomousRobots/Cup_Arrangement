import subprocess
import threading
import time
from dataclasses import dataclass

import numpy as np
import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Empty, Trigger
from teleop_bridge_msgs.msg import TargetTwistStates

from teleop_bridge.common.scene_layout import load_scene_object_poses_from_sdf


@dataclass(frozen=True)
class SceneResetTarget:
    model_name: str
    pose_xyz: np.ndarray
    pose_xyzw: np.ndarray


class TargetTwistResetManager(Node):
    def __init__(self):
        super().__init__("reset_manager")

        self.declare_parameter("input_topic", "/target_twist_states")
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("arm_velocity_topic", "/joint_group_velocity_controller/commands")
        self.declare_parameter("gripper_topic", "/hande_position_controller/commands")

        self.declare_parameter(
            "home_joint_names",
            [
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
        )
        self.declare_parameter("home_joint_positions", [0.0, 0.0, 1.57079632679, 0.0, 0.0, 0.0])

        self.declare_parameter("control_rate_hz", 40.0)
        self.declare_parameter("kick_velocities", [-0.35, 0.18, 0.28, 0.18, -0.28, -0.25])
        self.declare_parameter("kick_duration_sec", 1.5)
        self.declare_parameter("home_kp", 1.2)
        self.declare_parameter("home_max_vel", 0.60)
        self.declare_parameter("home_tolerance", 0.03)
        self.declare_parameter("home_timeout_sec", 8.0)
        self.declare_parameter("settle_sec", 0.3)

        self.declare_parameter("gripper_open_pos", 0.025)
        self.declare_parameter("gripper_command_mode", "position")
        self.declare_parameter("gripper_open_velocity", 0.03)
        self.declare_parameter("gripper_open_effort", 12.0)
        self.declare_parameter("gripper_hold_open_sec", 0.6)
        self.declare_parameter("gripper_joint_count", 1)
        self.declare_parameter("gripper_second_joint_negate", False)
        self.declare_parameter("min_reset_interval_sec", 2.5)

        self.declare_parameter("start_servo_service", "/servo_node/start_servo")
        self.declare_parameter("stop_servo_service", "/servo_node/stop_servo")
        self.declare_parameter("reset_servo_status_service", "/servo_node/reset_servo_status")
        self.declare_parameter("start_servo_retries", 8)
        self.declare_parameter("start_servo_retry_sleep_sec", 0.5)

        self.declare_parameter("reset_cube_with_ign_service", True)
        self.declare_parameter("world_name", "ur_hande_tabletop")
        self.declare_parameter("cube_model_name", "target_cube")
        self.declare_parameter("cube_reset_pose_xyz", [0.60, 0.25, 0.023])
        self.declare_parameter("cube_reset_pose_xyzw", [0.0, 0.0, 0.0, 1.0])
        self.declare_parameter("reset_scene_objects_with_ign_service", False)
        self.declare_parameter(
            "scene_layout_sdf_path",
            "/workspace/simulation/worlds/ur_hande_tabletop.sdf",
        )
        self.declare_parameter("scene_reset_lift_z", 0.01)
        self.declare_parameter("scene_reset_repeat_count", 3)
        self.declare_parameter("scene_reset_repeat_interval_sec", 0.06)
        # Use a string-typed default so ROS 2 does not infer BYTE_ARRAY from an empty list.
        self.declare_parameter("scene_reset_specs", [""])

        input_topic = str(self.get_parameter("input_topic").value)
        joint_states_topic = str(self.get_parameter("joint_states_topic").value)
        self.arm_velocity_topic = str(self.get_parameter("arm_velocity_topic").value)
        self.gripper_topic = str(self.get_parameter("gripper_topic").value)

        self.home_joint_names = [str(x) for x in self.get_parameter("home_joint_names").value]
        self.home_joint_positions = np.array(self.get_parameter("home_joint_positions").value, dtype=float)

        self.control_rate_hz = max(5.0, float(self.get_parameter("control_rate_hz").value))
        self.kick_velocities = np.array(self.get_parameter("kick_velocities").value, dtype=float)
        self.kick_duration_sec = max(0.0, float(self.get_parameter("kick_duration_sec").value))
        self.home_kp = max(0.0, float(self.get_parameter("home_kp").value))
        self.home_max_vel = max(0.0, float(self.get_parameter("home_max_vel").value))
        self.home_tolerance = max(0.001, float(self.get_parameter("home_tolerance").value))
        self.home_timeout_sec = max(0.1, float(self.get_parameter("home_timeout_sec").value))
        self.settle_sec = max(0.0, float(self.get_parameter("settle_sec").value))
        self.gripper_open_pos = float(self.get_parameter("gripper_open_pos").value)
        self.gripper_command_mode = str(self.get_parameter("gripper_command_mode").value).strip().lower()
        if self.gripper_command_mode not in ("position", "velocity", "effort", "effort_position"):
            self.get_logger().warn(
                f"Invalid gripper_command_mode={self.gripper_command_mode!r}; using 'position'."
            )
            self.gripper_command_mode = "position"
        self.gripper_open_velocity = max(0.0, float(self.get_parameter("gripper_open_velocity").value))
        self.gripper_open_effort = max(0.0, float(self.get_parameter("gripper_open_effort").value))
        self.gripper_hold_open_sec = max(0.0, float(self.get_parameter("gripper_hold_open_sec").value))
        self.gripper_joint_count = max(1, int(self.get_parameter("gripper_joint_count").value))
        self.gripper_second_joint_negate = bool(self.get_parameter("gripper_second_joint_negate").value)
        self.min_reset_interval_sec = max(0.0, float(self.get_parameter("min_reset_interval_sec").value))

        self.start_servo_service = str(self.get_parameter("start_servo_service").value)
        self.stop_servo_service = str(self.get_parameter("stop_servo_service").value)
        self.reset_servo_status_service = str(self.get_parameter("reset_servo_status_service").value)
        self.start_servo_retries = max(1, int(self.get_parameter("start_servo_retries").value))
        self.start_servo_retry_sleep_sec = max(0.05, float(self.get_parameter("start_servo_retry_sleep_sec").value))

        self.reset_cube_with_ign_service = bool(self.get_parameter("reset_cube_with_ign_service").value)
        self.world_name = str(self.get_parameter("world_name").value)
        self.cube_model_name = str(self.get_parameter("cube_model_name").value)
        self.cube_reset_pose_xyz = np.array(self.get_parameter("cube_reset_pose_xyz").value, dtype=float)
        self.cube_reset_pose_xyzw = np.array(self.get_parameter("cube_reset_pose_xyzw").value, dtype=float)
        self.reset_scene_objects_with_ign_service = bool(
            self.get_parameter("reset_scene_objects_with_ign_service").value
        )
        self.scene_layout_sdf_path = str(self.get_parameter("scene_layout_sdf_path").value)
        self.scene_reset_lift_z = self._parse_nonnegative(
            self.get_parameter("scene_reset_lift_z").value,
            "scene_reset_lift_z",
        )
        self.scene_reset_repeat_count = self._parse_int_min(
            self.get_parameter("scene_reset_repeat_count").value,
            "scene_reset_repeat_count",
            1,
        )
        self.scene_reset_repeat_interval_sec = self._parse_nonnegative(
            self.get_parameter("scene_reset_repeat_interval_sec").value,
            "scene_reset_repeat_interval_sec",
        )
        self.scene_reset_target_overrides = self._parse_scene_reset_specs(
            self.get_parameter("scene_reset_specs").value,
            "scene_reset_specs",
        )
        self.scene_reset_targets = self._resolve_scene_reset_targets(
            self.scene_reset_target_overrides,
            self.scene_layout_sdf_path,
        )

        if len(self.home_joint_names) != 6 or self.home_joint_positions.shape != (6,):
            self.get_logger().warn("Invalid home joint configuration; using 6-joint default.")
            self.home_joint_names = [
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ]
            self.home_joint_positions = np.array([0.0, 0.0, 1.57079632679, 0.0, 0.0, 0.0], dtype=float)

        if self.kick_velocities.shape != (6,):
            self.get_logger().warn("Invalid kick_velocities; using default 6-joint vector.")
            self.kick_velocities = np.array([-0.35, 0.18, 0.28, 0.18, -0.28, -0.25], dtype=float)

        self._arm_pub = self.create_publisher(Float64MultiArray, self.arm_velocity_topic, 20)
        self._gripper_pub = self.create_publisher(Float64MultiArray, self.gripper_topic, 20)
        self.create_subscription(TargetTwistStates, input_topic, self._on_target_twist, 20)
        self.create_subscription(JointState, joint_states_topic, self._on_joint_states, 20)

        self._stop_client = self.create_client(Trigger, self.stop_servo_service)
        self._start_client = self.create_client(Trigger, self.start_servo_service)
        self._reset_status_client = self.create_client(Empty, self.reset_servo_status_service)

        self._latest_joint_positions = {}
        self._have_joint_state = False
        self._last_reset_enable = False
        self._last_reset_robot_enable = False
        self._last_reset_scene_enable = False
        self._reset_lock = threading.Lock()
        self._reset_in_progress = False
        self._cancel_reset_requested = False
        self._last_reset_start_time = -1e9

        self.add_on_set_parameters_callback(self._on_parameter_change)

        self.get_logger().info(
            f"Reset manager started: input={input_topic}, arm_topic={self.arm_velocity_topic}, gripper_topic={self.gripper_topic}, "
            f"home={self.home_joint_positions.tolist()}, kick={self.kick_velocities.tolist()}, "
            f"gripper_command_mode={self.gripper_command_mode}, "
            f"gripper_joint_count={self.gripper_joint_count}, "
            f"gripper_second_joint_negate={self.gripper_second_joint_negate}, "
            f"cube={self.cube_model_name}@{self.cube_reset_pose_xyz.tolist()}, "
            f"scene_reset_targets={len(self.scene_reset_targets)}, "
            f"scene_reset_lift_z={self.scene_reset_lift_z:.4f}, "
            f"scene_reset_repeat_count={self.scene_reset_repeat_count}, "
            f"scene_reset_repeat_interval_sec={self.scene_reset_repeat_interval_sec:.3f}"
        )

    @staticmethod
    def _parse_vec(value, length: int, field_name: str) -> np.ndarray:
        vec = np.array(value, dtype=float)
        if vec.shape != (length,):
            raise ValueError(f"{field_name} must be a {length}-item numeric list.")
        return vec

    @staticmethod
    def _parse_nonnegative(value, field_name: str) -> float:
        v = float(value)
        if v < 0.0:
            raise ValueError(f"{field_name} must be >= 0.0.")
        return v

    @staticmethod
    def _parse_min(value, field_name: str, minimum: float) -> float:
        v = float(value)
        if v < minimum:
            raise ValueError(f"{field_name} must be >= {minimum}.")
        return v

    @staticmethod
    def _parse_int_min(value, field_name: str, minimum: int) -> int:
        v = int(value)
        if v < minimum:
            raise ValueError(f"{field_name} must be >= {minimum}.")
        return v

    @staticmethod
    def _parse_scene_reset_specs(value, field_name: str):
        targets = []
        if value is None:
            return targets
        for i, raw in enumerate(value):
            spec = str(raw).strip()
            if not spec:
                continue
            parts = [part.strip() for part in spec.split("|")]
            if len(parts) != 8:
                raise ValueError(
                    f"{field_name}[{i}] must have 8 pipe-separated fields: "
                    "model|x|y|z|qx|qy|qz|qw"
                )
            model_name = parts[0]
            if not model_name:
                raise ValueError(f"{field_name}[{i}] model name cannot be empty.")
            try:
                xyz = np.array([float(v) for v in parts[1:4]], dtype=float)
                xyzw = np.array([float(v) for v in parts[4:8]], dtype=float)
            except ValueError as exc:
                raise ValueError(f"{field_name}[{i}] contains a non-numeric pose value.") from exc
            targets.append(SceneResetTarget(model_name=model_name, pose_xyz=xyz, pose_xyzw=xyzw))
        return targets

    def _resolve_scene_reset_targets(self, overrides, sdf_path: str):
        if overrides:
            return overrides

        scene_layout = load_scene_object_poses_from_sdf(sdf_path, name_prefix="Sync_")
        if not scene_layout:
            self.get_logger().warn(
                f"Scene reset setup file not found or unreadable: {sdf_path}. "
                "Scene-object reset will stay empty until setup is available."
            )
            return []

        targets = []
        for pose in scene_layout.values():
            if pose.is_static:
                continue
            targets.append(
                SceneResetTarget(
                    model_name=pose.name,
                    pose_xyz=np.array(pose.pose_xyz, dtype=float),
                    pose_xyzw=np.array(pose.pose_xyzw, dtype=float),
                )
            )
        return targets

    def _on_parameter_change(self, params):
        with self._reset_lock:
            if self._reset_in_progress:
                return SetParametersResult(
                    successful=False,
                    reason="Cannot update reset parameters while a reset sequence is in progress.",
                )

        new_home_joint_names = self.home_joint_names
        new_home_joint_positions = self.home_joint_positions
        new_control_rate_hz = self.control_rate_hz
        new_kick_velocities = self.kick_velocities
        new_kick_duration_sec = self.kick_duration_sec
        new_home_kp = self.home_kp
        new_home_max_vel = self.home_max_vel
        new_home_tolerance = self.home_tolerance
        new_home_timeout_sec = self.home_timeout_sec
        new_settle_sec = self.settle_sec
        new_gripper_open_pos = self.gripper_open_pos
        new_gripper_command_mode = self.gripper_command_mode
        new_gripper_open_velocity = self.gripper_open_velocity
        new_gripper_open_effort = self.gripper_open_effort
        new_gripper_hold_open_sec = self.gripper_hold_open_sec
        new_gripper_joint_count = self.gripper_joint_count
        new_gripper_second_joint_negate = self.gripper_second_joint_negate
        new_min_reset_interval_sec = self.min_reset_interval_sec
        new_world_name = self.world_name
        new_cube_model_name = self.cube_model_name
        new_cube_reset_pose_xyz = self.cube_reset_pose_xyz
        new_cube_reset_pose_xyzw = self.cube_reset_pose_xyzw
        new_reset_cube_with_ign_service = self.reset_cube_with_ign_service
        new_reset_scene_objects_with_ign_service = self.reset_scene_objects_with_ign_service
        new_scene_layout_sdf_path = self.scene_layout_sdf_path
        new_scene_reset_lift_z = self.scene_reset_lift_z
        new_scene_reset_repeat_count = self.scene_reset_repeat_count
        new_scene_reset_repeat_interval_sec = self.scene_reset_repeat_interval_sec
        new_scene_reset_target_overrides = self.scene_reset_target_overrides
        new_scene_reset_targets = self.scene_reset_targets
        touched = []

        try:
            for param in params:
                name = param.name
                value = param.value

                if name == "home_joint_names":
                    vals = [str(v) for v in value]
                    if len(vals) != 6:
                        raise ValueError("home_joint_names must be a 6-item string list.")
                    new_home_joint_names = vals
                    touched.append(name)
                elif name == "home_joint_positions":
                    new_home_joint_positions = self._parse_vec(value, 6, "home_joint_positions")
                    touched.append(name)
                elif name == "control_rate_hz":
                    new_control_rate_hz = self._parse_min(value, "control_rate_hz", 5.0)
                    touched.append(name)
                elif name == "kick_velocities":
                    new_kick_velocities = self._parse_vec(value, 6, "kick_velocities")
                    touched.append(name)
                elif name == "kick_duration_sec":
                    new_kick_duration_sec = self._parse_nonnegative(value, "kick_duration_sec")
                    touched.append(name)
                elif name == "home_kp":
                    new_home_kp = self._parse_nonnegative(value, "home_kp")
                    touched.append(name)
                elif name == "home_max_vel":
                    new_home_max_vel = self._parse_nonnegative(value, "home_max_vel")
                    touched.append(name)
                elif name == "home_tolerance":
                    new_home_tolerance = self._parse_min(value, "home_tolerance", 0.001)
                    touched.append(name)
                elif name == "home_timeout_sec":
                    new_home_timeout_sec = self._parse_min(value, "home_timeout_sec", 0.1)
                    touched.append(name)
                elif name == "settle_sec":
                    new_settle_sec = self._parse_nonnegative(value, "settle_sec")
                    touched.append(name)
                elif name == "gripper_open_pos":
                    new_gripper_open_pos = float(value)
                    touched.append(name)
                elif name == "gripper_command_mode":
                    mode = str(value).strip().lower()
                    if mode not in ("position", "velocity", "effort", "effort_position"):
                        raise ValueError(
                            "gripper_command_mode must be 'position', 'velocity', 'effort', or 'effort_position'."
                        )
                    new_gripper_command_mode = mode
                    touched.append(name)
                elif name == "gripper_open_velocity":
                    new_gripper_open_velocity = self._parse_nonnegative(value, "gripper_open_velocity")
                    touched.append(name)
                elif name == "gripper_open_effort":
                    new_gripper_open_effort = self._parse_nonnegative(value, "gripper_open_effort")
                    touched.append(name)
                elif name == "gripper_hold_open_sec":
                    new_gripper_hold_open_sec = self._parse_nonnegative(value, "gripper_hold_open_sec")
                    touched.append(name)
                elif name == "gripper_joint_count":
                    new_gripper_joint_count = self._parse_int_min(value, "gripper_joint_count", 1)
                    touched.append(name)
                elif name == "gripper_second_joint_negate":
                    new_gripper_second_joint_negate = bool(value)
                    touched.append(name)
                elif name == "min_reset_interval_sec":
                    new_min_reset_interval_sec = self._parse_nonnegative(value, "min_reset_interval_sec")
                    touched.append(name)
                elif name == "world_name":
                    new_world_name = str(value)
                    touched.append(name)
                elif name == "cube_model_name":
                    new_cube_model_name = str(value)
                    touched.append(name)
                elif name == "cube_reset_pose_xyz":
                    new_cube_reset_pose_xyz = self._parse_vec(value, 3, "cube_reset_pose_xyz")
                    touched.append(name)
                elif name == "cube_reset_pose_xyzw":
                    new_cube_reset_pose_xyzw = self._parse_vec(value, 4, "cube_reset_pose_xyzw")
                    touched.append(name)
                elif name == "reset_cube_with_ign_service":
                    new_reset_cube_with_ign_service = bool(value)
                    touched.append(name)
                elif name == "reset_scene_objects_with_ign_service":
                    new_reset_scene_objects_with_ign_service = bool(value)
                    touched.append(name)
                elif name == "scene_layout_sdf_path":
                    new_scene_layout_sdf_path = str(value)
                    touched.append(name)
                elif name == "scene_reset_lift_z":
                    new_scene_reset_lift_z = self._parse_nonnegative(value, "scene_reset_lift_z")
                    touched.append(name)
                elif name == "scene_reset_repeat_count":
                    new_scene_reset_repeat_count = self._parse_int_min(value, "scene_reset_repeat_count", 1)
                    touched.append(name)
                elif name == "scene_reset_repeat_interval_sec":
                    new_scene_reset_repeat_interval_sec = self._parse_nonnegative(
                        value,
                        "scene_reset_repeat_interval_sec",
                    )
                    touched.append(name)
                elif name == "scene_reset_specs":
                    new_scene_reset_target_overrides = self._parse_scene_reset_specs(value, "scene_reset_specs")
                    touched.append(name)
        except ValueError as exc:
            return SetParametersResult(successful=False, reason=str(exc))

        new_scene_reset_targets = self._resolve_scene_reset_targets(
            new_scene_reset_target_overrides,
            new_scene_layout_sdf_path,
        )

        self.home_joint_names = new_home_joint_names
        self.home_joint_positions = new_home_joint_positions
        self.control_rate_hz = new_control_rate_hz
        self.kick_velocities = new_kick_velocities
        self.kick_duration_sec = new_kick_duration_sec
        self.home_kp = new_home_kp
        self.home_max_vel = new_home_max_vel
        self.home_tolerance = new_home_tolerance
        self.home_timeout_sec = new_home_timeout_sec
        self.settle_sec = new_settle_sec
        self.gripper_open_pos = new_gripper_open_pos
        self.gripper_command_mode = new_gripper_command_mode
        self.gripper_open_velocity = new_gripper_open_velocity
        self.gripper_open_effort = new_gripper_open_effort
        self.gripper_hold_open_sec = new_gripper_hold_open_sec
        self.gripper_joint_count = new_gripper_joint_count
        self.gripper_second_joint_negate = new_gripper_second_joint_negate
        self.min_reset_interval_sec = new_min_reset_interval_sec
        self.world_name = new_world_name
        self.cube_model_name = new_cube_model_name
        self.cube_reset_pose_xyz = new_cube_reset_pose_xyz
        self.cube_reset_pose_xyzw = new_cube_reset_pose_xyzw
        self.reset_cube_with_ign_service = new_reset_cube_with_ign_service
        self.reset_scene_objects_with_ign_service = new_reset_scene_objects_with_ign_service
        self.scene_layout_sdf_path = new_scene_layout_sdf_path
        self.scene_reset_lift_z = new_scene_reset_lift_z
        self.scene_reset_repeat_count = new_scene_reset_repeat_count
        self.scene_reset_repeat_interval_sec = new_scene_reset_repeat_interval_sec
        self.scene_reset_target_overrides = new_scene_reset_target_overrides
        self.scene_reset_targets = new_scene_reset_targets

        cube_pose_param_touched = (
            "cube_reset_pose_xyz" in touched
            or "cube_reset_pose_xyzw" in touched
            or "cube_model_name" in touched
            or "world_name" in touched
        )
        if cube_pose_param_touched and self.reset_cube_with_ign_service:
            self._reset_cube_pose_best_effort()
        if (
            "scene_reset_specs" in touched
            or "scene_layout_sdf_path" in touched
            or "scene_reset_lift_z" in touched
            or "world_name" in touched
        ) and self.reset_scene_objects_with_ign_service:
            self._reset_scene_objects_best_effort()

        if touched:
            self.get_logger().info(
                f"Live reset params updated ({', '.join(touched)}): "
                f"home={self.home_joint_positions.tolist()}, kick={self.kick_velocities.tolist()}, "
                f"gripper_command_mode={self.gripper_command_mode}, "
                f"gripper_joint_count={self.gripper_joint_count}, "
                f"gripper_second_joint_negate={self.gripper_second_joint_negate}, "
                f"cube={self.cube_model_name}@{self.cube_reset_pose_xyz.tolist()}, "
                f"scene_reset_targets={len(self.scene_reset_targets)}, "
                f"scene_reset_lift_z={self.scene_reset_lift_z:.4f}, "
                f"scene_reset_repeat_count={self.scene_reset_repeat_count}, "
                f"scene_reset_repeat_interval_sec={self.scene_reset_repeat_interval_sec:.3f}"
            )
        return SetParametersResult(successful=True)

    def _on_joint_states(self, msg: JointState):
        if not msg.name or not msg.position:
            return
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self._latest_joint_positions[name] = float(msg.position[i])
        self._have_joint_state = True

    def _on_target_twist(self, msg: TargetTwistStates):
        legacy_reset_enable = bool(msg.reset_enable)
        reset_robot_enable = bool(getattr(msg, "reset_robot_enable", legacy_reset_enable))
        reset_scene_enable = bool(getattr(msg, "reset_scene_enable", legacy_reset_enable))
        reset_enable = reset_robot_enable or reset_scene_enable

        reset_robot_edge = reset_robot_enable and not self._last_reset_robot_enable
        reset_scene_edge = reset_scene_enable and not self._last_reset_scene_enable
        if reset_robot_edge or reset_scene_edge:
            self._start_reset_if_idle(reset_robot_enable, reset_scene_enable)

        self._last_reset_enable = reset_enable
        self._last_reset_robot_enable = reset_robot_enable
        self._last_reset_scene_enable = reset_scene_enable

    def _start_reset_if_idle(self, reset_robot: bool, reset_scene: bool):
        now = time.monotonic()
        if reset_robot and now - self._last_reset_start_time < self.min_reset_interval_sec:
            self.get_logger().warn(
                f"Reset requested too soon after previous reset; ignoring "
                f"(min_reset_interval_sec={self.min_reset_interval_sec:.2f})."
            )
            return

        with self._reset_lock:
            if self._reset_in_progress:
                self.get_logger().warn("Reset requested while reset already in progress; ignoring new request.")
                return
            self._reset_in_progress = True
            self._cancel_reset_requested = False
            if reset_robot:
                self._last_reset_start_time = now

        thread = threading.Thread(target=self._run_reset_sequence, args=(reset_robot, reset_scene), daemon=True)
        thread.start()

    def _is_cancel_requested(self) -> bool:
        with self._reset_lock:
            return self._cancel_reset_requested

    def _run_reset_sequence(self, reset_robot: bool = True, reset_scene: bool = True):
        robot_reset_started = False
        try:
            if not reset_robot and not reset_scene:
                self.get_logger().warn("Reset requested with no reset targets; ignoring.")
                return

            if reset_scene and not reset_robot:
                self.get_logger().warn("Scene-object reset requested: restoring task-object poses only.")
                self._reset_cube_pose_best_effort()
                self._reset_scene_objects_best_effort()
                return

            robot_reset_started = True
            mode = "robot + scene" if reset_scene else "robot only"
            self.get_logger().warn(f"Robot reset requested: running {mode} reset sequence.")
            stopped_ok = self._call_trigger(self._stop_client, self.stop_servo_service)
            if stopped_ok:
                self.get_logger().info("Servo stop acknowledged.")
            self._call_empty(self._reset_status_client, self.reset_servo_status_service)

            self._hold_gripper_open(self.gripper_hold_open_sec)
            if self._is_cancel_requested():
                self.get_logger().warn("Reset cancelled before kick phase completed.")
                return

            self._run_constant_velocity(self.kick_velocities, self.kick_duration_sec)
            if self._is_cancel_requested():
                self.get_logger().warn("Reset cancelled during kick phase.")
                return

            if reset_scene:
                # Reset scene objects early so the change is visible immediately,
                # then reset again at the end as final cleanup after homing.
                self._reset_scene_objects_best_effort()
                if self._is_cancel_requested():
                    self.get_logger().warn("Reset cancelled after early scene-object reset.")
                    return

            self._run_home_controller()
            if self._is_cancel_requested():
                self.get_logger().warn("Reset cancelled during home phase.")
                return

            self._publish_arm_cmd(np.zeros(6, dtype=float))
            self._hold_gripper_open(self.gripper_hold_open_sec)
            if self._is_cancel_requested():
                self.get_logger().warn("Reset cancelled before scene-object reset.")
                return
            if reset_scene:
                self._reset_cube_pose_best_effort()
                self._reset_scene_objects_best_effort()

            if self.settle_sec > 0.0 and (not self._is_cancel_requested()):
                time.sleep(self.settle_sec)

        except Exception as exc:
            self.get_logger().error(f"Reset sequence failed: {exc}")
        finally:
            if robot_reset_started:
                self._publish_arm_cmd(np.zeros(6, dtype=float))
                started_ok = self._start_servo_with_retries()
                if not started_ok:
                    self.get_logger().warn("Servo start did not acknowledge after reset; teleop may remain inactive.")
                self._call_empty(self._reset_status_client, self.reset_servo_status_service)
            if self._is_cancel_requested():
                self.get_logger().warn("Reset sequence cancelled; teleop re-enabled.")
            else:
                self.get_logger().info("Reset sequence completed.")
            with self._reset_lock:
                self._reset_in_progress = False
                self._cancel_reset_requested = False

    def _publish_arm_cmd(self, values: np.ndarray):
        msg = Float64MultiArray()
        msg.data = [float(v) for v in values]
        self._arm_pub.publish(msg)

    def _publish_gripper(self, pos: float):
        msg = Float64MultiArray()
        if self.gripper_command_mode == "velocity":
            base = float(pos)
        else:
            base = float(pos)
        msg.data = [base] * self.gripper_joint_count
        if self.gripper_joint_count >= 2 and self.gripper_second_joint_negate:
            msg.data[1] = -base
        self._gripper_pub.publish(msg)

    def _hold_gripper_open(self, duration: float):
        if self.gripper_command_mode == "velocity":
            open_cmd = abs(self.gripper_open_velocity)
            if duration <= 0.0:
                self._publish_gripper(open_cmd)
                self._publish_gripper(0.0)
                return
            step = 1.0 / self.control_rate_hz
            end = time.monotonic() + duration
            while time.monotonic() < end:
                if self._is_cancel_requested():
                    break
                self._publish_gripper(open_cmd)
                time.sleep(step)
            self._publish_gripper(0.0)
            return

        if self.gripper_command_mode in ("effort", "effort_position"):
            open_cmd = abs(self.gripper_open_effort)
            if duration <= 0.0:
                self._publish_gripper(open_cmd)
                self._publish_gripper(0.0)
                return
            step = 1.0 / self.control_rate_hz
            end = time.monotonic() + duration
            while time.monotonic() < end:
                if self._is_cancel_requested():
                    break
                self._publish_gripper(open_cmd)
                time.sleep(step)
            self._publish_gripper(0.0)
            return

        if duration <= 0.0:
            self._publish_gripper(self.gripper_open_pos)
            return
        step = 1.0 / self.control_rate_hz
        end = time.monotonic() + duration
        while time.monotonic() < end:
            if self._is_cancel_requested():
                break
            self._publish_gripper(self.gripper_open_pos)
            time.sleep(step)

    def _run_constant_velocity(self, vel_cmd: np.ndarray, duration: float):
        if duration <= 0.0:
            return
        self.get_logger().info(f"Reset kick phase: duration={duration:.2f}s, cmd={vel_cmd.tolist()}")
        step = 1.0 / self.control_rate_hz
        end = time.monotonic() + duration
        while time.monotonic() < end:
            if self._is_cancel_requested():
                break
            self._publish_arm_cmd(vel_cmd)
            time.sleep(step)
        self._publish_arm_cmd(np.zeros(6, dtype=float))

    def _run_home_controller(self):
        self.get_logger().info(f"Reset home phase: target={self.home_joint_positions.tolist()}")
        if not self._have_joint_state:
            self.get_logger().warn("No joint_states seen yet; home phase may time out.")

        step = 1.0 / self.control_rate_hz
        deadline = time.monotonic() + self.home_timeout_sec
        last_max_err = float("inf")

        while time.monotonic() < deadline:
            if self._is_cancel_requested():
                self._publish_arm_cmd(np.zeros(6, dtype=float))
                return
            current = self._read_home_joint_vector()
            if current is None:
                time.sleep(step)
                continue

            err = self.home_joint_positions - current
            max_err = float(np.max(np.abs(err)))
            last_max_err = max_err

            if max_err <= self.home_tolerance:
                self.get_logger().info(f"Reset home reached (max_err={max_err:.4f} rad).")
                self._publish_arm_cmd(np.zeros(6, dtype=float))
                return

            cmd = self.home_kp * err
            cmd = np.clip(cmd, -self.home_max_vel, self.home_max_vel)
            self._publish_arm_cmd(cmd)
            time.sleep(step)

        self._publish_arm_cmd(np.zeros(6, dtype=float))
        self.get_logger().warn(
            f"Reset home timed out after {self.home_timeout_sec:.1f}s (last_max_err={last_max_err:.4f} rad)."
        )

    def _read_home_joint_vector(self):
        vals = []
        for name in self.home_joint_names:
            if name not in self._latest_joint_positions:
                return None
            vals.append(self._latest_joint_positions[name])
        return np.array(vals, dtype=float)

    def _call_trigger(self, client, service_name: str):
        if not client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn(f"Service not available: {service_name}")
            return False

        future = client.call_async(Trigger.Request())
        if not self._wait_future(future, timeout_sec=3.0):
            self.get_logger().warn(f"Service timeout: {service_name}")
            return False

        try:
            resp = future.result()
        except Exception as exc:
            self.get_logger().warn(f"Service call failed ({service_name}): {exc}")
            return False

        if not resp.success:
            self.get_logger().warn(f"Service returned success=false ({service_name}): {resp.message}")
            return False

        return True

    def _call_empty(self, client, service_name: str):
        if not client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn(f"Service not available: {service_name}")
            return False

        future = client.call_async(Empty.Request())
        if not self._wait_future(future, timeout_sec=3.0):
            self.get_logger().warn(f"Service timeout: {service_name}")
            return False

        try:
            _ = future.result()
        except Exception as exc:
            self.get_logger().warn(f"Service call failed ({service_name}): {exc}")
            return False
        return True

    def _start_servo_with_retries(self):
        for i in range(self.start_servo_retries):
            if self._call_trigger(self._start_client, self.start_servo_service):
                self.get_logger().info(
                    f"Servo start acknowledged (attempt {i + 1}/{self.start_servo_retries})."
                )
                return True
            if i < (self.start_servo_retries - 1):
                time.sleep(self.start_servo_retry_sleep_sec)
        return False

    @staticmethod
    def _wait_future(future, timeout_sec: float):
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            time.sleep(0.02)
        return future.done()

    def _reset_cube_pose_best_effort(self):
        if not self.reset_cube_with_ign_service:
            return

        if self.cube_reset_pose_xyz.shape != (3,) or self.cube_reset_pose_xyzw.shape != (4,):
            self.get_logger().warn("Invalid cube reset pose params; skipping cube reset.")
            return

        req = (
            f'name: "{self.cube_model_name}" '
            f'position: {{x: {self.cube_reset_pose_xyz[0]} y: {self.cube_reset_pose_xyz[1]} z: {self.cube_reset_pose_xyz[2]}}} '
            f'orientation: {{x: {self.cube_reset_pose_xyzw[0]} y: {self.cube_reset_pose_xyzw[1]} z: {self.cube_reset_pose_xyzw[2]} w: {self.cube_reset_pose_xyzw[3]}}}'
        )
        cmd = [
            "ign",
            "service",
            "-s",
            f"/world/{self.world_name}/set_pose",
            "--reqtype",
            "ignition.msgs.Pose",
            "--reptype",
            "ignition.msgs.Boolean",
            "--timeout",
            "2000",
            "--req",
            req,
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=4.0, check=False)
        except Exception as exc:
            self.get_logger().warn(f"Cube reset command failed to run: {exc}")
            return

        if result.returncode != 0:
            stderr = result.stderr.strip() or "unknown error"
            self.get_logger().warn(f"Cube reset failed (rc={result.returncode}): {stderr}")
            return

        self.get_logger().info(f"Cube reset requested for '{self.cube_model_name}'.")

    def _reset_scene_objects_best_effort(self):
        if not self.reset_scene_objects_with_ign_service:
            return
        if not self.scene_reset_targets:
            self.get_logger().warn(
                "Scene-object reset enabled but no scene reset targets were resolved from the setup; skipping."
            )
            return

        success_by_name = {target.model_name: False for target in self.scene_reset_targets}
        for attempt in range(self.scene_reset_repeat_count):
            if attempt > 0 and self.scene_reset_repeat_interval_sec > 0.0:
                time.sleep(self.scene_reset_repeat_interval_sec)
            for target in self.scene_reset_targets:
                pose_xyz = target.pose_xyz.copy()
                pose_xyz[2] += self.scene_reset_lift_z
                if self._set_model_pose_best_effort(target.model_name, pose_xyz, target.pose_xyzw):
                    success_by_name[target.model_name] = True

        success_count = sum(1 for ok in success_by_name.values() if ok)
        failed = [name for name, ok in success_by_name.items() if not ok]
        self.get_logger().info(
            f"Scene-object reset requested for {success_count}/{len(self.scene_reset_targets)} models "
            f"(z lift={self.scene_reset_lift_z:.4f} m, repeats={self.scene_reset_repeat_count})."
        )
        if failed:
            self.get_logger().warn(f"Scene-object reset failed for: {failed}")

    def _set_model_pose_best_effort(self, model_name: str, pose_xyz: np.ndarray, pose_xyzw: np.ndarray) -> bool:
        if pose_xyz.shape != (3,) or pose_xyzw.shape != (4,):
            self.get_logger().warn(f"Invalid reset pose for '{model_name}'; skipping.")
            return False

        req = (
            f'name: "{model_name}" '
            f'position: {{x: {pose_xyz[0]} y: {pose_xyz[1]} z: {pose_xyz[2]}}} '
            f'orientation: {{x: {pose_xyzw[0]} y: {pose_xyzw[1]} z: {pose_xyzw[2]} w: {pose_xyzw[3]}}}'
        )
        cmd = [
            "ign",
            "service",
            "-s",
            f"/world/{self.world_name}/set_pose",
            "--reqtype",
            "ignition.msgs.Pose",
            "--reptype",
            "ignition.msgs.Boolean",
            "--timeout",
            "2000",
            "--req",
            req,
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=4.0, check=False)
        except Exception as exc:
            self.get_logger().warn(f"Reset command failed to run for '{model_name}': {exc}")
            return False

        if result.returncode != 0:
            stderr = result.stderr.strip() or "unknown error"
            self.get_logger().warn(f"Reset failed for '{model_name}' (rc={result.returncode}): {stderr}")
            return False

        return True


def main(args=None):
    rclpy.init(args=args)
    node = TargetTwistResetManager()
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
