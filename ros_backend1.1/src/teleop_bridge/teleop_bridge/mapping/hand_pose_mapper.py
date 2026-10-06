import math
import time

import numpy as np
import rclpy
from rcl_interfaces.msg import SetParametersResult
from geometry_msgs.msg import PoseStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from teleop_bridge_msgs.msg import ReceivedPoseStates, TargetTwistStates
from tf2_ros import Buffer, TransformListener


class ReceivedPoseToTargetTwist(Node):
    def __init__(self):
        super().__init__("hand_pose_mapper")

        self.declare_parameter("input_topic", "/received_pose_states")
        self.declare_parameter("output_topic", "/target_twist_states")
        self.declare_parameter("publish_ee_pose_topics", True)
        self.declare_parameter("target_ee_pose_topic", "/teleop/target_ee_pose")
        self.declare_parameter("actual_ee_pose_topic", "/teleop/actual_ee_pose")
        self.declare_parameter("target_frame", "base_link")
        self.declare_parameter("ee_frame", "tool0")
        self.declare_parameter("publish_rate_hz", 60.0)
        self.declare_parameter("stale_timeout_sec", 0.25)
        self.declare_parameter("reset_hold_sec", 8.0)
        self.declare_parameter("position_mapping_mode", "world_delta")
        self.declare_parameter("attachment_use_absolute_position", True)
        self.declare_parameter("attachment_enable_rotation", False)
        self.declare_parameter("attachment_use_absolute_rotation", True)
        self.declare_parameter("attachment_base_xyz", [0.0, 0.0, 0.0])
        self.declare_parameter("attachment_base_xyzw", [0.0, 0.0, 0.0, 1.0])
        self.declare_parameter("attachment_tool_position_offset_xyz", [0.0, 0.0, 0.0])
        self.declare_parameter("attachment_tool_rotation_offset_xyzw", [0.0, 0.0, 0.0, 1.0])
        self.declare_parameter("attachment_kp_angular", 1.4)
        self.declare_parameter("attachment_max_angular_speed", 0.65)
        self.declare_parameter("attachment_angular_deadband", 0.03)

        self.declare_parameter("kp_linear", 2.0)
        self.declare_parameter("max_linear_speed", 0.30)
        self.declare_parameter("linear_deadband", 0.005)

        self.declare_parameter("kp_angular", 4.0)
        self.declare_parameter("max_angular_speed", 1.50)
        self.declare_parameter("angular_deadband", 0.02)
        self.declare_parameter("start_in_gamepad_mode", False)
        self.declare_parameter("control_mode", "hand_pose")
        self.declare_parameter("allow_unity_control_mode_switch", False)
        self.declare_parameter("gamepad_deadband", 0.15)
        self.declare_parameter("gamepad_linear_speed_xyz", [0.20, 0.20, 0.20])
        self.declare_parameter("gamepad_linear_sign_xyz", [1.0, -1.0, 1.0])
        self.declare_parameter("gamepad_angular_speed_xyz", [0.0, 0.75, 0.75])
        self.declare_parameter("gamepad_angular_sign_xyz", [1.0, 1.0, 1.0])

        # Calibrated default for this project setup:
        # Unity/controller: x=forward/back, y=up/down, z=left/right
        # Robot base_link:   x=forward/back, y=left/right, z=up/down
        self.declare_parameter("map_axes", ["x", "z", "y"])
        self.declare_parameter("map_signs", [1.0, -1.0, 1.0])
        self.declare_parameter("rot_map_axes", ["x", "z", "y"])
        self.declare_parameter("rot_map_signs", [1.0, -1.0, 1.0])
        self.declare_parameter("rot_scale_xyz", [1.0, 1.0, 1.0])
        self.declare_parameter("debug_rotation_mapping", False)
        self.declare_parameter("debug_rotation_log_period_sec", 0.25)
        self.declare_parameter("scale_xyz", [1.0, 1.0, 1.0])
        self.declare_parameter("offset_xyz", [0.0, 0.0, 0.0])
        self.declare_parameter("min_xyz", [0.15, -0.50, 0.05])
        self.declare_parameter("max_xyz", [0.75, 0.50, 0.70])

        input_topic = str(self.get_parameter("input_topic").value)
        output_topic = str(self.get_parameter("output_topic").value)
        self.publish_ee_pose_topics = bool(self.get_parameter("publish_ee_pose_topics").value)
        target_ee_pose_topic = str(self.get_parameter("target_ee_pose_topic").value)
        actual_ee_pose_topic = str(self.get_parameter("actual_ee_pose_topic").value)
        self.target_frame = str(self.get_parameter("target_frame").value)
        self.ee_frame = str(self.get_parameter("ee_frame").value)
        publish_rate_hz = max(1.0, float(self.get_parameter("publish_rate_hz").value))
        self.stale_timeout_sec = max(0.05, float(self.get_parameter("stale_timeout_sec").value))
        self.reset_hold_sec = max(0.05, float(self.get_parameter("reset_hold_sec").value))
        try:
            self.position_mapping_mode = self._parse_position_mapping_mode(
                self.get_parameter("position_mapping_mode").value
            )
        except ValueError as exc:
            self.get_logger().warn(f"{exc}; falling back to world_delta.")
            self.position_mapping_mode = "world_delta"

        self.kp_linear = max(0.0, float(self.get_parameter("kp_linear").value))
        self.max_linear_speed = max(0.0, float(self.get_parameter("max_linear_speed").value))
        self.linear_deadband = max(0.0, float(self.get_parameter("linear_deadband").value))

        self.kp_angular = max(0.0, float(self.get_parameter("kp_angular").value))
        self.max_angular_speed = max(0.0, float(self.get_parameter("max_angular_speed").value))
        self.angular_deadband = max(0.0, float(self.get_parameter("angular_deadband").value))
        self.attachment_kp_angular = max(0.0, float(self.get_parameter("attachment_kp_angular").value))
        self.attachment_max_angular_speed = max(
            0.0, float(self.get_parameter("attachment_max_angular_speed").value)
        )
        self.attachment_angular_deadband = max(
            0.0, float(self.get_parameter("attachment_angular_deadband").value)
        )
        self.start_in_gamepad_mode = bool(self.get_parameter("start_in_gamepad_mode").value)
        try:
            self.control_mode = self._parse_control_mode(self.get_parameter("control_mode").value)
        except ValueError as exc:
            self.get_logger().warn(f"{exc}; falling back to start_in_gamepad_mode.")
            self.control_mode = "gamepad" if self.start_in_gamepad_mode else "hand_pose"
        self.allow_unity_control_mode_switch = bool(
            self.get_parameter("allow_unity_control_mode_switch").value
        )
        self.gamepad_deadband = max(0.0, float(self.get_parameter("gamepad_deadband").value))
        self.gamepad_linear_speed_xyz = np.array(self.get_parameter("gamepad_linear_speed_xyz").value, dtype=float)
        if self.gamepad_linear_speed_xyz.shape != (3,):
            self.get_logger().warn("Invalid gamepad_linear_speed_xyz; falling back to [0.20,0.20,0.20].")
            self.gamepad_linear_speed_xyz = np.array([0.20, 0.20, 0.20], dtype=float)
        self.gamepad_linear_sign_xyz = np.array(self.get_parameter("gamepad_linear_sign_xyz").value, dtype=float)
        if self.gamepad_linear_sign_xyz.shape != (3,):
            self.get_logger().warn("Invalid gamepad_linear_sign_xyz; falling back to [1,-1,1].")
            self.gamepad_linear_sign_xyz = np.array([1.0, -1.0, 1.0], dtype=float)
        self.gamepad_angular_speed_xyz = np.array(self.get_parameter("gamepad_angular_speed_xyz").value, dtype=float)
        if self.gamepad_angular_speed_xyz.shape != (3,):
            self.get_logger().warn("Invalid gamepad_angular_speed_xyz; falling back to [0,0.75,0.75].")
            self.gamepad_angular_speed_xyz = np.array([0.0, 0.75, 0.75], dtype=float)
        self.gamepad_angular_sign_xyz = np.array(self.get_parameter("gamepad_angular_sign_xyz").value, dtype=float)
        if self.gamepad_angular_sign_xyz.shape != (3,):
            self.get_logger().warn("Invalid gamepad_angular_sign_xyz; falling back to [1,1,1].")
            self.gamepad_angular_sign_xyz = np.array([1.0, 1.0, 1.0], dtype=float)

        self.map_axes = [str(v).lower() for v in self.get_parameter("map_axes").value]
        if len(self.map_axes) != 3 or any(axis not in ("x", "y", "z") for axis in self.map_axes):
            self.get_logger().warn("Invalid map_axes; falling back to ['x','z','y'].")
            self.map_axes = ["x", "z", "y"]

        self.map_signs = np.array(self.get_parameter("map_signs").value, dtype=float)
        if self.map_signs.shape != (3,):
            self.get_logger().warn("Invalid map_signs; falling back to [1,-1,1].")
            self.map_signs = np.array([1.0, -1.0, 1.0], dtype=float)

        self.rot_map_axes = [str(v).lower() for v in self.get_parameter("rot_map_axes").value]
        if len(self.rot_map_axes) != 3 or any(axis not in ("x", "y", "z") for axis in self.rot_map_axes):
            self.get_logger().warn("Invalid rot_map_axes; falling back to ['x','z','y'].")
            self.rot_map_axes = ["x", "z", "y"]

        self.rot_map_signs = np.array(self.get_parameter("rot_map_signs").value, dtype=float)
        if self.rot_map_signs.shape != (3,):
            self.get_logger().warn("Invalid rot_map_signs; falling back to [1,-1,1].")
            self.rot_map_signs = np.array([1.0, -1.0, 1.0], dtype=float)

        self.rot_scale_xyz = np.array(self.get_parameter("rot_scale_xyz").value, dtype=float)
        if self.rot_scale_xyz.shape != (3,):
            self.get_logger().warn("Invalid rot_scale_xyz; falling back to [1,1,1].")
            self.rot_scale_xyz = np.array([1.0, 1.0, 1.0], dtype=float)
        self.debug_rotation_mapping = bool(self.get_parameter("debug_rotation_mapping").value)
        self.debug_rotation_log_period_sec = max(
            0.05, float(self.get_parameter("debug_rotation_log_period_sec").value)
        )

        self.scale_xyz = np.array(self.get_parameter("scale_xyz").value, dtype=float)
        if self.scale_xyz.shape != (3,):
            self.get_logger().warn("Invalid scale_xyz; falling back to [1,1,1].")
            self.scale_xyz = np.array([1.0, 1.0, 1.0], dtype=float)

        self.offset_xyz = np.array(self.get_parameter("offset_xyz").value, dtype=float)
        if self.offset_xyz.shape != (3,):
            self.get_logger().warn("Invalid offset_xyz; falling back to [0,0,0].")
            self.offset_xyz = np.array([0.0, 0.0, 0.0], dtype=float)

        self.min_xyz = np.array(self.get_parameter("min_xyz").value, dtype=float)
        if self.min_xyz.shape != (3,):
            self.get_logger().warn("Invalid min_xyz; falling back to [0.15,-0.50,0.05].")
            self.min_xyz = np.array([0.15, -0.50, 0.05], dtype=float)

        self.max_xyz = np.array(self.get_parameter("max_xyz").value, dtype=float)
        if self.max_xyz.shape != (3,):
            self.get_logger().warn("Invalid max_xyz; falling back to [0.75,0.50,0.70].")
            self.max_xyz = np.array([0.75, 0.50, 0.70], dtype=float)
        if np.any(self.min_xyz > self.max_xyz):
            self.get_logger().warn("Invalid workspace bounds (min_xyz > max_xyz); using defaults.")
            self.min_xyz = np.array([0.15, -0.50, 0.05], dtype=float)
            self.max_xyz = np.array([0.75, 0.50, 0.70], dtype=float)

        self.attachment_use_absolute_position = bool(
            self.get_parameter("attachment_use_absolute_position").value
        )
        self.attachment_enable_rotation = bool(
            self.get_parameter("attachment_enable_rotation").value
        )
        self.attachment_use_absolute_rotation = bool(
            self.get_parameter("attachment_use_absolute_rotation").value
        )
        self.attachment_base_xyz = np.array(
            self.get_parameter("attachment_base_xyz").value, dtype=float
        )
        if self.attachment_base_xyz.shape != (3,):
            self.get_logger().warn("Invalid attachment_base_xyz; falling back to [0,0,0].")
            self.attachment_base_xyz = np.zeros(3, dtype=float)
        self.attachment_base_xyzw = self._normalize_quat(
            np.array(self.get_parameter("attachment_base_xyzw").value, dtype=float)
        )
        if self.attachment_base_xyzw.shape != (4,):
            self.get_logger().warn("Invalid attachment_base_xyzw; falling back to identity.")
            self.attachment_base_xyzw = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        self.attachment_tool_position_offset_xyz = np.array(
            self.get_parameter("attachment_tool_position_offset_xyz").value, dtype=float
        )
        if self.attachment_tool_position_offset_xyz.shape != (3,):
            self.get_logger().warn(
                "Invalid attachment_tool_position_offset_xyz; falling back to [0,0,0]."
            )
            self.attachment_tool_position_offset_xyz = np.zeros(3, dtype=float)
        self.attachment_tool_rotation_offset_xyzw = self._normalize_quat(
            np.array(self.get_parameter("attachment_tool_rotation_offset_xyzw").value, dtype=float)
        )
        if self.attachment_tool_rotation_offset_xyzw.shape != (4,):
            self.get_logger().warn("Invalid attachment_tool_rotation_offset_xyzw; falling back to identity.")
            self.attachment_tool_rotation_offset_xyzw = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)

        self.sub = self.create_subscription(ReceivedPoseStates, input_topic, self._on_pose_states, 20)
        self.pub = self.create_publisher(TargetTwistStates, output_topic, 20)
        self.target_pose_pub = self.create_publisher(PoseStamped, target_ee_pose_topic, 20)
        self.actual_pose_pub = self.create_publisher(PoseStamped, actual_ee_pose_topic, 20)
        self.create_timer(1.0 / publish_rate_hz, self._publish_loop)
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._last_rx_time = 0.0
        self._tracked = False
        self._rotate_enable = False
        self._gripper_cmd = 0
        self._reset_button = False
        self._reset_button_last = False
        self._reset_robot_button_last = False
        self._reset_scene_button_last = False
        self._reset_latched = False
        self._reset_latch_until = 0.0
        self._reset_scene_latch_until = 0.0
        self._recenter_button_last = False
        self._recenter_hold_active = False
        self._teleop_enable = False
        self._teleop_enable_last = False
        self._attachment_mode = False
        self._attachment_mode_last = False
        self._mode_switch_last = False
        self._gamepad_mode = self.control_mode == "gamepad"
        self._left_stick = np.zeros(2, dtype=float)
        self._right_stick = np.zeros(2, dtype=float)
        self._left_grip_value = 0.0
        self._left_trigger_value = 0.0

        self._target_pos = np.zeros(3, dtype=float)
        self._latest_hand_pos = np.zeros(3, dtype=float)
        self._latest_hand_world_pos = np.zeros(3, dtype=float)
        self._position_hand_world_ref = np.zeros(3, dtype=float)
        self._workspace_pose_valid = False
        self._workspace_pos = np.zeros(3, dtype=float)
        self._workspace_rot = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        self._position_session_active = False
        self._position_recenter_pending = False
        self._position_recenter_hand_ref_valid = False
        self._position_recenter_hand_ref = np.zeros(3, dtype=float)
        self._position_recenter_hand_world_ref = np.zeros(3, dtype=float)
        self._position_hand_ref = np.zeros(3, dtype=float)
        self._position_ee_ref = np.zeros(3, dtype=float)
        self._hand_rot = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        self._hand_world_rot = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        self._have_target = False

        self._rotate_session_active = False
        self._rotate_hand_ref = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        self._rotate_ee_ref = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        self._attachment_session_active = False
        self._attachment_hand_ref = np.zeros(3, dtype=float)
        self._attachment_hand_world_ref = np.zeros(3, dtype=float)
        self._attachment_ee_ref = np.zeros(3, dtype=float)
        self._attachment_hand_rot_ref = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        self._attachment_ee_rot_ref = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        self._attachment_runtime_offset_valid = False
        self._attachment_position_offset_unity_workspace = np.zeros(3, dtype=float)
        self._attachment_rotation_offset_unity = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        self._attachment_adjustment_mode = False
        self._last_rotation_input_rotvec = np.zeros(3, dtype=float)
        self._last_rotation_mapped_rotvec = np.zeros(3, dtype=float)
        self._last_rotation_debug_log_time = 0.0

        self._rx_count = 0
        self._rx_window_start = time.monotonic()
        self._last_log_time = time.monotonic()
        self._last_tf_warn_time = 0.0

        self.get_logger().info(
            f"Mapping {input_topic} -> {output_topic}, target_frame={self.target_frame}, "
            f"ee_frame={self.ee_frame}, stale_timeout_sec={self.stale_timeout_sec:.3f}"
        )
        self.get_logger().info(
            f"linear: kp={self.kp_linear:.3f}, max={self.max_linear_speed:.3f}, deadband={self.linear_deadband:.3f} | "
            f"angular: kp={self.kp_angular:.3f}, max={self.max_angular_speed:.3f}, deadband={self.angular_deadband:.3f}"
        )
        self.get_logger().info(
            f"control_mode_default={'gamepad' if self._gamepad_mode else 'hand_pose'}, "
            f"gamepad_deadband={self.gamepad_deadband:.3f}, "
            f"gamepad_linear_speed_xyz={self.gamepad_linear_speed_xyz.tolist()}, "
            f"gamepad_linear_sign_xyz={self.gamepad_linear_sign_xyz.tolist()}, "
            f"gamepad_angular_speed_xyz={self.gamepad_angular_speed_xyz.tolist()}, "
            f"gamepad_angular_sign_xyz={self.gamepad_angular_sign_xyz.tolist()}"
        )
        self.get_logger().info(
            f"pos_map_axes={self.map_axes}, pos_map_signs={self.map_signs.tolist()}, "
            f"position_mapping_mode={self.position_mapping_mode}, "
            f"rot_map_axes={self.rot_map_axes}, rot_map_signs={self.rot_map_signs.tolist()}, rot_scale_xyz={self.rot_scale_xyz.tolist()}, "
            f"debug_rotation_mapping={self.debug_rotation_mapping}, debug_rotation_log_period_sec={self.debug_rotation_log_period_sec:.3f}, "
            f"scale_xyz={self.scale_xyz.tolist()}, offset_xyz={self.offset_xyz.tolist()}, "
            f"min_xyz={self.min_xyz.tolist()}, max_xyz={self.max_xyz.tolist()}"
        )
        self.get_logger().info(
            f"attachment absolute: position={self.attachment_use_absolute_position}, "
            f"enable_rotation={self.attachment_enable_rotation}, "
            f"rotation={self.attachment_use_absolute_rotation}, "
            f"attachment_kp_angular={self.attachment_kp_angular:.3f}, "
            f"attachment_max_angular_speed={self.attachment_max_angular_speed:.3f}, "
            f"attachment_angular_deadband={self.attachment_angular_deadband:.4f}, "
            f"base_xyz={self.attachment_base_xyz.tolist()}, "
            f"base_xyzw={self.attachment_base_xyzw.tolist()}, "
            f"tool_pos_offset_xyz={self.attachment_tool_position_offset_xyz.tolist()}, "
            f"tool_rot_offset_xyzw={self.attachment_tool_rotation_offset_xyzw.tolist()}"
        )
        self.add_on_set_parameters_callback(self._on_parameter_change)

    @staticmethod
    def _parse_position_mapping_mode(value) -> str:
        mode = str(value).strip().lower()
        aliases = {
            "delta": "world_delta",
            "mapping": "unity_world_delta",
            "world_mapping": "unity_world_delta",
            "world": "world_delta",
            "old": "absolute",
            "legacy": "absolute",
        }
        mode = aliases.get(mode, mode)
        if mode not in ("absolute", "world_delta", "unity_world_delta", "unity_world_absolute"):
            raise ValueError(
                "position_mapping_mode must be 'world_delta', 'absolute', "
                "'unity_world_delta', or 'unity_world_absolute'"
            )
        return mode

    @classmethod
    def _parse_optional_position_mapping_mode(cls, value):
        if value is None:
            return None
        mode = str(value).strip()
        if not mode:
            return None
        try:
            return cls._parse_position_mapping_mode(mode)
        except ValueError:
            return None

    @staticmethod
    def _parse_control_mode(value) -> str:
        mode = str(value).strip().lower()
        aliases = {
            "hand": "hand_pose",
            "pose": "hand_pose",
            "controller": "hand_pose",
            "controller_pose": "hand_pose",
            "game": "gamepad",
            "thumbstick": "gamepad",
        }
        mode = aliases.get(mode, mode)
        if mode not in ("hand_pose", "gamepad"):
            raise ValueError("control_mode must be 'hand_pose' or 'gamepad'")
        return mode

    @classmethod
    def _parse_optional_control_mode(cls, value):
        if value is None:
            return None
        mode = str(value).strip()
        if not mode:
            return None
        try:
            return cls._parse_control_mode(mode)
        except ValueError:
            return None

    @staticmethod
    def _parse_map_axes(value):
        axes = [str(v).lower() for v in value]
        if len(axes) != 3 or any(axis not in ("x", "y", "z") for axis in axes):
            raise ValueError("map_axes must be a 3-item list using only x/y/z.")
        return axes

    @staticmethod
    def _parse_vec3(value, field_name: str) -> np.ndarray:
        vec = np.array(value, dtype=float)
        if vec.shape != (3,):
            raise ValueError(f"{field_name} must be a 3-item numeric list.")
        return vec

    def _parse_quat(self, value, field_name: str) -> np.ndarray:
        quat = np.array(value, dtype=float)
        if quat.shape != (4,):
            raise ValueError(f"{field_name} must be a 4-item numeric list [x,y,z,w].")
        return self._normalize_quat(quat)

    def _on_parameter_change(self, params):
        new_target_frame = self.target_frame
        new_ee_frame = self.ee_frame
        new_stale_timeout_sec = self.stale_timeout_sec
        new_reset_hold_sec = self.reset_hold_sec
        new_position_mapping_mode = self.position_mapping_mode
        new_kp_linear = self.kp_linear
        new_max_linear_speed = self.max_linear_speed
        new_linear_deadband = self.linear_deadband
        new_kp_angular = self.kp_angular
        new_max_angular_speed = self.max_angular_speed
        new_angular_deadband = self.angular_deadband
        new_attachment_kp_angular = self.attachment_kp_angular
        new_attachment_max_angular_speed = self.attachment_max_angular_speed
        new_attachment_angular_deadband = self.attachment_angular_deadband
        new_start_in_gamepad_mode = self.start_in_gamepad_mode
        new_control_mode = self.control_mode
        new_allow_unity_control_mode_switch = self.allow_unity_control_mode_switch
        new_gamepad_deadband = self.gamepad_deadband
        new_gamepad_linear_speed_xyz = self.gamepad_linear_speed_xyz
        new_gamepad_linear_sign_xyz = self.gamepad_linear_sign_xyz
        new_gamepad_angular_speed_xyz = self.gamepad_angular_speed_xyz
        new_gamepad_angular_sign_xyz = self.gamepad_angular_sign_xyz
        new_map_axes = self.map_axes
        new_map_signs = self.map_signs
        new_rot_map_axes = self.rot_map_axes
        new_rot_map_signs = self.rot_map_signs
        new_rot_scale_xyz = self.rot_scale_xyz
        new_debug_rotation_mapping = self.debug_rotation_mapping
        new_debug_rotation_log_period_sec = self.debug_rotation_log_period_sec
        new_scale_xyz = self.scale_xyz
        new_offset_xyz = self.offset_xyz
        new_min_xyz = self.min_xyz
        new_max_xyz = self.max_xyz
        new_attachment_use_absolute_position = self.attachment_use_absolute_position
        new_attachment_enable_rotation = self.attachment_enable_rotation
        new_attachment_use_absolute_rotation = self.attachment_use_absolute_rotation
        new_attachment_base_xyz = self.attachment_base_xyz
        new_attachment_base_xyzw = self.attachment_base_xyzw
        new_attachment_tool_position_offset_xyz = self.attachment_tool_position_offset_xyz
        new_attachment_tool_rotation_offset_xyzw = self.attachment_tool_rotation_offset_xyzw
        touched = []

        try:
            for param in params:
                if param.name == "map_axes":
                    new_map_axes = self._parse_map_axes(param.value)
                    touched.append("map_axes")
                elif param.name == "map_signs":
                    new_map_signs = self._parse_vec3(param.value, "map_signs")
                    touched.append("map_signs")
                elif param.name == "position_mapping_mode":
                    new_position_mapping_mode = self._parse_position_mapping_mode(param.value)
                    touched.append("position_mapping_mode")
                elif param.name == "rot_map_axes":
                    new_rot_map_axes = self._parse_map_axes(param.value)
                    touched.append("rot_map_axes")
                elif param.name == "rot_map_signs":
                    new_rot_map_signs = self._parse_vec3(param.value, "rot_map_signs")
                    touched.append("rot_map_signs")
                elif param.name == "rot_scale_xyz":
                    new_rot_scale_xyz = self._parse_vec3(param.value, "rot_scale_xyz")
                    touched.append("rot_scale_xyz")
                elif param.name == "debug_rotation_mapping":
                    new_debug_rotation_mapping = bool(param.value)
                    touched.append("debug_rotation_mapping")
                elif param.name == "debug_rotation_log_period_sec":
                    new_debug_rotation_log_period_sec = max(0.05, float(param.value))
                    touched.append("debug_rotation_log_period_sec")
                elif param.name == "scale_xyz":
                    new_scale_xyz = self._parse_vec3(param.value, "scale_xyz")
                    touched.append("scale_xyz")
                elif param.name == "offset_xyz":
                    new_offset_xyz = self._parse_vec3(param.value, "offset_xyz")
                    touched.append("offset_xyz")
                elif param.name == "min_xyz":
                    new_min_xyz = self._parse_vec3(param.value, "min_xyz")
                    touched.append("min_xyz")
                elif param.name == "max_xyz":
                    new_max_xyz = self._parse_vec3(param.value, "max_xyz")
                    touched.append("max_xyz")
                elif param.name == "attachment_use_absolute_position":
                    new_attachment_use_absolute_position = bool(param.value)
                    touched.append("attachment_use_absolute_position")
                elif param.name == "attachment_enable_rotation":
                    new_attachment_enable_rotation = bool(param.value)
                    touched.append("attachment_enable_rotation")
                elif param.name == "attachment_use_absolute_rotation":
                    new_attachment_use_absolute_rotation = bool(param.value)
                    touched.append("attachment_use_absolute_rotation")
                elif param.name == "attachment_base_xyz":
                    new_attachment_base_xyz = self._parse_vec3(param.value, "attachment_base_xyz")
                    touched.append("attachment_base_xyz")
                elif param.name == "attachment_base_xyzw":
                    new_attachment_base_xyzw = self._parse_quat(param.value, "attachment_base_xyzw")
                    touched.append("attachment_base_xyzw")
                elif param.name == "attachment_tool_position_offset_xyz":
                    new_attachment_tool_position_offset_xyz = self._parse_vec3(
                        param.value, "attachment_tool_position_offset_xyz"
                    )
                    touched.append("attachment_tool_position_offset_xyz")
                elif param.name == "attachment_tool_rotation_offset_xyzw":
                    new_attachment_tool_rotation_offset_xyzw = self._parse_quat(
                        param.value, "attachment_tool_rotation_offset_xyzw"
                    )
                    touched.append("attachment_tool_rotation_offset_xyzw")
                elif param.name == "target_frame":
                    new_target_frame = str(param.value)
                    touched.append("target_frame")
                elif param.name == "ee_frame":
                    new_ee_frame = str(param.value)
                    touched.append("ee_frame")
                elif param.name == "stale_timeout_sec":
                    new_stale_timeout_sec = max(0.05, float(param.value))
                    touched.append("stale_timeout_sec")
                elif param.name == "reset_hold_sec":
                    new_reset_hold_sec = max(0.05, float(param.value))
                    touched.append("reset_hold_sec")
                elif param.name == "kp_linear":
                    new_kp_linear = max(0.0, float(param.value))
                    touched.append("kp_linear")
                elif param.name == "max_linear_speed":
                    new_max_linear_speed = max(0.0, float(param.value))
                    touched.append("max_linear_speed")
                elif param.name == "linear_deadband":
                    new_linear_deadband = max(0.0, float(param.value))
                    touched.append("linear_deadband")
                elif param.name == "kp_angular":
                    new_kp_angular = max(0.0, float(param.value))
                    touched.append("kp_angular")
                elif param.name == "max_angular_speed":
                    new_max_angular_speed = max(0.0, float(param.value))
                    touched.append("max_angular_speed")
                elif param.name == "angular_deadband":
                    new_angular_deadband = max(0.0, float(param.value))
                    touched.append("angular_deadband")
                elif param.name == "attachment_kp_angular":
                    new_attachment_kp_angular = max(0.0, float(param.value))
                    touched.append("attachment_kp_angular")
                elif param.name == "attachment_max_angular_speed":
                    new_attachment_max_angular_speed = max(0.0, float(param.value))
                    touched.append("attachment_max_angular_speed")
                elif param.name == "attachment_angular_deadband":
                    new_attachment_angular_deadband = max(0.0, float(param.value))
                    touched.append("attachment_angular_deadband")
                elif param.name == "start_in_gamepad_mode":
                    new_start_in_gamepad_mode = bool(param.value)
                    new_control_mode = "gamepad" if new_start_in_gamepad_mode else "hand_pose"
                    touched.append("start_in_gamepad_mode")
                elif param.name == "control_mode":
                    new_control_mode = self._parse_control_mode(param.value)
                    new_start_in_gamepad_mode = new_control_mode == "gamepad"
                    touched.append("control_mode")
                elif param.name == "allow_unity_control_mode_switch":
                    new_allow_unity_control_mode_switch = bool(param.value)
                    touched.append("allow_unity_control_mode_switch")
                elif param.name == "gamepad_deadband":
                    new_gamepad_deadband = max(0.0, float(param.value))
                    touched.append("gamepad_deadband")
                elif param.name == "gamepad_linear_speed_xyz":
                    new_gamepad_linear_speed_xyz = self._parse_vec3(param.value, "gamepad_linear_speed_xyz")
                    touched.append("gamepad_linear_speed_xyz")
                elif param.name == "gamepad_linear_sign_xyz":
                    new_gamepad_linear_sign_xyz = self._parse_vec3(param.value, "gamepad_linear_sign_xyz")
                    touched.append("gamepad_linear_sign_xyz")
                elif param.name == "gamepad_angular_speed_xyz":
                    new_gamepad_angular_speed_xyz = self._parse_vec3(param.value, "gamepad_angular_speed_xyz")
                    touched.append("gamepad_angular_speed_xyz")
                elif param.name == "gamepad_angular_sign_xyz":
                    new_gamepad_angular_sign_xyz = self._parse_vec3(param.value, "gamepad_angular_sign_xyz")
                    touched.append("gamepad_angular_sign_xyz")
        except ValueError as exc:
            return SetParametersResult(successful=False, reason=str(exc))

        if np.any(new_min_xyz > new_max_xyz):
            return SetParametersResult(
                successful=False, reason="Invalid workspace bounds: min_xyz must be <= max_xyz."
            )

        self.target_frame = new_target_frame
        self.ee_frame = new_ee_frame
        self.stale_timeout_sec = new_stale_timeout_sec
        self.reset_hold_sec = new_reset_hold_sec
        old_position_mapping_mode = self.position_mapping_mode
        self.position_mapping_mode = new_position_mapping_mode
        self.kp_linear = new_kp_linear
        self.max_linear_speed = new_max_linear_speed
        self.linear_deadband = new_linear_deadband
        self.kp_angular = new_kp_angular
        self.max_angular_speed = new_max_angular_speed
        self.angular_deadband = new_angular_deadband
        self.attachment_kp_angular = new_attachment_kp_angular
        self.attachment_max_angular_speed = new_attachment_max_angular_speed
        self.attachment_angular_deadband = new_attachment_angular_deadband
        self.start_in_gamepad_mode = new_start_in_gamepad_mode
        self.control_mode = new_control_mode
        self.allow_unity_control_mode_switch = new_allow_unity_control_mode_switch
        self.gamepad_deadband = new_gamepad_deadband
        self.gamepad_linear_speed_xyz = new_gamepad_linear_speed_xyz
        self.gamepad_linear_sign_xyz = new_gamepad_linear_sign_xyz
        self.gamepad_angular_speed_xyz = new_gamepad_angular_speed_xyz
        self.gamepad_angular_sign_xyz = new_gamepad_angular_sign_xyz
        self.map_axes = new_map_axes
        self.map_signs = new_map_signs
        self.rot_map_axes = new_rot_map_axes
        self.rot_map_signs = new_rot_map_signs
        self.rot_scale_xyz = new_rot_scale_xyz
        self.debug_rotation_mapping = new_debug_rotation_mapping
        self.debug_rotation_log_period_sec = new_debug_rotation_log_period_sec
        self.scale_xyz = new_scale_xyz
        self.offset_xyz = new_offset_xyz
        self.min_xyz = new_min_xyz
        self.max_xyz = new_max_xyz
        self.attachment_use_absolute_position = new_attachment_use_absolute_position
        self.attachment_enable_rotation = new_attachment_enable_rotation
        self.attachment_use_absolute_rotation = new_attachment_use_absolute_rotation
        self.attachment_base_xyz = new_attachment_base_xyz
        self.attachment_base_xyzw = new_attachment_base_xyzw
        self.attachment_tool_position_offset_xyz = new_attachment_tool_position_offset_xyz
        self.attachment_tool_rotation_offset_xyzw = new_attachment_tool_rotation_offset_xyzw

        if (
            old_position_mapping_mode != self.position_mapping_mode
            or any(
                name in touched
                for name in (
                    "target_frame",
                    "ee_frame",
                    "map_axes",
                    "map_signs",
                    "scale_xyz",
                    "offset_xyz",
                    "attachment_use_absolute_position",
                    "attachment_enable_rotation",
                    "attachment_use_absolute_rotation",
                    "attachment_base_xyz",
                    "attachment_base_xyzw",
                    "attachment_tool_position_offset_xyz",
                    "attachment_tool_rotation_offset_xyzw",
                    "attachment_kp_angular",
                    "attachment_max_angular_speed",
                    "attachment_angular_deadband",
                )
            )
        ):
            self._position_session_active = False
            self._attachment_session_active = False

        if touched:
            if "control_mode" in touched or "start_in_gamepad_mode" in touched:
                self._gamepad_mode = self.control_mode == "gamepad"
                self._request_position_recenter()
                self.get_logger().warn(f"Control mode set to {self.control_mode.upper()} by parameter update.")

            self.get_logger().info(
                f"Updated mapper params ({', '.join(touched)}): "
                f"target_frame={self.target_frame}, ee_frame={self.ee_frame}, "
                f"position_mapping_mode={self.position_mapping_mode}, control_mode={self.control_mode}, "
                f"allow_unity_control_mode_switch={self.allow_unity_control_mode_switch}, "
                f"kp_linear={self.kp_linear:.3f}, max_linear_speed={self.max_linear_speed:.3f}, linear_deadband={self.linear_deadband:.4f}, "
                f"kp_angular={self.kp_angular:.3f}, max_angular_speed={self.max_angular_speed:.3f}, angular_deadband={self.angular_deadband:.4f}, "
                f"gamepad_deadband={self.gamepad_deadband:.3f}, "
                f"gamepad_linear_speed_xyz={self.gamepad_linear_speed_xyz.tolist()}, "
                f"gamepad_linear_sign_xyz={self.gamepad_linear_sign_xyz.tolist()}, "
                f"gamepad_angular_speed_xyz={self.gamepad_angular_speed_xyz.tolist()}, "
                f"gamepad_angular_sign_xyz={self.gamepad_angular_sign_xyz.tolist()}, "
                f"pos_map_axes={self.map_axes}, pos_map_signs={self.map_signs.tolist()}, "
                f"rot_map_axes={self.rot_map_axes}, rot_map_signs={self.rot_map_signs.tolist()}, rot_scale_xyz={self.rot_scale_xyz.tolist()}, "
                f"debug_rotation_mapping={self.debug_rotation_mapping}, debug_rotation_log_period_sec={self.debug_rotation_log_period_sec:.3f}, "
                f"scale_xyz={self.scale_xyz.tolist()}, offset_xyz={self.offset_xyz.tolist()}, "
                f"min_xyz={self.min_xyz.tolist()}, max_xyz={self.max_xyz.tolist()}, "
                f"attachment_abs_pos={self.attachment_use_absolute_position}, "
                f"attachment_enable_rot={self.attachment_enable_rotation}, "
                f"attachment_abs_rot={self.attachment_use_absolute_rotation}, "
                f"attachment_kp_angular={self.attachment_kp_angular:.3f}, "
                f"attachment_max_angular_speed={self.attachment_max_angular_speed:.3f}, "
                f"attachment_angular_deadband={self.attachment_angular_deadband:.4f}, "
                f"attachment_base_xyz={self.attachment_base_xyz.tolist()}, "
                f"attachment_base_xyzw={self.attachment_base_xyzw.tolist()}, "
                f"attachment_tool_pos_offset={self.attachment_tool_position_offset_xyz.tolist()}, "
                f"attachment_tool_rot_offset={self.attachment_tool_rotation_offset_xyzw.tolist()}"
            )

        return SetParametersResult(successful=True)

    def _on_pose_states(self, msg: ReceivedPoseStates):
        self._last_rx_time = time.monotonic()
        self._rx_count += 1

        self._tracked = bool(msg.tracked)
        self._rotate_enable = bool(msg.rotate_enable)
        self._gripper_cmd = self._arbitrate_gripper(msg.close_enable, msg.open_enable)
        legacy_reset_button = bool(msg.reset_enable)
        reset_robot_button = bool(getattr(msg, "reset_robot_enable", legacy_reset_button))
        reset_scene_button = bool(getattr(msg, "reset_scene_enable", legacy_reset_button))
        reset_button = reset_robot_button or reset_scene_button

        if reset_robot_button and (not self._reset_robot_button_last):
            self._reset_latched = True
            self._reset_latch_until = time.monotonic() + self.reset_hold_sec
            self._request_position_recenter()
            self.get_logger().warn(
                f"Robot reset requested; mapper will pause teleop for "
                f"{self.reset_hold_sec:.1f}s while reset manager runs."
            )
        if reset_scene_button and (not self._reset_scene_button_last):
            self._reset_scene_latch_until = time.monotonic() + 0.5
            self.get_logger().warn("Scene-object reset requested.")
        recenter_button = bool(msg.recenter_enable)
        if recenter_button and (not self._recenter_button_last):
            self._request_position_recenter()
            self.get_logger().warn("Right thumbstick clutch held: arm motion paused until release.")
        elif (not recenter_button) and self._recenter_button_last:
            self.get_logger().warn("Right thumbstick clutch released: current hand pose will become the new position reference.")
        teleop_enable = bool(msg.teleop_enable)
        if teleop_enable and (not self._teleop_enable_last):
            self._request_position_recenter()
            self.get_logger().warn("Teleop engaged by right grip: current hand pose will become the new position reference.")
        elif (not teleop_enable) and self._teleop_enable_last:
            self._position_session_active = False
            self._rotate_session_active = False
            self._attachment_session_active = False
            self.get_logger().warn("Teleop disengaged by right grip release.")
        attachment_mode = bool(getattr(msg, "attachment_mode", False))
        if attachment_mode != self._attachment_mode_last:
            self._position_session_active = False
            self._rotate_session_active = False
            self._attachment_session_active = False
            mode_name = "ATTACHMENT" if attachment_mode else "DELTA"
            self.get_logger().warn(f"Position mapping switched to {mode_name} behavior by Unity attachment state.")
        mode_switch = bool(msg.mode_switch_enable)
        requested_control_mode = self._parse_optional_control_mode(getattr(msg, "control_mode", ""))
        if self.allow_unity_control_mode_switch:
            if requested_control_mode is not None:
                requested_gamepad_mode = requested_control_mode == "gamepad"
                if requested_gamepad_mode != self._gamepad_mode:
                    self._gamepad_mode = requested_gamepad_mode
                    self.control_mode = requested_control_mode
                    self.start_in_gamepad_mode = requested_gamepad_mode
                    self._request_position_recenter()
                    self.get_logger().warn(f"Control mode set to {requested_control_mode.upper()} by Unity state.")
            elif mode_switch and (not self._mode_switch_last):
                self._gamepad_mode = not self._gamepad_mode
                self.control_mode = "gamepad" if self._gamepad_mode else "hand_pose"
                self.start_in_gamepad_mode = self._gamepad_mode
                self._request_position_recenter()
                mode_name = "GAMEPAD" if self._gamepad_mode else "HAND_POSE"
                self.get_logger().warn(f"Control mode toggled to {mode_name} by left Y-button edge.")
        requested_mapping_mode = self._parse_optional_position_mapping_mode(getattr(msg, "mapping_mode", ""))
        if requested_mapping_mode is not None and requested_mapping_mode != self.position_mapping_mode:
            self.position_mapping_mode = requested_mapping_mode
            self._request_position_recenter()
            self.get_logger().warn(f"Position mapping mode set to {requested_mapping_mode} by Unity state.")

        workspace_valid = bool(getattr(msg, "workspace_pose_valid", False))
        if workspace_valid:
            self._workspace_pos = np.array(
                [
                    float(msg.workspace_pose.position.x),
                    float(msg.workspace_pose.position.y),
                    float(msg.workspace_pose.position.z),
                ],
                dtype=float,
            )
            self._workspace_rot = self._normalize_quat(
                np.array(
                    [
                        float(msg.workspace_pose.orientation.x),
                        float(msg.workspace_pose.orientation.y),
                        float(msg.workspace_pose.orientation.z),
                        float(msg.workspace_pose.orientation.w),
                    ],
                    dtype=float,
                )
            )
        self._workspace_pose_valid = workspace_valid
        self._reset_button = reset_button
        self._reset_button_last = reset_button
        self._reset_robot_button_last = reset_robot_button
        self._reset_scene_button_last = reset_scene_button
        self._recenter_button_last = recenter_button
        self._recenter_hold_active = recenter_button
        self._teleop_enable = teleop_enable
        self._teleop_enable_last = teleop_enable
        self._attachment_mode = attachment_mode
        self._attachment_mode_last = attachment_mode
        self._mode_switch_last = mode_switch
        self._attachment_adjustment_mode = bool(getattr(msg, "attachment_adjustment_mode", False))

        attachment_offset_valid = bool(getattr(msg, "attachment_offset_valid", False))
        if attachment_offset_valid and hasattr(msg, "attachment_offset"):
            self._attachment_runtime_offset_valid = True
            self._attachment_position_offset_unity_workspace = np.array(
                [
                    float(msg.attachment_offset.position.x),
                    float(msg.attachment_offset.position.y),
                    float(msg.attachment_offset.position.z),
                ],
                dtype=float,
            )
            self._attachment_rotation_offset_unity = self._normalize_quat(
                np.array(
                    [
                        float(msg.attachment_offset.orientation.x),
                        float(msg.attachment_offset.orientation.y),
                        float(msg.attachment_offset.orientation.z),
                        float(msg.attachment_offset.orientation.w),
                    ],
                    dtype=float,
                )
            )
        else:
            self._attachment_runtime_offset_valid = False
            self._attachment_position_offset_unity_workspace = np.zeros(3, dtype=float)
            self._attachment_rotation_offset_unity = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)

        if not self._tracked:
            self._have_target = False
            self._position_session_active = False
            self._rotate_session_active = False
            return

        unity_pos = np.array(
            [float(msg.pose.position.x), float(msg.pose.position.y), float(msg.pose.position.z)],
            dtype=float,
        )
        self._latest_hand_world_pos = unity_pos.copy()
        hand_workspace_pos = self._unity_world_pos_to_workspace(unity_pos)
        if self._uses_unity_world_mapping():
            mapped_pos = self._map_unity_vector(hand_workspace_pos)
        else:
            mapped_pos = self._map_unity_vector(unity_pos)
        self._latest_hand_pos = mapped_pos
        if self._position_recenter_pending:
            self._position_recenter_hand_ref = mapped_pos.copy()
            self._position_recenter_hand_world_ref = unity_pos.copy()
            self._position_recenter_hand_ref_valid = True
        if self.position_mapping_mode in ("absolute", "unity_world_absolute"):
            scaled_pos = (mapped_pos * self.scale_xyz) + self.offset_xyz
            self._target_pos = np.clip(scaled_pos, self.min_xyz, self.max_xyz)
        self._have_target = True

        q = np.array(
            [
                float(msg.pose.orientation.x),
                float(msg.pose.orientation.y),
                float(msg.pose.orientation.z),
                float(msg.pose.orientation.w),
            ],
            dtype=float,
        )
        self._hand_world_rot = self._normalize_quat(q)
        self._hand_rot = self._unity_world_rot_to_workspace(self._hand_world_rot) if self._uses_unity_world_mapping() else self._hand_world_rot
        self._left_stick = np.array([float(msg.left_stick_x), float(msg.left_stick_y)], dtype=float)
        self._right_stick = np.array([float(msg.right_stick_x), float(msg.right_stick_y)], dtype=float)
        self._left_grip_value = float(msg.left_grip_value)
        self._left_trigger_value = float(msg.left_trigger_value)

        if not self._rotate_enable:
            self._rotate_session_active = False

    def _publish_loop(self):
        now = time.monotonic()
        stale = (now - self._last_rx_time) > self.stale_timeout_sec
        packet_active = self._have_target and not stale
        input_active = self._tracked and packet_active
        arm_input_active = input_active and self._teleop_enable

        linear = np.zeros(3, dtype=float)
        angular = np.zeros(3, dtype=float)
        gripper_cmd = self._gripper_cmd if packet_active else 0
        rotate_enable = False
        tracked_active = False
        reset_enable = False
        reset_robot_enable = False
        reset_scene_enable = False
        tf_ok = False
        ee_pos_for_pose = None
        ee_rot_for_pose = None
        target_pos_for_pose = None
        target_rot_for_pose = None

        if self._reset_latched and now >= self._reset_latch_until:
            self._reset_latched = False
            self._reset_latch_until = 0.0
            self.get_logger().warn("Reset pause auto-released; teleop can resume.")

        # Robot reset is latched long enough for the reset manager to stop Servo,
        # home the arm, and restart Servo. Scene reset is a short pulse because it
        # only asks Gazebo to restore task-object poses.
        reset_robot_enable = self._reset_latched
        reset_scene_enable = now <= self._reset_scene_latch_until
        reset_enable = reset_robot_enable

        if reset_robot_enable:
            tracked_active = True
            self._position_session_active = False
            self._rotate_session_active = False
            self._attachment_session_active = False
        elif self._recenter_hold_active:
            tracked_active = input_active
            self._position_session_active = False
            self._rotate_session_active = False
            self._attachment_session_active = False
        elif arm_input_active:
            if self._gamepad_mode:
                tracked_active = True
                self._position_session_active = False
                linear = self._compute_gamepad_linear()
                pose_tf_ok, ee_pos_for_pose, ee_rot_for_pose = self._lookup_ee_pose(now)
                if pose_tf_ok:
                    tf_ok = True
                    target_pos_for_pose = ee_pos_for_pose + self._estimate_velocity_target_offset(linear)
                    target_rot_for_pose = ee_rot_for_pose
                if self._rotate_enable:
                    tf_ok, _, ee_rot = self._lookup_ee_pose(now)
                    if tf_ok:
                        ee_rot_for_pose = ee_rot
                        if not self._rotate_session_active:
                            self._rotate_hand_ref = self._hand_rot.copy()
                            self._rotate_ee_ref = ee_rot.copy()
                            self._rotate_session_active = True

                        hand_delta_rotvec = self._quat_delta_to_rotvec(self._rotate_hand_ref, self._hand_rot)
                        mapped_delta_rotvec = self._map_rotation_delta(hand_delta_rotvec)
                        target_delta_quat = self._rotvec_to_quat(mapped_delta_rotvec)
                        target_ee_rot = self._normalize_quat(self._quat_multiply(target_delta_quat, self._rotate_ee_ref))
                        target_rot_for_pose = target_ee_rot

                        angular_error = self._quat_delta_to_rotvec(ee_rot, target_ee_rot)
                        angular = self.kp_angular * angular_error
                        angular = self._apply_speed_limits(angular, self.max_angular_speed, self.angular_deadband)
                        rotate_enable = True
                    else:
                        self._rotate_session_active = False
                else:
                    self._rotate_session_active = False
                    angular = self._compute_gamepad_angular()
                    if np.linalg.norm(angular) > 0.0:
                        rotate_enable = True
            else:
                tf_ok, ee_pos, ee_rot = self._lookup_ee_pose(now)
                if tf_ok:
                    tracked_active = True

                    if self._attachment_mode:
                        self._ensure_attachment_reference(ee_pos, ee_rot)
                    target_pos = self._compute_position_target(ee_pos)
                    ee_pos_for_pose = ee_pos
                    ee_rot_for_pose = ee_rot
                    target_pos_for_pose = target_pos
                    target_rot_for_pose = ee_rot
                    linear_error = target_pos - ee_pos
                    linear = self.kp_linear * linear_error
                    linear = self._apply_speed_limits(linear, self.max_linear_speed, self.linear_deadband)

                    if self._attachment_mode:
                        self._rotate_session_active = False
                        if self.attachment_enable_rotation:
                            target_ee_rot = self._compute_attachment_rotation_target()
                            target_rot_for_pose = target_ee_rot

                            angular_error = self._quat_delta_to_rotvec(ee_rot, target_ee_rot)
                            angular = self.attachment_kp_angular * angular_error
                            angular = self._apply_speed_limits(
                                angular,
                                self.attachment_max_angular_speed,
                                self.attachment_angular_deadband,
                            )
                            rotate_enable = True
                        else:
                            target_rot_for_pose = ee_rot
                    elif self._rotate_enable:
                        if not self._rotate_session_active:
                            self._rotate_hand_ref = self._hand_rot.copy()
                            self._rotate_ee_ref = ee_rot.copy()
                            self._rotate_session_active = True

                        hand_delta_rotvec = self._quat_delta_to_rotvec(self._rotate_hand_ref, self._hand_rot)
                        mapped_delta_rotvec = self._map_rotation_delta(hand_delta_rotvec)
                        target_delta_quat = self._rotvec_to_quat(mapped_delta_rotvec)
                        target_ee_rot = self._normalize_quat(self._quat_multiply(target_delta_quat, self._rotate_ee_ref))
                        target_rot_for_pose = target_ee_rot

                        angular_error = self._quat_delta_to_rotvec(ee_rot, target_ee_rot)
                        angular = self.kp_angular * angular_error
                        angular = self._apply_speed_limits(angular, self.max_angular_speed, self.angular_deadband)
                        rotate_enable = True
                    else:
                        self._rotate_session_active = False

                else:
                    self._rotate_session_active = False

        else:
            self._position_session_active = False
            self._rotate_session_active = False
            self._attachment_session_active = False

        msg = TargetTwistStates()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.target_frame

        msg.twist.linear.x = float(linear[0])
        msg.twist.linear.y = float(linear[1])
        msg.twist.linear.z = float(linear[2])
        msg.twist.angular.x = float(angular[0])
        msg.twist.angular.y = float(angular[1])
        msg.twist.angular.z = float(angular[2])
        msg.gripper_cmd = int(gripper_cmd)
        msg.rotate_enable = bool(rotate_enable)
        msg.tracked = bool(tracked_active or (packet_active and gripper_cmd != 0))
        msg.reset_enable = bool(reset_enable)
        if hasattr(msg, "reset_robot_enable"):
            msg.reset_robot_enable = bool(reset_robot_enable)
        if hasattr(msg, "reset_scene_enable"):
            msg.reset_scene_enable = bool(reset_scene_enable)
        self.pub.publish(msg)
        self._publish_ee_pose_pair(
            msg.header.stamp,
            ee_pos_for_pose,
            ee_rot_for_pose,
            target_pos_for_pose,
            target_rot_for_pose,
        )
        if rotate_enable:
            self._log_rotation_debug(now, angular, ee_rot_for_pose, target_rot_for_pose)

        if now - self._last_log_time > 2.0:
            dt = max(now - self._rx_window_start, 1e-6)
            rx_hz = self._rx_count / dt
            self.get_logger().info(
                f"RX {rx_hz:.1f} Hz, mode={'gamepad' if self._gamepad_mode else 'hand_pose'}, "
                f"stale={stale}, tracked={tracked_active}, tf_ok={tf_ok}, rotate={rotate_enable}, "
                f"reset={reset_enable}, reset_robot={reset_robot_enable}, reset_scene={reset_scene_enable}, teleop={self._teleop_enable}, attachment={self._attachment_mode}, clutch={self._recenter_hold_active}, gripper_cmd={gripper_cmd}, lin=({linear[0]:.3f},{linear[1]:.3f},{linear[2]:.3f}), "
                f"ang=({angular[0]:.3f},{angular[1]:.3f},{angular[2]:.3f})"
            )
            self._rx_count = 0
            self._rx_window_start = now
            self._last_log_time = now

    def _estimate_velocity_target_offset(self, linear: np.ndarray) -> np.ndarray:
        if self.kp_linear > 1e-6:
            return linear / self.kp_linear
        return linear * 0.1

    def _publish_ee_pose_pair(
        self,
        stamp,
        actual_pos,
        actual_rot,
        target_pos,
        target_rot,
    ):
        if not self.publish_ee_pose_topics:
            return
        if actual_pos is None or actual_rot is None or target_pos is None:
            return
        if target_rot is None:
            target_rot = actual_rot

        self.actual_pose_pub.publish(self._make_pose_msg(stamp, actual_pos, actual_rot))
        self.target_pose_pub.publish(self._make_pose_msg(stamp, target_pos, target_rot))

    def _make_pose_msg(self, stamp, pos: np.ndarray, rot: np.ndarray) -> PoseStamped:
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self.target_frame
        msg.pose.position.x = float(pos[0])
        msg.pose.position.y = float(pos[1])
        msg.pose.position.z = float(pos[2])
        msg.pose.orientation.x = float(rot[0])
        msg.pose.orientation.y = float(rot[1])
        msg.pose.orientation.z = float(rot[2])
        msg.pose.orientation.w = float(rot[3])
        return msg

    @staticmethod
    def _arbitrate_gripper(close_enable: bool, open_enable: bool) -> int:
        if close_enable and not open_enable:
            return 1
        if open_enable and not close_enable:
            return -1
        return 0

    @staticmethod
    def _remap_vector(unity_xyz: np.ndarray, axes, signs: np.ndarray) -> np.ndarray:
        lookup = {
            "x": float(unity_xyz[0]),
            "y": float(unity_xyz[1]),
            "z": float(unity_xyz[2]),
        }
        return np.array(
            [
                lookup[axes[0]],
                lookup[axes[1]],
                lookup[axes[2]],
            ],
            dtype=float,
        ) * signs

    def _map_unity_vector(self, unity_xyz: np.ndarray) -> np.ndarray:
        return self._remap_vector(unity_xyz, self.map_axes, self.map_signs)

    def _map_rotation_delta(self, unity_rotvec: np.ndarray) -> np.ndarray:
        mapped = self._remap_vector(unity_rotvec, self.rot_map_axes, self.rot_map_signs) * self.rot_scale_xyz
        self._last_rotation_input_rotvec = unity_rotvec.copy()
        self._last_rotation_mapped_rotvec = mapped.copy()
        return mapped

    def _log_rotation_debug(
        self,
        now: float,
        angular_cmd: np.ndarray,
        actual_rot,
        target_rot,
    ):
        if not self.debug_rotation_mapping:
            return
        if now - self._last_rotation_debug_log_time < self.debug_rotation_log_period_sec:
            return
        self._last_rotation_debug_log_time = now

        angular_error = np.zeros(3, dtype=float)
        if actual_rot is not None and target_rot is not None:
            angular_error = self._quat_delta_to_rotvec(actual_rot, target_rot)

        self.get_logger().warn(
            "ROT_DEBUG "
            f"frame={self.target_frame}, ee={self.ee_frame}, "
            f"mode={'gamepad' if self._gamepad_mode else 'hand_pose'}, "
            f"attachment={self._attachment_mode}, workspace_valid={self._workspace_pose_valid}, "
            f"raw_rotvec={self._fmt_vec(self._last_rotation_input_rotvec)}, "
            f"mapped_rotvec={self._fmt_vec(self._last_rotation_mapped_rotvec)}, "
            f"angular_error={self._fmt_vec(angular_error)}, "
            f"angular_cmd={self._fmt_vec(angular_cmd)}, "
            f"rot_map_axes={self.rot_map_axes}, "
            f"rot_map_signs={self.rot_map_signs.tolist()}, "
            f"rot_scale_xyz={self.rot_scale_xyz.tolist()}"
        )

    @staticmethod
    def _fmt_vec(vec: np.ndarray) -> str:
        return f"({vec[0]:+.3f},{vec[1]:+.3f},{vec[2]:+.3f})"

    def _uses_unity_world_mapping(self) -> bool:
        return self.position_mapping_mode in ("unity_world_delta", "unity_world_absolute")

    def _request_position_recenter(self):
        self._position_session_active = False
        self._attachment_session_active = False
        self._position_recenter_pending = True
        self._position_recenter_hand_ref_valid = False
        self._position_recenter_hand_world_ref = np.zeros(3, dtype=float)

    def _compute_position_target(self, ee_pos: np.ndarray) -> np.ndarray:
        if self.position_mapping_mode in ("absolute", "unity_world_absolute"):
            return self._target_pos
        if self._attachment_mode:
            return self._compute_attachment_position_target()

        if (not self._position_session_active) or self._position_recenter_pending:
            if self._position_recenter_hand_ref_valid:
                self._position_hand_ref = self._position_recenter_hand_ref.copy()
                self._position_hand_world_ref = self._position_recenter_hand_world_ref.copy()
            else:
                self._position_hand_ref = self._latest_hand_pos.copy()
                self._position_hand_world_ref = self._latest_hand_world_pos.copy()
            self._position_ee_ref = ee_pos.copy()
            self._position_session_active = True
            self._position_recenter_pending = False
            self._position_recenter_hand_ref_valid = False
            self.get_logger().info(
                f"Captured {self.position_mapping_mode} position reference: "
                f"hand={self._position_hand_ref.tolist()}, "
                f"hand_world={self._position_hand_world_ref.tolist()}, ee={self._position_ee_ref.tolist()}"
            )

        if self.position_mapping_mode == "unity_world_delta":
            # Keep the user's MR/headset world as the fixed motion reference.
            # The current workspace rotation converts that world delta into the
            # simulation/workspace frame so the visible EE moves with the hand
            # even after the workspace is yawed or moved.
            unity_world_delta = self._latest_hand_world_pos - self._position_hand_world_ref
            unity_workspace_delta = self._unity_world_vector_to_workspace(unity_world_delta)
            hand_delta = self._map_unity_vector(unity_workspace_delta)
        else:
            hand_delta = self._latest_hand_pos - self._position_hand_ref
        target_pos = self._position_ee_ref + (hand_delta * self.scale_xyz) + self.offset_xyz
        self._target_pos = np.clip(target_pos, self.min_xyz, self.max_xyz)
        return self._target_pos

    def _ensure_attachment_reference(self, ee_pos: np.ndarray, ee_rot: np.ndarray):
        if self._attachment_session_active:
            return

        self._attachment_hand_ref = self._latest_hand_pos.copy()
        self._attachment_hand_world_ref = self._latest_hand_world_pos.copy()
        self._attachment_ee_ref = ee_pos.copy()
        self._attachment_hand_rot_ref = self._hand_rot.copy()
        self._attachment_ee_rot_ref = ee_rot.copy()
        self._attachment_session_active = True
        self.get_logger().info(
            "Captured attachment reference: "
            f"hand={self._attachment_hand_ref.tolist()}, "
            f"hand_world={self._attachment_hand_world_ref.tolist()}, "
            f"ee={self._attachment_ee_ref.tolist()}"
        )

    def _compute_attachment_position_target(self) -> np.ndarray:
        if not self._attachment_session_active:
            self._target_pos = np.clip(self._target_pos, self.min_xyz, self.max_xyz)
            return self._target_pos

        if self.attachment_use_absolute_position and self._workspace_pose_valid:
            desired_unity_world_pos = self._latest_hand_world_pos.copy()
            if self._attachment_runtime_offset_valid:
                desired_unity_world_pos = desired_unity_world_pos + self._unity_workspace_vector_to_world(
                    self._attachment_position_offset_unity_workspace
                )
            desired_gazebo_world = self._unity_world_pos_to_gazebo_world(desired_unity_world_pos)
            if np.any(np.abs(self.attachment_tool_position_offset_xyz) > 1e-9):
                hand_gazebo_world_rot = self._unity_world_rot_to_gazebo_world(self._hand_world_rot)
                desired_gazebo_world = desired_gazebo_world + self._quat_rotate_vector(
                    hand_gazebo_world_rot,
                    self.attachment_tool_position_offset_xyz,
                )
            target_pos = self._gazebo_world_pos_to_target_frame(desired_gazebo_world)
            self._target_pos = np.clip(target_pos, self.min_xyz, self.max_xyz)
            return self._target_pos

        if self.position_mapping_mode == "unity_world_delta":
            unity_world_delta = self._latest_hand_world_pos - self._attachment_hand_world_ref
            unity_workspace_delta = self._unity_world_vector_to_workspace(unity_world_delta)
            hand_delta = self._map_unity_vector(unity_workspace_delta)
        else:
            hand_delta = self._latest_hand_pos - self._attachment_hand_ref

        target_pos = self._attachment_ee_ref + (hand_delta * self.scale_xyz)
        self._target_pos = np.clip(target_pos, self.min_xyz, self.max_xyz)
        return self._target_pos

    def _compute_attachment_rotation_target(self) -> np.ndarray:
        if not self._attachment_session_active:
            return self._attachment_ee_rot_ref

        if self.attachment_use_absolute_rotation and self._workspace_pose_valid:
            desired_unity_world_rot = self._hand_world_rot
            if self._attachment_runtime_offset_valid:
                desired_unity_world_rot = self._normalize_quat(
                    self._quat_multiply(desired_unity_world_rot, self._attachment_rotation_offset_unity)
                )
            desired_gazebo_world_rot = self._unity_world_rot_to_gazebo_world(desired_unity_world_rot)
            desired_gazebo_world_rot = self._normalize_quat(
                self._quat_multiply(desired_gazebo_world_rot, self.attachment_tool_rotation_offset_xyzw)
            )
            return self._gazebo_world_rot_to_target_frame(desired_gazebo_world_rot)

        hand_delta_rotvec = self._quat_delta_to_rotvec(self._attachment_hand_rot_ref, self._hand_rot)
        mapped_delta_rotvec = self._map_rotation_delta(hand_delta_rotvec)
        target_delta_quat = self._rotvec_to_quat(mapped_delta_rotvec)
        return self._normalize_quat(self._quat_multiply(target_delta_quat, self._attachment_ee_rot_ref))

    def _unity_world_pos_to_gazebo_world(self, unity_world_pos: np.ndarray) -> np.ndarray:
        unity_workspace_pos = self._unity_world_pos_to_workspace(unity_world_pos)
        return self._unity_workspace_vector_to_gazebo_world(unity_workspace_pos)

    def _unity_world_rot_to_gazebo_world(self, unity_world_rot: np.ndarray) -> np.ndarray:
        unity_workspace_rot = self._unity_world_rot_to_workspace(unity_world_rot)
        return self._unity_workspace_rot_to_gazebo_world(unity_workspace_rot)

    @staticmethod
    def _unity_workspace_vector_to_gazebo_world(unity_xyz: np.ndarray) -> np.ndarray:
        # Inverse of Unity.Robotics ROSGeometry FLU conversion:
        # Unity local (x=right, y=up, z=forward) -> Gazebo/ROS world (x=forward, y=left, z=up).
        return np.array(
            [
                float(unity_xyz[2]),
                -float(unity_xyz[0]),
                float(unity_xyz[1]),
            ],
            dtype=float,
        )

    def _unity_workspace_rot_to_gazebo_world(self, unity_quat: np.ndarray) -> np.ndarray:
        # Same basis conversion as vectors, but applied to a full orientation matrix.
        gazebo_to_unity = np.array(
            [
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=float,
        )
        unity_rot = self._quat_to_matrix(unity_quat)
        gazebo_rot = gazebo_to_unity.T @ unity_rot @ gazebo_to_unity
        return self._matrix_to_quat(gazebo_rot)

    def _gazebo_world_pos_to_target_frame(self, gazebo_world_pos: np.ndarray) -> np.ndarray:
        delta_world = gazebo_world_pos - self.attachment_base_xyz
        return self._quat_rotate_vector(self._quat_conjugate(self.attachment_base_xyzw), delta_world)

    def _gazebo_world_rot_to_target_frame(self, gazebo_world_rot: np.ndarray) -> np.ndarray:
        return self._normalize_quat(
            self._quat_multiply(self._quat_conjugate(self.attachment_base_xyzw), gazebo_world_rot)
        )

    def _quat_to_matrix(self, q: np.ndarray) -> np.ndarray:
        x, y, z, w = self._normalize_quat(q)
        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        wx = w * x
        wy = w * y
        wz = w * z
        return np.array(
            [
                [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
                [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
                [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
            ],
            dtype=float,
        )

    def _matrix_to_quat(self, m: np.ndarray) -> np.ndarray:
        trace = float(m[0, 0] + m[1, 1] + m[2, 2])
        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * s
            qx = (m[2, 1] - m[1, 2]) / s
            qy = (m[0, 2] - m[2, 0]) / s
            qz = (m[1, 0] - m[0, 1]) / s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = math.sqrt(max(0.0, 1.0 + m[0, 0] - m[1, 1] - m[2, 2])) * 2.0
            qw = (m[2, 1] - m[1, 2]) / s
            qx = 0.25 * s
            qy = (m[0, 1] + m[1, 0]) / s
            qz = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = math.sqrt(max(0.0, 1.0 + m[1, 1] - m[0, 0] - m[2, 2])) * 2.0
            qw = (m[0, 2] - m[2, 0]) / s
            qx = (m[0, 1] + m[1, 0]) / s
            qy = 0.25 * s
            qz = (m[1, 2] + m[2, 1]) / s
        else:
            s = math.sqrt(max(0.0, 1.0 + m[2, 2] - m[0, 0] - m[1, 1])) * 2.0
            qw = (m[1, 0] - m[0, 1]) / s
            qx = (m[0, 2] + m[2, 0]) / s
            qy = (m[1, 2] + m[2, 1]) / s
            qz = 0.25 * s
        return self._normalize_quat(np.array([qx, qy, qz, qw], dtype=float))


    def _compute_gamepad_linear(self) -> np.ndarray:
        raw = np.array(
            [
                float(self._left_stick[1]),
                float(self._left_stick[0]),
                float(self._left_trigger_value - self._left_grip_value),
            ],
            dtype=float,
        )
        processed = np.zeros(3, dtype=float)
        for i, value in enumerate(raw):
            if abs(value) >= self.gamepad_deadband:
                processed[i] = value
        return processed * self.gamepad_linear_speed_xyz * self.gamepad_linear_sign_xyz

    def _compute_gamepad_angular(self) -> np.ndarray:
        raw = np.array(
            [
                0.0,
                float(self._right_stick[1]),
                float(self._right_stick[0]),
            ],
            dtype=float,
        )
        processed = np.zeros(3, dtype=float)
        for i, value in enumerate(raw):
            if abs(value) >= self.gamepad_deadband:
                processed[i] = value
        return processed * self.gamepad_angular_speed_xyz * self.gamepad_angular_sign_xyz

    @staticmethod
    def _normalize_quat(q: np.ndarray) -> np.ndarray:
        if q.shape != (4,):
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        n = float(np.linalg.norm(q))
        if n < 1e-9:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        return q / n

    @staticmethod
    def _quat_conjugate(q: np.ndarray) -> np.ndarray:
        return np.array([-q[0], -q[1], -q[2], q[3]], dtype=float)

    @staticmethod
    def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ax, ay, az, aw = a
        bx, by, bz, bw = b
        return np.array(
            [
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
                aw * bw - ax * bx - ay * by - az * bz,
            ],
            dtype=float,
        )

    def _quat_rotate_vector(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        q = self._normalize_quat(q)
        vec_quat = np.array([float(v[0]), float(v[1]), float(v[2]), 0.0], dtype=float)
        rotated = self._quat_multiply(self._quat_multiply(q, vec_quat), self._quat_conjugate(q))
        return rotated[:3]

    def _unity_world_vector_to_workspace(self, unity_world_vector: np.ndarray) -> np.ndarray:
        if not self._workspace_pose_valid:
            return unity_world_vector
        return self._quat_rotate_vector(self._quat_conjugate(self._workspace_rot), unity_world_vector)

    def _unity_workspace_vector_to_world(self, unity_workspace_vector: np.ndarray) -> np.ndarray:
        if not self._workspace_pose_valid:
            return unity_workspace_vector
        return self._quat_rotate_vector(self._workspace_rot, unity_workspace_vector)

    def _unity_world_pos_to_workspace(self, unity_world_pos: np.ndarray) -> np.ndarray:
        if not self._workspace_pose_valid:
            return unity_world_pos
        return self._unity_world_vector_to_workspace(unity_world_pos - self._workspace_pos)

    def _unity_world_rot_to_workspace(self, unity_world_rot: np.ndarray) -> np.ndarray:
        if not self._workspace_pose_valid:
            return self._normalize_quat(unity_world_rot)
        return self._normalize_quat(
            self._quat_multiply(self._quat_conjugate(self._workspace_rot), unity_world_rot)
        )

    def _quat_delta_to_rotvec(self, q_ref: np.ndarray, q_curr: np.ndarray) -> np.ndarray:
        q_delta = self._quat_multiply(q_curr, self._quat_conjugate(q_ref))
        q_delta = self._normalize_quat(q_delta)

        if q_delta[3] < 0.0:
            q_delta = -q_delta

        xyz = q_delta[:3]
        xyz_norm = float(np.linalg.norm(xyz))
        if xyz_norm < 1e-9:
            return np.zeros(3, dtype=float)

        w = max(-1.0, min(1.0, float(q_delta[3])))
        angle = 2.0 * math.atan2(xyz_norm, w)
        axis = xyz / xyz_norm
        return axis * angle

    @staticmethod
    def _rotvec_to_quat(rotvec: np.ndarray) -> np.ndarray:
        angle = float(np.linalg.norm(rotvec))
        if angle < 1e-9:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)

        axis = rotvec / angle
        half = 0.5 * angle
        s = math.sin(half)
        c = math.cos(half)
        return np.array([axis[0] * s, axis[1] * s, axis[2] * s, c], dtype=float)

    def _lookup_ee_pose(self, now: float):
        try:
            tf = self._tf_buffer.lookup_transform(self.target_frame, self.ee_frame, rclpy.time.Time())
        except Exception as exc:
            if now - self._last_tf_warn_time > 2.0:
                self.get_logger().warn(
                    f"TF lookup failed for {self.target_frame} <- {self.ee_frame}: {exc}"
                )
                self._last_tf_warn_time = now
            return False, np.zeros(3, dtype=float), np.array([0.0, 0.0, 0.0, 1.0], dtype=float)

        ee_pos = np.array(
            [
                float(tf.transform.translation.x),
                float(tf.transform.translation.y),
                float(tf.transform.translation.z),
            ],
            dtype=float,
        )
        ee_rot = self._normalize_quat(
            np.array(
                [
                    float(tf.transform.rotation.x),
                    float(tf.transform.rotation.y),
                    float(tf.transform.rotation.z),
                    float(tf.transform.rotation.w),
                ],
                dtype=float,
            )
        )
        return True, ee_pos, ee_rot

    @staticmethod
    def _apply_speed_limits(vec: np.ndarray, max_speed: float, deadband: float) -> np.ndarray:
        mag = float(np.linalg.norm(vec))
        if mag < deadband:
            return np.zeros(3, dtype=float)
        if max_speed > 0.0 and mag > max_speed:
            vec = vec * (max_speed / mag)
        return vec


def main(args=None):
    rclpy.init(args=args)
    node = ReceivedPoseToTargetTwist()
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
