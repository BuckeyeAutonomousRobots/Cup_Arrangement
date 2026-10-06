import math
import time

import rclpy
from geometry_msgs.msg import Pose
from rclpy.node import Node
from teleop_bridge_msgs.msg import ReceivedPoseStates


class DualDebugHandGenerator(Node):
    """Publish Quest-like hand messages for backend-only Servo debugging."""

    def __init__(self):
        super().__init__("debug_hand_generator")

        self.declare_parameter("arm", "both")
        self.declare_parameter("left_topic", "/left_arm/received_pose_states")
        self.declare_parameter("right_topic", "/right_arm/received_pose_states")
        self.declare_parameter("publish_rate_hz", 60.0)
        self.declare_parameter("duration_sec", 0.0)
        self.declare_parameter("settle_sec", 1.0)
        self.declare_parameter("pattern", "line_x")
        self.declare_parameter("motion_period_sec", 4.0)
        self.declare_parameter("amplitude_x", 0.16)
        self.declare_parameter("amplitude_y", 0.0)
        self.declare_parameter("amplitude_z", 0.0)
        self.declare_parameter("left_base_x", 0.0)
        self.declare_parameter("left_base_y", 0.10)
        self.declare_parameter("left_base_z", 1.20)
        self.declare_parameter("right_base_x", 0.0)
        self.declare_parameter("right_base_y", -0.10)
        self.declare_parameter("right_base_z", 1.20)
        self.declare_parameter("workspace_pose_valid", True)
        self.declare_parameter("workspace_x", 0.0)
        self.declare_parameter("workspace_y", 0.0)
        self.declare_parameter("workspace_z", 0.0)
        self.declare_parameter("workspace_yaw_deg", 0.0)
        self.declare_parameter("teleop_enable", True)
        self.declare_parameter("rotate_enable", False)
        self.declare_parameter("attachment_mode", False)
        self.declare_parameter("control_mode", "hand_pose")
        self.declare_parameter("mapping_mode", "")

        arm = str(self.get_parameter("arm").value).strip().lower()
        if arm not in ("left", "right", "both"):
            self.get_logger().warn(f"Invalid arm={arm!r}; using both.")
            arm = "both"
        self.arm = arm

        self.left_topic = str(self.get_parameter("left_topic").value)
        self.right_topic = str(self.get_parameter("right_topic").value)
        self.publish_rate_hz = max(1.0, float(self.get_parameter("publish_rate_hz").value))
        self.duration_sec = max(0.0, float(self.get_parameter("duration_sec").value))
        self.settle_sec = max(0.0, float(self.get_parameter("settle_sec").value))
        self.pattern = str(self.get_parameter("pattern").value).strip().lower()
        self.motion_period_sec = max(0.25, float(self.get_parameter("motion_period_sec").value))
        self.amplitude = (
            float(self.get_parameter("amplitude_x").value),
            float(self.get_parameter("amplitude_y").value),
            float(self.get_parameter("amplitude_z").value),
        )
        self.left_base = (
            float(self.get_parameter("left_base_x").value),
            float(self.get_parameter("left_base_y").value),
            float(self.get_parameter("left_base_z").value),
        )
        self.right_base = (
            float(self.get_parameter("right_base_x").value),
            float(self.get_parameter("right_base_y").value),
            float(self.get_parameter("right_base_z").value),
        )
        self.workspace_pose_valid = bool(self.get_parameter("workspace_pose_valid").value)
        self.workspace_pos = (
            float(self.get_parameter("workspace_x").value),
            float(self.get_parameter("workspace_y").value),
            float(self.get_parameter("workspace_z").value),
        )
        self.workspace_yaw_deg = float(self.get_parameter("workspace_yaw_deg").value)
        self.teleop_enable = bool(self.get_parameter("teleop_enable").value)
        self.rotate_enable = bool(self.get_parameter("rotate_enable").value)
        self.attachment_mode = bool(self.get_parameter("attachment_mode").value)
        self.control_mode = str(self.get_parameter("control_mode").value)
        self.mapping_mode = str(self.get_parameter("mapping_mode").value)

        self.left_pub = self.create_publisher(ReceivedPoseStates, self.left_topic, 20)
        self.right_pub = self.create_publisher(ReceivedPoseStates, self.right_topic, 20)

        self.start_time = time.monotonic()
        self.done = False
        self.timer = self.create_timer(1.0 / self.publish_rate_hz, self._tick)

        self.get_logger().info(
            "Debug hand generator started: "
            f"arm={self.arm}, pattern={self.pattern}, rate={self.publish_rate_hz:.1f}Hz, "
            f"period={self.motion_period_sec:.2f}s, amplitude={self.amplitude}, "
            f"duration={'infinite' if self.duration_sec <= 0 else f'{self.duration_sec:.1f}s'}"
        )
        self.get_logger().info(
            "Stop quest_controller_receiver or close the Unity app while using this, "
            "otherwise live headset packets can overwrite the debug signal."
        )

    @staticmethod
    def _yaw_quat(degrees: float):
        half = math.radians(degrees) * 0.5
        return 0.0, 0.0, math.sin(half), math.cos(half)

    def _offset(self, t: float):
        if t < self.settle_sec:
            return 0.0, 0.0, 0.0

        u = 2.0 * math.pi * ((t - self.settle_sec) / self.motion_period_sec)
        s = math.sin(u)
        c = math.cos(u)
        ax, ay, az = self.amplitude

        if self.pattern == "line_y":
            amplitude_y = ay if abs(ay) > 1e-9 else ax
            return 0.0, amplitude_y * s, 0.0
        if self.pattern == "line_z":
            amplitude_z = az if abs(az) > 1e-9 else ax
            return 0.0, 0.0, amplitude_z * s
        if self.pattern == "circle_xy":
            radius_x = ax if abs(ax) > 1e-9 else 0.08
            radius_y = ay if abs(ay) > 1e-9 else radius_x
            return radius_x * c, radius_y * s, 0.0
        if self.pattern == "circle_xz":
            radius_x = ax if abs(ax) > 1e-9 else 0.08
            radius_z = az if abs(az) > 1e-9 else radius_x
            return radius_x * c, 0.0, radius_z * s
        return ax * s, ay * s, az * s

    def _make_msg(self, arm: str, base, offset, tracked: bool):
        msg = ReceivedPoseStates()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "unity_world"
        msg.tracked = tracked
        msg.pose.position.x = float(base[0] + offset[0])
        msg.pose.position.y = float(base[1] + offset[1])
        msg.pose.position.z = float(base[2] + offset[2])
        msg.pose.orientation.w = 1.0

        msg.workspace_pose_valid = self.workspace_pose_valid
        msg.workspace_pose.position.x = self.workspace_pos[0]
        msg.workspace_pose.position.y = self.workspace_pos[1]
        msg.workspace_pose.position.z = self.workspace_pos[2]
        qx, qy, qz, qw = self._yaw_quat(self.workspace_yaw_deg)
        msg.workspace_pose.orientation.x = qx
        msg.workspace_pose.orientation.y = qy
        msg.workspace_pose.orientation.z = qz
        msg.workspace_pose.orientation.w = qw

        msg.grip_value = 1.0 if self.teleop_enable else 0.0
        msg.trigger_value = 0.0
        msg.rotate_enable = self.rotate_enable
        msg.close_enable = False
        msg.open_enable = False
        msg.reset_enable = False
        msg.reset_robot_enable = False
        msg.reset_scene_enable = False
        msg.recenter_enable = False
        msg.teleop_enable = self.teleop_enable
        msg.source = f"debug_{arm}_hand_generator"
        msg.control_mode = self.control_mode
        msg.attachment_mode = self.attachment_mode
        msg.mapping_mode = self.mapping_mode
        msg.attachment_adjustment_mode = False
        msg.attachment_offset_valid = False
        msg.attachment_offset.orientation.w = 1.0
        msg.mode_switch_enable = False
        msg.left_stick_x = 0.0
        msg.left_stick_y = 0.0
        msg.right_stick_x = 0.0
        msg.right_stick_y = 0.0
        msg.left_grip_value = 1.0 if arm == "left" and self.teleop_enable else 0.0
        msg.left_trigger_value = 0.0
        return msg

    def _tick(self):
        elapsed = time.monotonic() - self.start_time
        active = self.duration_sec <= 0.0 or elapsed <= self.duration_sec
        offset = self._offset(elapsed) if active else (0.0, 0.0, 0.0)

        if self.arm in ("left", "both"):
            self.left_pub.publish(self._make_msg("left", self.left_base, offset, active))
        if self.arm in ("right", "both"):
            self.right_pub.publish(self._make_msg("right", self.right_base, offset, active))

        if not active:
            self.done = True
            self.timer.cancel()


def main(args=None):
    rclpy.init(args=args)
    node = DualDebugHandGenerator()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
