from dataclasses import dataclass

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer, TransformListener


@dataclass
class Point3:
        x: float
        y: float
        z: float


class FakeHandPublisher(Node):
    def __init__(self):
        super().__init__("fake_hand_publisher")

        self.declare_parameter("target_pose_topic", "/servo_node/target_pose")
        self.declare_parameter("servo_twist_topic", "/servo_node/delta_twist_cmds")
        self.declare_parameter("pose_only", True)
        self.declare_parameter("frame_id", "base_link")
        self.declare_parameter("ee_frame", "tool0")
        self.declare_parameter("publish_rate_hz", 60.0)
        self.declare_parameter("segment_duration_s", 4.0)
        self.declare_parameter("loop", True)
        self.declare_parameter("use_current_ee_orientation", True)
        self.declare_parameter("target_qx", 0.0)
        self.declare_parameter("target_qy", 0.0)
        self.declare_parameter("target_qz", 0.0)
        self.declare_parameter("target_qw", 1.0)

        # === Edit these two points to change the motion ===
        self.point_a = Point3(0.35, -0.15, 0.25)
        self.point_b = Point3(0.45, 0.15, 0.35)

        self._target_pose_topic = self.get_parameter("target_pose_topic").get_parameter_value().string_value
        self._pose_only = self.get_parameter("pose_only").get_parameter_value().bool_value
        self._frame_id = self.get_parameter("frame_id").get_parameter_value().string_value
        self._ee_frame = self.get_parameter("ee_frame").get_parameter_value().string_value
        self._publish_rate_hz = self.get_parameter("publish_rate_hz").get_parameter_value().double_value
        self._segment_duration_s = max(0.5, self.get_parameter("segment_duration_s").get_parameter_value().double_value)
        self._loop = self.get_parameter("loop").get_parameter_value().bool_value
        self._use_current_ee_orientation = self.get_parameter("use_current_ee_orientation").get_parameter_value().bool_value
        self._target_qx = self.get_parameter("target_qx").get_parameter_value().double_value
        self._target_qy = self.get_parameter("target_qy").get_parameter_value().double_value
        self._target_qz = self.get_parameter("target_qz").get_parameter_value().double_value
        self._target_qw = self.get_parameter("target_qw").get_parameter_value().double_value

        self._pose_pub = self.create_publisher(PoseStamped, self._target_pose_topic, 10)
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._t0 = self.get_clock().now().nanoseconds / 1e9
        self._timer = self.create_timer(1.0 / max(self._publish_rate_hz, 1.0), self._tick)

        self.get_logger().info(
            f"Publishing PoseStamped only to {self._target_pose_topic}"
        )
        if not self._pose_only:
            self.get_logger().warn(
                "pose_only=false is deprecated for fake_hand_publisher; twist is no longer published from this node. Use debug_hand_generator or hand_pose_mapper instead."
            )
        self.get_logger().info(
            f"A=({self.point_a.x:.3f},{self.point_a.y:.3f},{self.point_a.z:.3f}) B=({self.point_b.x:.3f},{self.point_b.y:.3f},{self.point_b.z:.3f})"
        )

    def _lerp(self, a: float, b: float, t: float) -> float:
        return a + (b - a) * t

    def _tick(self):
        now_s = self.get_clock().now().nanoseconds / 1e9
        elapsed = now_s - self._t0
        phase = elapsed / self._segment_duration_s

        if self._loop:
            # triangle wave 0->1->0
            p = phase % 2.0
            u = p if p <= 1.0 else (2.0 - p)
        else:
            u = min(1.0, max(0.0, phase))

        tx = self._lerp(self.point_a.x, self.point_b.x, u)
        ty = self._lerp(self.point_a.y, self.point_b.y, u)
        tz = self._lerp(self.point_a.z, self.point_b.z, u)

        target = PoseStamped()
        target.header.stamp = self.get_clock().now().to_msg()
        target.header.frame_id = self._frame_id
        target.pose.position.x = float(tx)
        target.pose.position.y = float(ty)
        target.pose.position.z = float(tz)

        orientation_set_from_tf = False
        if self._use_current_ee_orientation:
            try:
                tf = self._tf_buffer.lookup_transform(self._frame_id, self._ee_frame, rclpy.time.Time())
                target.pose.orientation = tf.transform.rotation
                orientation_set_from_tf = True
            except Exception:
                orientation_set_from_tf = False

        if not orientation_set_from_tf:
            target.pose.orientation.x = float(self._target_qx)
            target.pose.orientation.y = float(self._target_qy)
            target.pose.orientation.z = float(self._target_qz)
            target.pose.orientation.w = float(self._target_qw)

        self._pose_pub.publish(target)


def main(args=None):
    rclpy.init(args=args)
    node = FakeHandPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
