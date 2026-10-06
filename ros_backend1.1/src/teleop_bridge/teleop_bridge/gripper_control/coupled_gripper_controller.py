import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from teleop_bridge_msgs.msg import TargetTwistStates


class CoupledHandeGripperController(Node):
    """Single-aperture Hand-E gripper controller for Gazebo.

    The simulated Hand-E has two independent prismatic finger joints, while the
    real gripper behaves like one coupled mechanism.  This node keeps one shared
    scalar aperture target and publishes the same target to both finger joints.

    When contact blocks one finger, the shared target is clamped near the most
    open measured finger instead of continuing to drive one rail deep into the
    object.  That makes both fingers behave like one coupled gripper and avoids
    the asymmetric rail fighting seen with fully independent joint targets.
    """

    def __init__(self):
        super().__init__("coupled_gripper_controller")

        self.declare_parameter("input_topic", "/target_twist_states")
        self.declare_parameter("output_topic", "/hande_position_controller/commands")
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("controlled_joint_names", [""])
        self.declare_parameter("publish_rate_hz", 60.0)
        self.declare_parameter("speed_m_per_s", 0.03)
        self.declare_parameter("close_speed_m_per_s", 0.0)
        self.declare_parameter("open_speed_m_per_s", 0.0)
        self.declare_parameter("min_pos", 0.0)
        self.declare_parameter("max_pos", 0.025)
        self.declare_parameter("initial_pos", 0.025)
        self.declare_parameter("initialize_from_joint_state", True)
        self.declare_parameter("squeeze_margin_m", 0.003)
        self.declare_parameter("joint_count", 2)
        self.declare_parameter("second_joint_negate", False)
        self.declare_parameter("stale_timeout_sec", 0.25)
        self.declare_parameter("require_tracked", True)

        self._input_topic = str(self.get_parameter("input_topic").value)
        self._output_topic = str(self.get_parameter("output_topic").value)
        self._joint_states_topic = str(self.get_parameter("joint_states_topic").value)
        self._controlled_joint_names = [
            str(v) for v in self.get_parameter("controlled_joint_names").value if str(v).strip()
        ]
        self._publish_rate_hz = max(1.0, float(self.get_parameter("publish_rate_hz").value))
        self._speed = max(0.0, float(self.get_parameter("speed_m_per_s").value))
        close_speed = max(0.0, float(self.get_parameter("close_speed_m_per_s").value))
        open_speed = max(0.0, float(self.get_parameter("open_speed_m_per_s").value))
        self._close_speed = close_speed if close_speed > 0.0 else self._speed
        self._open_speed = open_speed if open_speed > 0.0 else self._speed
        self._min_pos = float(self.get_parameter("min_pos").value)
        self._max_pos = float(self.get_parameter("max_pos").value)
        self._target_pos = float(self.get_parameter("initial_pos").value)
        self._initialize_from_joint_state = bool(self.get_parameter("initialize_from_joint_state").value)
        self._squeeze_margin = max(0.0, float(self.get_parameter("squeeze_margin_m").value))
        self._joint_count = max(1, int(self.get_parameter("joint_count").value))
        self._second_joint_negate = bool(self.get_parameter("second_joint_negate").value)
        self._stale_timeout_sec = max(0.05, float(self.get_parameter("stale_timeout_sec").value))
        self._require_tracked = bool(self.get_parameter("require_tracked").value)

        if self._min_pos > self._max_pos:
            self._min_pos, self._max_pos = self._max_pos, self._min_pos
        self._target_pos = self._clamp(self._target_pos)

        self._pub = self.create_publisher(Float64MultiArray, self._output_topic, 20)
        self.create_subscription(TargetTwistStates, self._input_topic, self._on_input, 20)
        self.create_subscription(JointState, self._joint_states_topic, self._on_joint_states, 20)
        self.create_timer(1.0 / self._publish_rate_hz, self._tick)

        self._hold_cmd = 0
        self._tracked = False
        self._reset_enable = False
        self._latest_joint_positions = {}
        self._target_initialized = not self._initialize_from_joint_state
        self._last_rx_time = 0.0
        self._last_tick_time = time.monotonic()
        self._last_log_time = time.monotonic()
        self._rx_count = 0
        self._rx_window_start = time.monotonic()

        self.get_logger().info(
            f"Coupled Hand-E gripper controller started: {self._input_topic} -> {self._output_topic}, "
            f"joints={self._controlled_joint_names}, range=[{self._min_pos:.4f},{self._max_pos:.4f}], "
            f"speed={self._speed:.4f}, close_speed={self._close_speed:.4f}, open_speed={self._open_speed:.4f}, "
            f"squeeze_margin={self._squeeze_margin:.4f}, "
            f"initialize_from_joint_state={self._initialize_from_joint_state}, "
            f"joint_count={self._joint_count}, second_joint_negate={self._second_joint_negate}, "
            f"require_tracked={self._require_tracked}"
        )

    def _clamp(self, value: float) -> float:
        return max(self._min_pos, min(self._max_pos, float(value)))

    def _on_input(self, msg: TargetTwistStates):
        if msg.gripper_cmd > 0:
            self._hold_cmd = 1
        elif msg.gripper_cmd < 0:
            self._hold_cmd = -1
        else:
            self._hold_cmd = 0

        self._tracked = bool(msg.tracked)
        self._reset_enable = bool(msg.reset_enable)
        self._last_rx_time = time.monotonic()
        self._rx_count += 1

    def _on_joint_states(self, msg: JointState):
        if not self._controlled_joint_names:
            return
        n = min(len(msg.name), len(msg.position))
        for i in range(n):
            name = str(msg.name[i])
            if name in self._controlled_joint_names:
                self._latest_joint_positions[name] = float(msg.position[i])

    def _current_finger_positions(self):
        vals = []
        for name in self._controlled_joint_names[: self._joint_count]:
            if name in self._latest_joint_positions:
                vals.append(self._latest_joint_positions[name])
        return vals

    def _integrate_shared_target(self, hold_cmd: int, dt: float):
        positions = self._current_finger_positions()
        if not self._target_initialized and positions:
            self._target_pos = self._contact_aperture_target(positions)
            self._target_initialized = True

        if self._reset_enable:
            self._target_pos = self._max_pos
            self._target_initialized = True
            return

        if hold_cmd > 0:
            self._target_pos -= self._close_speed * dt
        elif hold_cmd < 0:
            self._target_pos += self._open_speed * dt
        self._target_pos = self._clamp(self._target_pos)

        if hold_cmd <= 0 or not positions:
            return

        # Closing in this gripper range moves toward min_pos.  If one finger is
        # blocked by contact, it becomes the most-open measured finger.  Do not
        # let the shared target run far beyond that blocked finger; instead make
        # the other finger back off so the pair acts like one aperture.
        most_open = max(positions)
        minimum_target_allowed_by_contact = self._clamp(most_open - self._squeeze_margin)
        if self._target_pos < minimum_target_allowed_by_contact:
            self._target_pos = minimum_target_allowed_by_contact

    def _contact_aperture_target(self, positions):
        if not positions:
            return self._target_pos
        if len(positions) == 1:
            return self._clamp(positions[0])
        spread = max(positions) - min(positions)
        if spread > self._squeeze_margin:
            return self._clamp(max(positions) - self._squeeze_margin)
        return self._clamp(sum(positions) / len(positions))

    def _publish_position(self):
        base = float(self._target_pos)
        msg = Float64MultiArray()
        msg.data = [base] * self._joint_count
        if self._joint_count >= 2 and self._second_joint_negate:
            msg.data[1] = -base
        self._pub.publish(msg)
        return msg.data

    def _tick(self):
        now = time.monotonic()
        dt = max(0.0, now - self._last_tick_time)
        self._last_tick_time = now

        stale = (now - self._last_rx_time) > self._stale_timeout_sec
        tracked_ok = (not self._require_tracked) or self._tracked
        effective_hold_cmd = self._hold_cmd if (not stale and tracked_ok and not self._reset_enable) else 0

        self._integrate_shared_target(effective_hold_cmd, dt)
        output = self._publish_position()

        if now - self._last_log_time > 2.0:
            dt_rx = max(now - self._rx_window_start, 1e-6)
            rx_hz = self._rx_count / dt_rx
            self.get_logger().info(
                f"RX {rx_hz:.1f} Hz, stale={stale}, tracked={self._tracked}, "
                f"reset={self._reset_enable}, hold_cmd={effective_hold_cmd}, "
                f"target_pos={self._target_pos:.4f}, fingers={self._current_finger_positions()}, output={output}"
            )
            self._rx_count = 0
            self._rx_window_start = now
            self._last_log_time = now


def main(args=None):
    rclpy.init(args=args)
    node = CoupledHandeGripperController()
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
