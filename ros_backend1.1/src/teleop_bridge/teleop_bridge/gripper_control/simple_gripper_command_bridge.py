import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from teleop_bridge_msgs.msg import TargetTwistStates


class TargetTwistToGripperCmd(Node):
    def __init__(self):
        super().__init__("simple_gripper_command_bridge")

        self.declare_parameter("input_topic", "/target_twist_states")
        self.declare_parameter("output_topic", "/hande_position_controller/commands")
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("controlled_joint_names", [""])
        self.declare_parameter("publish_rate_hz", 60.0)
        self.declare_parameter("speed_m_per_s", 0.03)
        self.declare_parameter("min_pos", 0.0)
        self.declare_parameter("max_pos", 0.025)
        self.declare_parameter("initial_pos", 0.025)
        self.declare_parameter("command_mode", "position")
        self.declare_parameter("effort_kp", 900.0)
        self.declare_parameter("effort_kd", 1.5)
        self.declare_parameter("max_effort", 35.0)
        self.declare_parameter("limit_stop_margin_m", 0.003)
        self.declare_parameter("joint_count", 1)
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
        self._min_pos = float(self.get_parameter("min_pos").value)
        self._max_pos = float(self.get_parameter("max_pos").value)
        self._target_pos = float(self.get_parameter("initial_pos").value)
        self._command_mode = str(self.get_parameter("command_mode").value).strip().lower()
        self._effort_kp = max(0.0, float(self.get_parameter("effort_kp").value))
        self._effort_kd = max(0.0, float(self.get_parameter("effort_kd").value))
        self._max_effort = max(0.0, float(self.get_parameter("max_effort").value))
        self._limit_stop_margin = max(0.0, float(self.get_parameter("limit_stop_margin_m").value))
        self._joint_count = max(1, int(self.get_parameter("joint_count").value))
        self._second_joint_negate = bool(self.get_parameter("second_joint_negate").value)
        self._stale_timeout_sec = max(0.05, float(self.get_parameter("stale_timeout_sec").value))
        self._require_tracked = bool(self.get_parameter("require_tracked").value)

        if self._command_mode not in ("position", "velocity", "effort_position"):
            self.get_logger().warn(f"Unsupported command_mode={self._command_mode!r}; using 'position'.")
            self._command_mode = "position"

        if self._min_pos > self._max_pos:
            self._min_pos, self._max_pos = self._max_pos, self._min_pos
        self._target_pos = max(self._min_pos, min(self._max_pos, self._target_pos))

        self._pub = self.create_publisher(Float64MultiArray, self._output_topic, 20)
        self._sub = self.create_subscription(TargetTwistStates, self._input_topic, self._on_input, 20)
        self._joint_state_sub = self.create_subscription(
            JointState, self._joint_states_topic, self._on_joint_states, 20
        )
        self.create_timer(1.0 / self._publish_rate_hz, self._tick)

        self._hold_cmd = 0
        self._tracked = False
        self._reset_enable = False
        self._latest_joint_positions = {}
        self._latest_joint_velocities = {}
        self._last_rx_time = 0.0
        self._last_tick_time = time.monotonic()

        self._rx_count = 0
        self._rx_window_start = time.monotonic()
        self._last_log_time = time.monotonic()

        self.get_logger().info(
            f"TargetTwist->Gripper bridge started: {self._input_topic} -> {self._output_topic}, "
            f"range=[{self._min_pos:.4f},{self._max_pos:.4f}], speed={self._speed:.4f}, "
            f"command_mode={self._command_mode}, joint_count={self._joint_count}, second_joint_negate={self._second_joint_negate}, "
            f"effort_kp={self._effort_kp:.1f}, effort_kd={self._effort_kd:.1f}, max_effort={self._max_effort:.1f}, "
            f"limit_stop_margin={self._limit_stop_margin:.4f}, controlled_joint_names={self._controlled_joint_names}, "
            f"publish_rate_hz={self._publish_rate_hz:.1f}, "
            f"stale_timeout_sec={self._stale_timeout_sec:.3f}, "
            f"require_tracked={self._require_tracked}"
        )

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
        n = len(msg.name)
        for i in range(n):
            name = str(msg.name[i])
            if name not in self._controlled_joint_names:
                continue
            if i < len(msg.position):
                self._latest_joint_positions[name] = float(msg.position[i])
            if i < len(msg.velocity):
                self._latest_joint_velocities[name] = float(msg.velocity[i])

    def _velocity_limit_guard(self, cmd: int) -> int:
        if self._command_mode != "velocity" or cmd == 0 or not self._controlled_joint_names:
            return cmd
        positions = [
            self._latest_joint_positions[name]
            for name in self._controlled_joint_names
            if name in self._latest_joint_positions
        ]
        if len(positions) != len(self._controlled_joint_names):
            return cmd
        # All controlled fingers share the same command.  Stop only when the
        # slowest finger has reached the relevant limit; otherwise one early
        # finger can stop the other and leave the gripper visibly off-center.
        if cmd > 0 and max(positions) <= self._min_pos + self._limit_stop_margin:
            return 0
        if cmd < 0 and min(positions) >= self._max_pos - self._limit_stop_margin:
            return 0
        return cmd

    def _tick(self):
        now = time.monotonic()
        dt = max(0.0, now - self._last_tick_time)
        self._last_tick_time = now

        stale = (now - self._last_rx_time) > self._stale_timeout_sec
        tracked_ok = (not self._require_tracked) or self._tracked
        effective_hold_cmd = self._hold_cmd if (not stale and tracked_ok and (not self._reset_enable)) else 0
        effective_hold_cmd = self._velocity_limit_guard(effective_hold_cmd)

        out = Float64MultiArray()
        if self._reset_enable and self._command_mode == "effort_position":
            self._target_pos = self._max_pos

        if self._command_mode == "velocity":
            # +1 closes (towards min), -1 opens (towards max).
            base = -self._speed if effective_hold_cmd > 0 else self._speed if effective_hold_cmd < 0 else 0.0
            out.data = [base] * self._joint_count
        elif self._command_mode == "effort_position":
            self._integrate_target(effective_hold_cmd, dt)
            out.data = self._effort_position_outputs()
            base = out.data[0] if out.data else 0.0
        else:
            if effective_hold_cmd != 0 and self._speed > 0.0:
                self._integrate_target(effective_hold_cmd, dt)
            base = float(self._target_pos)
            out.data = [base] * self._joint_count
        if self._joint_count >= 2 and self._second_joint_negate and len(out.data) >= 2:
            out.data[1] = -base
        self._pub.publish(out)

        if now - self._last_log_time > 2.0:
            dt_rx = max(now - self._rx_window_start, 1e-6)
            rx_hz = self._rx_count / dt_rx
            self.get_logger().info(
                f"RX {rx_hz:.1f} Hz, stale={stale}, tracked={self._tracked}, reset={self._reset_enable}, hold_cmd={effective_hold_cmd}, "
                f"target_pos={self._target_pos:.4f}, output={out.data}"
            )
            self._rx_count = 0
            self._rx_window_start = now
            self._last_log_time = now

    def _integrate_target(self, hold_cmd: int, dt: float):
        if hold_cmd == 0 or self._speed <= 0.0:
            return
        # +1 closes (towards min), -1 opens (towards max).
        direction = -1.0 if hold_cmd > 0 else 1.0
        self._target_pos += direction * self._speed * dt
        self._target_pos = max(self._min_pos, min(self._max_pos, self._target_pos))

    def _effort_position_outputs(self):
        outputs = []
        for i in range(self._joint_count):
            name = self._controlled_joint_names[i] if i < len(self._controlled_joint_names) else ""
            if not name or name not in self._latest_joint_positions:
                outputs.append(0.0)
                continue
            pos = self._latest_joint_positions[name]
            vel = self._latest_joint_velocities.get(name, 0.0)
            effort = self._effort_kp * (self._target_pos - pos) - self._effort_kd * vel
            effort = max(-self._max_effort, min(self._max_effort, effort))
            outputs.append(float(effort))
        return outputs


def main(args=None):
    rclpy.init(args=args)
    node = TargetTwistToGripperCmd()
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
