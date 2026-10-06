import time

import rclpy
from geometry_msgs.msg import TwistStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_srvs.srv import Trigger
from teleop_bridge_msgs.msg import TargetTwistStates


class TargetTwistToServoCmd(Node):
    def __init__(self):
        super().__init__("servo_command_bridge")

        self.declare_parameter("input_topic", "/target_twist_states")
        self.declare_parameter("output_topic", "/servo_node/delta_twist_cmds")
        self.declare_parameter("default_frame_id", "base_link")
        self.declare_parameter("publish_rate_hz", 60.0)
        self.declare_parameter("stale_timeout_sec", 0.25)
        self.declare_parameter("require_tracked", True)
        self.declare_parameter("auto_start_servo", True)
        self.declare_parameter("start_servo_service", "/servo_node/start_servo")
        self.declare_parameter("start_retry_sec", 1.0)
        self.declare_parameter("start_request_timeout_sec", 5.0)
        self.declare_parameter("start_on_reset_release", True)

        self._input_topic = str(self.get_parameter("input_topic").value)
        self._output_topic = str(self.get_parameter("output_topic").value)
        self._default_frame_id = str(self.get_parameter("default_frame_id").value)
        self._publish_rate_hz = max(1.0, float(self.get_parameter("publish_rate_hz").value))
        self._stale_timeout_sec = max(0.05, float(self.get_parameter("stale_timeout_sec").value))
        self._require_tracked = bool(self.get_parameter("require_tracked").value)
        self._auto_start_servo = bool(self.get_parameter("auto_start_servo").value)
        self._start_servo_service = str(self.get_parameter("start_servo_service").value)
        self._start_retry_sec = max(0.2, float(self.get_parameter("start_retry_sec").value))
        self._start_request_timeout_sec = max(1.0, float(self.get_parameter("start_request_timeout_sec").value))
        self._start_request_time = 0.0
        self._start_on_reset_release = bool(self.get_parameter("start_on_reset_release").value)

        self._pub = self.create_publisher(TwistStamped, self._output_topic, 20)
        self._sub = self.create_subscription(TargetTwistStates, self._input_topic, self._on_input, 20)
        self.create_timer(1.0 / self._publish_rate_hz, self._publish_loop)

        self._last_rx_time = 0.0
        self._latest_msg = TargetTwistStates()
        self._have_msg = False

        self._rx_count = 0
        self._rx_window_start = time.monotonic()
        self._last_log_time = time.monotonic()

        self._servo_started = False
        self._start_future = None
        self._last_start_log_time = 0.0
        self._last_reset_active = False
        self._start_client = None
        if self._auto_start_servo:
            self._start_client = self.create_client(Trigger, self._start_servo_service)
            self.create_timer(self._start_retry_sec, self._maybe_start_servo)

        self.get_logger().info(
            f"TargetTwist->Servo bridge started: {self._input_topic} -> {self._output_topic}, "
            f"publish_rate_hz={self._publish_rate_hz:.1f}, stale_timeout_sec={self._stale_timeout_sec:.3f}, "
            f"require_tracked={self._require_tracked}, auto_start_servo={self._auto_start_servo}, "
            f"start_on_reset_release={self._start_on_reset_release}"
        )

    def _on_input(self, msg: TargetTwistStates):
        self._latest_msg = msg
        self._have_msg = True
        self._last_rx_time = time.monotonic()
        self._rx_count += 1

    def _maybe_start_servo(self):
        if self._servo_started or self._start_client is None:
            return

        self._request_start_servo(force=False, reason="auto-start")

    def _request_start_servo(self, force: bool, reason: str):
        if self._start_client is None:
            return

        if (not force) and self._servo_started:
            return

        now = time.monotonic()
        if self._start_future is not None and not self._start_future.done():
            if now - self._start_request_time < self._start_request_timeout_sec:
                return
            self._start_future.cancel()
            self._start_future = None
            self.get_logger().warn(f"Servo start timed out; retrying {self._start_servo_service}")

        if not self._start_client.service_is_ready():
            if now - self._last_start_log_time > 2.0:
                self.get_logger().info(
                    f"Waiting for {self._start_servo_service} service before start request ({reason})."
                )
                self._last_start_log_time = now
            return

        req = Trigger.Request()
        self._start_request_time = now
        self._start_future = self._start_client.call_async(req)
        self._start_future.add_done_callback(lambda fut: self._on_start_servo_done(fut, reason))

    def _on_start_servo_done(self, future, reason: str):
        if future.cancelled():
            return
        try:
            resp = future.result()
        except Exception as exc:
            self.get_logger().warn(f"Servo start request failed ({reason}): {exc}")
            return

        message = resp.message.strip() if resp.message else ""
        already_running = "already" in message.lower() and "run" in message.lower()

        if resp.success or already_running:
            self._servo_started = True
            msg = message or "success"
            self.get_logger().info(
                f"Servo start acknowledged via {self._start_servo_service} ({reason}): {msg}"
            )
        else:
            msg = message or "service returned success=false"
            self.get_logger().warn(f"Servo start rejected ({reason}): {msg}")

    def _publish_loop(self):
        now = time.monotonic()
        stale = (now - self._last_rx_time) > self._stale_timeout_sec
        tracked_ok = (not self._require_tracked) or bool(self._latest_msg.tracked)
        reset_active = bool(self._latest_msg.reset_enable)
        active = self._have_msg and (not stale) and tracked_ok and (not reset_active)

        if self._start_on_reset_release and self._auto_start_servo:
            if (not reset_active) and self._last_reset_active:
                # Reset sequence issues a /stop_servo call. Re-request start when reset is released
                # so teleop can recover even if reset-manager start retries were missed.
                self._servo_started = False
                self._request_start_servo(force=True, reason="reset-release")
            self._last_reset_active = reset_active

        out = TwistStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        frame_id = self._default_frame_id
        if active and self._latest_msg.header.frame_id:
            frame_id = self._latest_msg.header.frame_id
        out.header.frame_id = frame_id

        if active:
            out.twist.linear.x = float(self._latest_msg.twist.linear.x)
            out.twist.linear.y = float(self._latest_msg.twist.linear.y)
            out.twist.linear.z = float(self._latest_msg.twist.linear.z)
            out.twist.angular.x = float(self._latest_msg.twist.angular.x)
            out.twist.angular.y = float(self._latest_msg.twist.angular.y)
            out.twist.angular.z = float(self._latest_msg.twist.angular.z)

        self._pub.publish(out)

        if now - self._last_log_time > 2.0:
            dt = max(now - self._rx_window_start, 1e-6)
            rx_hz = self._rx_count / dt
            self.get_logger().info(
                f"RX {rx_hz:.1f} Hz, stale={stale}, tracked={bool(self._latest_msg.tracked)}, reset={reset_active}, "
                f"active={active}, twist=lin({out.twist.linear.x:.3f},{out.twist.linear.y:.3f},{out.twist.linear.z:.3f}) "
                f"ang({out.twist.angular.x:.3f},{out.twist.angular.y:.3f},{out.twist.angular.z:.3f})"
            )
            self._rx_count = 0
            self._rx_window_start = now
            self._last_log_time = now


def main(args=None):
    rclpy.init(args=args)
    node = TargetTwistToServoCmd()
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
