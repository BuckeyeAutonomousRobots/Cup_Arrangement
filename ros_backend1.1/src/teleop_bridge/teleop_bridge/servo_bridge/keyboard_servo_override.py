import select
import sys
import termios
import time
import tty

import numpy as np
import rclpy
from geometry_msgs.msg import TwistStamped
from rcl_interfaces.msg import SetParametersResult
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_srvs.srv import Trigger


HELP_TEXT = """
Keyboard Servo Controller
-------------------------
Translation:
  W/S  forward/back   (+/- X)
  A/D  left/right     (+/- Y)
  Q/E  up/down        (+/- Z)

Rotation:
  U/J  roll  +/- angular X
  I/K  pitch +/- angular Y
  O/L  yaw   +/- angular Z

Other:
  Space  stop immediately
  X      quit
  Ctrl-C quit

This node publishes directly to /servo_node/delta_twist_cmds.
Run it only when you intentionally want keyboard control.
"""


class KeyboardServoCmd(Node):
    def __init__(self):
        super().__init__("keyboard_servo_override")

        self.declare_parameter("output_topic", "/servo_node/delta_twist_cmds")
        self.declare_parameter("frame_id", "base_link")
        self.declare_parameter("publish_rate_hz", 60.0)
        self.declare_parameter("key_timeout_sec", 0.25)
        self.declare_parameter("linear_speed_xyz", [0.15, 0.15, 0.15])
        self.declare_parameter("linear_sign_xyz", [1.0, 1.0, 1.0])
        self.declare_parameter("angular_speed_xyz", [0.50, 0.50, 0.50])
        self.declare_parameter("angular_sign_xyz", [1.0, 1.0, 1.0])
        self.declare_parameter("auto_start_servo", True)
        self.declare_parameter("start_servo_service", "/servo_node/start_servo")

        self.output_topic = str(self.get_parameter("output_topic").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        publish_rate_hz = max(1.0, float(self.get_parameter("publish_rate_hz").value))
        self.key_timeout_sec = max(0.05, float(self.get_parameter("key_timeout_sec").value))
        self.linear_speed_xyz = self._parse_vec3(
            self.get_parameter("linear_speed_xyz").value,
            np.array([0.15, 0.15, 0.15], dtype=float),
            "linear_speed_xyz",
        )
        self.linear_sign_xyz = self._parse_vec3(
            self.get_parameter("linear_sign_xyz").value,
            np.array([1.0, 1.0, 1.0], dtype=float),
            "linear_sign_xyz",
        )
        self.angular_speed_xyz = self._parse_vec3(
            self.get_parameter("angular_speed_xyz").value,
            np.array([0.50, 0.50, 0.50], dtype=float),
            "angular_speed_xyz",
        )
        self.angular_sign_xyz = self._parse_vec3(
            self.get_parameter("angular_sign_xyz").value,
            np.array([1.0, 1.0, 1.0], dtype=float),
            "angular_sign_xyz",
        )
        self.auto_start_servo = bool(self.get_parameter("auto_start_servo").value)
        self.start_servo_service = str(self.get_parameter("start_servo_service").value)

        self.pub = self.create_publisher(TwistStamped, self.output_topic, 20)
        self.create_timer(1.0 / publish_rate_hz, self._publish_loop)
        self.create_timer(0.01, self._poll_keyboard)
        self.add_on_set_parameters_callback(self._on_params)

        self._linear_axis = np.zeros(3, dtype=float)
        self._angular_axis = np.zeros(3, dtype=float)
        self._linear_expiry = np.zeros(3, dtype=float)
        self._angular_expiry = np.zeros(3, dtype=float)
        self._quit = False
        self._old_termios = None
        self._servo_started = False
        self._start_future = None
        self._last_start_log_time = 0.0
        self._start_client = None

        if self.auto_start_servo:
            self._start_client = self.create_client(Trigger, self.start_servo_service)
            self.create_timer(1.0, self._maybe_start_servo)

        self._setup_terminal()
        print(HELP_TEXT, flush=True)
        self.get_logger().info(
            f"Keyboard servo started: output_topic={self.output_topic}, frame_id={self.frame_id}, "
            f"linear_speed_xyz={self.linear_speed_xyz.tolist()}, linear_sign_xyz={self.linear_sign_xyz.tolist()}, "
            f"angular_speed_xyz={self.angular_speed_xyz.tolist()}, angular_sign_xyz={self.angular_sign_xyz.tolist()}, "
            f"key_timeout_sec={self.key_timeout_sec:.3f}"
        )

    @staticmethod
    def _parse_vec3(value, default: np.ndarray, name: str) -> np.ndarray:
        try:
            vec = np.array(value, dtype=float)
        except (TypeError, ValueError):
            return default
        if vec.shape != (3,):
            return default
        return vec

    @staticmethod
    def _require_vec3(value, name: str) -> np.ndarray:
        try:
            vec = np.array(value, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a numeric array with 3 values") from exc
        if vec.shape != (3,):
            raise ValueError(f"{name} must be a numeric array with 3 values")
        return vec

    def _on_params(self, params):
        restart_only = {
            "output_topic",
            "publish_rate_hz",
            "auto_start_servo",
            "start_servo_service",
        }
        frame_id = self.frame_id
        key_timeout_sec = self.key_timeout_sec
        linear_speed_xyz = self.linear_speed_xyz.copy()
        linear_sign_xyz = self.linear_sign_xyz.copy()
        angular_speed_xyz = self.angular_speed_xyz.copy()
        angular_sign_xyz = self.angular_sign_xyz.copy()

        try:
            for param in params:
                if param.name in restart_only:
                    return SetParametersResult(
                        successful=False,
                        reason=f"{param.name} requires restarting keyboard_servo_override",
                    )
                if param.name == "frame_id":
                    frame_id = str(param.value)
                elif param.name == "key_timeout_sec":
                    key_timeout_sec = max(0.05, float(param.value))
                elif param.name == "linear_speed_xyz":
                    linear_speed_xyz = self._require_vec3(param.value, param.name)
                elif param.name == "linear_sign_xyz":
                    linear_sign_xyz = self._require_vec3(param.value, param.name)
                elif param.name == "angular_speed_xyz":
                    angular_speed_xyz = self._require_vec3(param.value, param.name)
                elif param.name == "angular_sign_xyz":
                    angular_sign_xyz = self._require_vec3(param.value, param.name)
        except ValueError as exc:
            return SetParametersResult(successful=False, reason=str(exc))

        self.frame_id = frame_id
        self.key_timeout_sec = key_timeout_sec
        self.linear_speed_xyz = linear_speed_xyz
        self.linear_sign_xyz = linear_sign_xyz
        self.angular_speed_xyz = angular_speed_xyz
        self.angular_sign_xyz = angular_sign_xyz
        self.get_logger().info(
            f"Keyboard tuning updated: frame_id={self.frame_id}, key_timeout_sec={self.key_timeout_sec:.3f}, "
            f"linear_speed_xyz={self.linear_speed_xyz.tolist()}, linear_sign_xyz={self.linear_sign_xyz.tolist()}, "
            f"angular_speed_xyz={self.angular_speed_xyz.tolist()}, angular_sign_xyz={self.angular_sign_xyz.tolist()}"
        )
        return SetParametersResult(successful=True)

    @property
    def quit_requested(self) -> bool:
        return self._quit

    def _setup_terminal(self):
        if not sys.stdin.isatty():
            self.get_logger().error("keyboard_servo_override requires an interactive TTY. Use docker exec -it.")
            self._quit = True
            return

        self._old_termios = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

    def _restore_terminal(self):
        if self._old_termios is not None and sys.stdin.isatty():
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_termios)
            self._old_termios = None

    def _maybe_start_servo(self):
        if self._servo_started or self._start_client is None:
            return
        if self._start_future is not None and not self._start_future.done():
            return

        now = time.monotonic()
        if not self._start_client.service_is_ready():
            if now - self._last_start_log_time > 2.0:
                self.get_logger().info(f"Waiting for {self.start_servo_service} before keyboard control.")
                self._last_start_log_time = now
            return

        self._start_future = self._start_client.call_async(Trigger.Request())
        self._start_future.add_done_callback(self._on_start_servo_done)

    def _on_start_servo_done(self, future):
        try:
            resp = future.result()
        except Exception as exc:
            self.get_logger().warn(f"Servo start request failed: {exc}")
            return

        message = resp.message.strip() if resp.message else ""
        already_running = "already" in message.lower() and "run" in message.lower()
        if resp.success or already_running:
            self._servo_started = True
            self.get_logger().info(f"Servo start acknowledged: {message or 'success'}")
        else:
            self.get_logger().warn(f"Servo start rejected: {message or 'success=false'}")

    def _poll_keyboard(self):
        if self._quit or not sys.stdin.isatty():
            return

        while select.select([sys.stdin], [], [], 0.0)[0]:
            ch = sys.stdin.read(1)
            if ch in ("\x03", "\x04", "\x1b", "x", "X"):
                self._quit = True
                self._linear_axis[:] = 0.0
                self._angular_axis[:] = 0.0
                return

            now = time.monotonic()
            key = ch.lower()
            if key == " ":
                self._linear_axis[:] = 0.0
                self._angular_axis[:] = 0.0
                self._linear_expiry[:] = 0.0
                self._angular_expiry[:] = 0.0
                continue

            self._apply_key(key, now)

    def _apply_key(self, key: str, now: float):
        linear_keys = {
            "w": (0, 1.0),
            "s": (0, -1.0),
            "a": (1, 1.0),
            "d": (1, -1.0),
            "q": (2, 1.0),
            "e": (2, -1.0),
        }
        angular_keys = {
            "u": (0, 1.0),
            "j": (0, -1.0),
            "i": (1, 1.0),
            "k": (1, -1.0),
            "o": (2, 1.0),
            "l": (2, -1.0),
        }

        if key in linear_keys:
            axis, value = linear_keys[key]
            self._linear_axis[axis] = value
            self._linear_expiry[axis] = now + self.key_timeout_sec
        elif key in angular_keys:
            axis, value = angular_keys[key]
            self._angular_axis[axis] = value
            self._angular_expiry[axis] = now + self.key_timeout_sec

    def _publish_loop(self):
        now = time.monotonic()
        for i in range(3):
            if self._linear_expiry[i] <= now:
                self._linear_axis[i] = 0.0
            if self._angular_expiry[i] <= now:
                self._angular_axis[i] = 0.0

        linear = self._linear_axis * self.linear_speed_xyz * self.linear_sign_xyz
        angular = self._angular_axis * self.angular_speed_xyz * self.angular_sign_xyz
        self._publish_twist(linear, angular)

    def _publish_twist(self, linear: np.ndarray, angular: np.ndarray):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.twist.linear.x = float(linear[0])
        msg.twist.linear.y = float(linear[1])
        msg.twist.linear.z = float(linear[2])
        msg.twist.angular.x = float(angular[0])
        msg.twist.angular.y = float(angular[1])
        msg.twist.angular.z = float(angular[2])
        self.pub.publish(msg)

    def destroy_node(self):
        self._publish_twist(np.zeros(3, dtype=float), np.zeros(3, dtype=float))
        self._restore_terminal()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = KeyboardServoCmd()
    try:
        while rclpy.ok() and not node.quit_requested:
            rclpy.spin_once(node, timeout_sec=0.1)
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
