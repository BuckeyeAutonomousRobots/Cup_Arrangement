import math
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Float64MultiArray, Int8
from teleop_bridge_msgs.msg import TargetTwistStates


class ServoResponseSampler(Node):
    """Summarize mapper target speed, Servo output, status, and collision scale."""

    def __init__(self):
        super().__init__("servo_response_sampler")
        self.declare_parameter("duration_sec", 10.0)
        self.declare_parameter("arms", "left,right")

        self.duration_sec = max(0.5, float(self.get_parameter("duration_sec").value))
        arms_raw = str(self.get_parameter("arms").value)
        self.arms = []
        for arm in [a.strip().lower() for a in arms_raw.split(",")]:
            if arm in ("left", "right") and arm not in self.arms:
                self.arms.append(arm)
        if not self.arms:
            self.arms = ["left", "right"]

        self.data = {
            arm: {"target": [], "cmd": [], "status": [], "scale": []}
            for arm in self.arms
        }

        for arm in self.arms:
            ns = f"/{arm}_arm"
            self.create_subscription(
                TargetTwistStates,
                f"{ns}/target_twist_states",
                lambda msg, arm_name=arm: self._on_target(arm_name, msg),
                20,
            )
            self.create_subscription(
                Float64MultiArray,
                f"/{arm}_joint_group_velocity_controller/commands",
                lambda msg, arm_name=arm: self._on_cmd(arm_name, msg),
                20,
            )
            self.create_subscription(
                Int8,
                f"{ns}/servo_node/status",
                lambda msg, arm_name=arm: self.data[arm_name]["status"].append(int(msg.data)),
                20,
            )
            self.create_subscription(
                Float64,
                f"{ns}/servo_node/collision_velocity_scale",
                lambda msg, arm_name=arm: self.data[arm_name]["scale"].append(float(msg.data)),
                20,
            )

        self.start_time = time.monotonic()
        self.done = False
        self._timer = self.create_timer(0.1, self._tick)
        self.get_logger().info(
            f"Sampling Servo response for {self.duration_sec:.1f}s on arms={self.arms}"
        )

    def _on_target(self, arm: str, msg: TargetTwistStates):
        lin = math.sqrt(
            msg.twist.linear.x * msg.twist.linear.x
            + msg.twist.linear.y * msg.twist.linear.y
            + msg.twist.linear.z * msg.twist.linear.z
        )
        ang = math.sqrt(
            msg.twist.angular.x * msg.twist.angular.x
            + msg.twist.angular.y * msg.twist.angular.y
            + msg.twist.angular.z * msg.twist.angular.z
        )
        self.data[arm]["target"].append((lin, ang, bool(msg.tracked)))

    def _on_cmd(self, arm: str, msg: Float64MultiArray):
        norm = math.sqrt(sum(v * v for v in msg.data)) if msg.data else 0.0
        max_joint = max((abs(v) for v in msg.data), default=0.0)
        self.data[arm]["cmd"].append((norm, max_joint))

    def _tick(self):
        if time.monotonic() - self.start_time < self.duration_sec:
            return
        self._print_summary()
        self.done = True
        self.destroy_timer(self._timer)

    @staticmethod
    def _max(values, default=0.0):
        return max(values) if values else default

    def _print_summary(self):
        print("=== Servo response summary ===")
        for arm in self.arms:
            arm_data = self.data[arm]
            targets = arm_data["target"]
            active_targets = [
                target for target in targets
                if target[2] and (target[0] > 1e-4 or target[1] > 1e-4)
            ]
            cmds = arm_data["cmd"]
            nonzero_cmds = [cmd for cmd in cmds if cmd[0] > 1e-6]
            statuses = arm_data["status"]
            scales = arm_data["scale"]
            status_counts = {status: statuses.count(status) for status in sorted(set(statuses))}

            print(
                f"{arm}: target_msgs={len(targets)} active={len(active_targets)} "
                f"max_target_lin={self._max([t[0] for t in active_targets]):.4f} "
                f"max_target_ang={self._max([t[1] for t in active_targets]):.4f}"
            )
            print(
                f"{arm}: cmd_msgs={len(cmds)} nonzero_cmd={len(nonzero_cmds)} "
                f"max_cmd_norm={self._max([c[0] for c in nonzero_cmds]):.4f} "
                f"max_joint={self._max([c[1] for c in nonzero_cmds]):.4f}"
            )
            print(
                f"{arm}: status_counts={status_counts} "
                f"scale_min={min(scales) if scales else None} "
                f"scale_max={max(scales) if scales else None} "
                f"scale_last={scales[-1] if scales else None}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = ServoResponseSampler()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
