#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class JointStatesFilter(Node):
    def __init__(self):
        super().__init__("joint_states_filter")
        self._source_topic = self.declare_parameter("source_topic", "/joint_states").value
        self._output_topic = self.declare_parameter("output_topic", "/joint_states_servo").value
        self._drop_suffixes = tuple(
            self.declare_parameter("drop_suffixes", ["_mimic"]).value
        )
        self._keep_prefixes = tuple(
            prefix
            for prefix in self.declare_parameter("keep_prefixes", [""]).value
            if str(prefix)
        )

        self._publisher = self.create_publisher(JointState, self._output_topic, 20)
        self._subscription = self.create_subscription(
            JointState, self._source_topic, self._on_joint_states, 20
        )

        self.get_logger().info(
            f"Filtering {self._source_topic} -> {self._output_topic}, "
            f"keep_prefixes={self._keep_prefixes or 'ALL'}, dropping suffixes={self._drop_suffixes}"
        )

    def _on_joint_states(self, msg: JointState):
        keep_indices = [
            idx
            for idx, name in enumerate(msg.name)
            if (not self._keep_prefixes or any(name.startswith(prefix) for prefix in self._keep_prefixes))
            and not any(name.endswith(suffix) for suffix in self._drop_suffixes)
        ]

        # Multiple controller managers publish on /joint_states. A message for the
        # other arm can legitimately contain no joints for this filter. Publishing
        # an empty JointState would make Servo alternate between valid and empty
        # robot states, which corrupts collision checking and velocity scaling.
        if not keep_indices:
            return

        if len(keep_indices) == len(msg.name):
            self._publisher.publish(msg)
            return

        filtered = JointState()
        filtered.header = msg.header
        filtered.name = [msg.name[idx] for idx in keep_indices]
        filtered.position = [msg.position[idx] for idx in keep_indices] if msg.position else []
        filtered.velocity = [msg.velocity[idx] for idx in keep_indices] if msg.velocity else []
        filtered.effort = [msg.effort[idx] for idx in keep_indices] if msg.effort else []
        self._publisher.publish(filtered)


def main():
    rclpy.init()
    node = JointStatesFilter()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
