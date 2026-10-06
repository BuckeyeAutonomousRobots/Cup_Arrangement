import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import math
import time

class JointSender(Node):
    def __init__(self):
        super().__init__('joint_sender')
        # Unity Subscriber expects this topic based on our previous config
        self.publisher_ = self.create_publisher(JointTrajectory, '/ur5e_joint_trajectory', 10)
        self.timer = self.create_timer(0.1, self.timer_callback) # 10Hz
        self.i = 0
        self.start_time = time.time()

    def timer_callback(self):
        msg = JointTrajectory()

        # Standard UR Joint Names (Must match Unity/URDF)
        msg.joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint"
        ]

        point = JointTrajectoryPoint()

        # Calculate a gentle sine wave for the shoulder pan
        t = time.time() - self.start_time
        pan = 0.5 * math.sin(t)

        # [Pan, Lift, Elbow, Wrist1, Wrist2, Wrist3]
        # Lift -1.57 (Upright), Elbow 1.57 (Bent 90 deg)
        point.positions = [pan, -1.57, 1.57, 0.0, 0.0, 0.0]

        # Velocities are optional but good for smoothness
        point.velocities = [0.0] * 6
        point.accelerations = [0.0] * 6

        # Time from start is critical for trajectory execution
        point.time_from_start = Duration(sec=0, nanosec=100000000) # 0.1s in future

        msg.points = [point]

        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing Trajectory Point: Pan={pan:.2f}')
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    sender = JointSender()
    rclpy.spin(sender)
    sender.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
