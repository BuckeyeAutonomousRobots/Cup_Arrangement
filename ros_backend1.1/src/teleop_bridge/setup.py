from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'teleop_bridge'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='User',
    maintainer_email='user@example.com',
    description='VR Hand Tracking to ROS 2 Bridge',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # Clear data-flow names for new work.
            'hand_pose_mapper = teleop_bridge.mapping.hand_pose_mapper:main',
            'servo_command_bridge = teleop_bridge.servo_bridge.servo_command_bridge:main',
            'keyboard_servo_override = teleop_bridge.servo_bridge.keyboard_servo_override:main',
            'simple_gripper_command_bridge = teleop_bridge.gripper_control.simple_gripper_command_bridge:main',
            'coupled_gripper_controller = teleop_bridge.gripper_control.coupled_gripper_controller:main',
            'reset_manager = teleop_bridge.reset_control.reset_manager:main',
            'task_pose_sync_publisher = teleop_bridge.task_sync.task_pose_sync_publisher:main',
            'runtime_task_manager = teleop_bridge.task_sync.runtime_task_manager:main',
            'rubik_task_controller = teleop_bridge.task_sync.rubik_task_controller:main',
            'contact_haptic_publisher = teleop_bridge.haptic_feedback.contact_haptic_publisher:main',
            'debug_hand_generator = teleop_bridge.test_tools.debug_hand_generator:main',
            # Compatibility aliases used by existing docs/scripts/log grep patterns.
            'test_joints = teleop_bridge.test_tools.test_joints:main',
            'fake_hand_publisher = teleop_bridge.test_tools.fake_hand_publisher:main',
            'dual_debug_hand_generator = teleop_bridge.test_tools.debug_hand_generator:main',
            'servo_response_sampler = teleop_bridge.test_tools.servo_response_sampler:main',
            'received_pose_to_target_twist = teleop_bridge.mapping.hand_pose_mapper:main',
            'target_twist_to_servo_cmd = teleop_bridge.servo_bridge.servo_command_bridge:main',
            'target_twist_to_gripper_cmd = teleop_bridge.gripper_control.simple_gripper_command_bridge:main',
            'coupled_hande_gripper_controller = teleop_bridge.gripper_control.coupled_gripper_controller:main',
            'target_twist_reset_manager = teleop_bridge.reset_control.reset_manager:main',
            'cube_pose_sync_publisher = teleop_bridge.task_sync.task_pose_sync_publisher:main',
            'gazebo_contact_haptic_publisher = teleop_bridge.haptic_feedback.contact_haptic_publisher:main',
            'rubik2x2_mechanism_controller = teleop_bridge.task_sync.rubik_task_controller:main',
            'task_manager = teleop_bridge.task_sync.runtime_task_manager:main',
            'keyboard_servo_cmd = teleop_bridge.servo_bridge.keyboard_servo_override:main',
        ],
    },
)
