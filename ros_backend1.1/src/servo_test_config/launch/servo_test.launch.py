import os
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import xacro


def load_yaml(package_name: str, relative_path: str):
    pkg_share = get_package_share_directory(package_name)
    path = os.path.join(pkg_share, relative_path)
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _launch_setup(context, *args, **kwargs):
    # Resolve launch arguments (LaunchConfiguration -> str)
    ur_type_value = LaunchConfiguration("ur_type").perform(context)
    initial_positions_file_value = LaunchConfiguration("initial_positions_file").perform(context)

    # UR control (real or fake hardware)
    ur_control = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("ur_robot_driver"), "launch", "ur_control.launch.py")
        ),
        launch_arguments={
            "ur_type": LaunchConfiguration("ur_type"),
            "robot_ip": LaunchConfiguration("robot_ip"),
            "use_fake_hardware": LaunchConfiguration("use_fake_hardware"),
            "headless_mode": LaunchConfiguration("headless_mode"),
            "initial_positions_file": LaunchConfiguration("initial_positions_file"),
            "description_package": "ur_hande_description",
            "description_file": "ur_hande.urdf.xacro",
            "launch_rviz": "false",
            "initial_joint_controller": "joint_trajectory_controller",
        }.items(),
    )

    # Robot description (xacro needs raw strings)
    urdf_xacro = os.path.join(
        get_package_share_directory("ur_hande_description"), "urdf", "ur_hande.urdf.xacro"
    )
    robot_description_config = xacro.process_file(
        urdf_xacro,
        mappings={"name": "ur", "ur_type": ur_type_value, "initial_positions_file": initial_positions_file_value},
    )
    robot_description = {"robot_description": robot_description_config.toxml()}

    # Semantic (SRDF)
    srdf_xacro = os.path.join(get_package_share_directory("ur_moveit_config"), "srdf", "ur.srdf.xacro")
    semantic_config = xacro.process_file(srdf_xacro, mappings={"name": "ur", "prefix": ""})
    robot_description_semantic = {"robot_description_semantic": semantic_config.toxml()}

    # Kinematics (ROS2 param-file format)
    kin = load_yaml("ur_moveit_config", "config/kinematics.yaml")
    robot_kin = kin.get("/**", {}).get("ros__parameters", {}).get("robot_description_kinematics", {})
    robot_description_kinematics = {"robot_description_kinematics": robot_kin}

    # Servo parameters
    servo_yaml = load_yaml("servo_test_config", "config/servo.yaml")
    servo_params = {"moveit_servo": servo_yaml}

    # Nodes
    servo_node = Node(
        package="moveit_servo",
        executable="servo_node_main",
        name="servo_node",
        output="screen",
        parameters=[
            servo_params,
            robot_description,
            robot_description_semantic,
            robot_description_kinematics,
        ],
    )

    return [ur_control, servo_node]


def generate_launch_description():
    default_initial_positions = os.path.join(
        get_package_share_directory("servo_test_config"), "config", "initial_positions.yaml"
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("ur_type", default_value="ur5e"),
            DeclareLaunchArgument("robot_ip", default_value="127.0.0.1"),
            DeclareLaunchArgument("use_fake_hardware", default_value="true"),
            DeclareLaunchArgument("headless_mode", default_value="true"),
            DeclareLaunchArgument("initial_positions_file", default_value=default_initial_positions),
            OpaqueFunction(function=_launch_setup),
        ]
    )
