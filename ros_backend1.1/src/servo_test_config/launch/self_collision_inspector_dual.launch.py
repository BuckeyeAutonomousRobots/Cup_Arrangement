import os

import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def make_robot_description(prefix: str, ur_type: str, controllers_file: str, initial_positions_file: str):
    urdf_xacro = os.path.join(
        get_package_share_directory("ur_hande_description"), "urdf", "ur_hande.urdf.xacro"
    )
    robot_description_config = xacro.process_file(
        urdf_xacro,
        mappings={
            "name": "ur",
            "ur_type": ur_type,
            "tf_prefix": prefix,
            "use_fake_hardware": "false",
            "sim_gazebo": "false",
            "sim_ignition": "true",
            "simulation_controllers": controllers_file,
            "initial_positions_file": initial_positions_file,
        },
    )
    return {"robot_description": robot_description_config.toxml()}


def make_semantic_description(prefix: str):
    srdf_xacro = os.path.join(get_package_share_directory("ur_moveit_config"), "srdf", "ur.srdf.xacro")
    semantic_config = xacro.process_file(srdf_xacro, mappings={"name": "ur", "prefix": prefix})
    return {"robot_description_semantic": semantic_config.toxml()}


def make_inspector_node(
    arm: str,
    prefix: str,
    controllers_file: str,
    ur_type: str,
    initial_positions_file: str,
):
    return Node(
        package="ur_moveit_config",
        executable="self_collision_inspector",
        namespace=arm,
        name="self_collision_inspector",
        output="screen",
        parameters=[
            make_robot_description(prefix, ur_type, controllers_file, initial_positions_file),
            make_semantic_description(prefix),
            {
                "joint_states_topic": f"/{arm}/joint_states_servo",
                "group_name": f"{prefix}ur_manipulator",
                "distance_threshold": 0.08,
                "max_pairs": 30,
                "wait_timeout_sec": 4.0,
                "use_sim_time": True,
            },
        ],
    )


def _launch_setup(context, *args, **kwargs):
    ur_type = LaunchConfiguration("ur_type").perform(context)
    left_controllers = LaunchConfiguration("left_simulation_controllers").perform(context)
    right_controllers = LaunchConfiguration("right_simulation_controllers").perform(context)

    return [
        make_inspector_node(
            "left_arm",
            "left_",
            left_controllers,
            ur_type,
            "/workspace/src/ur_hande_description/config/initial_positions.yaml",
        ),
        make_inspector_node(
            "right_arm",
            "right_",
            right_controllers,
            ur_type,
            "/workspace/src/ur_hande_description/config/initial_positions_right_180.yaml",
        ),
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("ur_type", default_value="ur5e"),
            DeclareLaunchArgument(
                "left_simulation_controllers",
                default_value="/workspace/simulation/config/ur5e_gz_controllers_left.yaml",
            ),
            DeclareLaunchArgument(
                "right_simulation_controllers",
                default_value="/workspace/simulation/config/ur5e_gz_controllers_right.yaml",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
