import copy
import os

import xacro
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def load_yaml(package_name: str, relative_path: str):
    pkg_share = get_package_share_directory(package_name)
    path = os.path.join(pkg_share, relative_path)
    with open(path, "r") as f:
        return yaml.safe_load(f)


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


def make_kinematics(prefix: str):
    kin = load_yaml("ur_moveit_config", "config/kinematics.yaml")
    base_kin = kin.get("/**", {}).get("ros__parameters", {}).get("robot_description_kinematics", {})
    solver = copy.deepcopy(base_kin.get("ur_manipulator", {}))
    return {"robot_description_kinematics": {f"{prefix}ur_manipulator": solver}}


def make_planning(prefix: str):
    planning = load_yaml("ur_moveit_config", "config/joint_limits.yaml")
    planning = copy.deepcopy(planning) if planning else {}
    joint_limits = planning.get("joint_limits", {})
    if joint_limits:
        planning["joint_limits"] = {
            f"{prefix}{joint_name}": limits for joint_name, limits in joint_limits.items()
        }
    return {"robot_description_planning": planning}


def make_servo_params(prefix: str, namespace: str, command_topic: str):
    servo_yaml = load_yaml("servo_test_config", "config/servo_gz.yaml")
    servo = copy.deepcopy(servo_yaml)
    check_collisions_override = os.environ.get("SERVO_CHECK_COLLISIONS", "").strip().lower()
    if check_collisions_override in {"0", "false", "off", "no"}:
        servo["check_collisions"] = False
    elif check_collisions_override in {"1", "true", "on", "yes"}:
        servo["check_collisions"] = True
    for env_name, param_name in (
        ("SERVO_SELF_COLLISION_THRESHOLD", "self_collision_proximity_threshold"),
        ("SERVO_SCENE_COLLISION_THRESHOLD", "scene_collision_proximity_threshold"),
        ("SERVO_COLLISION_CHECK_RATE", "collision_check_rate"),
    ):
        override = os.environ.get(env_name, "").strip()
        if override:
            servo[param_name] = float(override)
    servo["move_group_name"] = f"{prefix}ur_manipulator"
    servo["planning_frame"] = f"{prefix}base_link"
    servo["ee_frame_name"] = f"{prefix}tool0"
    servo["robot_link_command_frame"] = f"{prefix}tool0"
    servo["command_out_topic"] = command_topic
    servo["cartesian_command_in_topic"] = f"/{namespace}/servo_node/delta_twist_cmds"
    servo["joint_command_in_topic"] = f"/{namespace}/servo_node/delta_joint_cmds"
    servo["joint_topic"] = f"/{namespace}/joint_states_servo"
    servo["status_topic"] = f"/{namespace}/servo_node/status"
    return {"moveit_servo": servo}


def make_arm_nodes(
    arm: str,
    prefix: str,
    controllers_file: str,
    command_topic: str,
    ur_type: str,
    initial_positions_file: str,
):
    robot_description = make_robot_description(prefix, ur_type, controllers_file, initial_positions_file)
    robot_description_semantic = make_semantic_description(prefix)
    robot_description_kinematics = make_kinematics(prefix)
    robot_description_planning = make_planning(prefix)
    servo_params = make_servo_params(prefix, arm, command_topic)

    joint_filter = Node(
        package="servo_test_config",
        executable="joint_states_filter",
        namespace=arm,
        name="joint_states_filter",
        output="screen",
        parameters=[
            {
                "source_topic": "/joint_states",
                "output_topic": f"/{arm}/joint_states_servo",
                "keep_prefixes": [prefix],
                "drop_suffixes": ["_mimic"],
            }
        ],
    )

    servo_node = Node(
        package="moveit_servo",
        executable="servo_node_main",
        namespace=arm,
        name="servo_node",
        output="screen",
        parameters=[
            servo_params,
            robot_description,
            robot_description_semantic,
            robot_description_kinematics,
            robot_description_planning,
            {"use_sim_time": True},
        ],
    )
    return [joint_filter, servo_node]


def _launch_setup(context, *args, **kwargs):
    ur_type = LaunchConfiguration("ur_type").perform(context)
    left_controllers = LaunchConfiguration("left_simulation_controllers").perform(context)
    right_controllers = LaunchConfiguration("right_simulation_controllers").perform(context)

    return [
        *make_arm_nodes(
            "left_arm",
            "left_",
            left_controllers,
            "/left_joint_group_velocity_controller/commands",
            ur_type,
            "/workspace/src/ur_hande_description/config/initial_positions.yaml",
        ),
        *make_arm_nodes(
            "right_arm",
            "right_",
            right_controllers,
            "/right_joint_group_velocity_controller/commands",
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
