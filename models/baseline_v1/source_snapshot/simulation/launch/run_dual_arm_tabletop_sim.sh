#!/usr/bin/env bash
set -euo pipefail

set +u
source /opt/ros/humble/setup.bash
if [[ ! -f /workspace/install/setup.bash ]]; then
  echo "[error] Missing /workspace/install/setup.bash"
  echo "        Build the workspace first: ./scripts/backend11_lifecycle.sh build_ws"
  exit 1
fi
source /workspace/install/setup.bash
set -u

rm -f /tmp/dual_sim_ready
WORLD_FILE="/workspace/simulation/worlds/ur_hande_dual_arm_tabletop.sdf"
WORLD_NAME="ur_hande_dual_arm_tabletop"
SIM_HEADLESS="${SIM_HEADLESS:-1}"
LEFT_URDF_FILE="/tmp/left_ur5e_hande_dual.urdf"
RIGHT_URDF_FILE="/tmp/right_ur5e_hande_dual.urdf"
LEFT_RSP_PARAMS="/tmp/left_robot_state_publisher_params.yaml"
RIGHT_RSP_PARAMS="/tmp/right_robot_state_publisher_params.yaml"

pkill -9 -x ruby || true
pkill -9 -x robot_state_pub || true
pkill -9 -x ros2_control_node || true
pkill -9 -x parameter_bridge || true
pkill -9 -x image_bridge || true
pkill -9 -f 'ign gazebo' || true
pkill -9 -f 'robot_state_publisher' || true
pkill -9 -f 'ros_gz_bridge.*parameter_bridge' || true
pkill -9 -f 'ros_gz_image.*image_bridge' || true
pkill -9 -x ros2 || true

export IGN_GAZEBO_SYSTEM_PLUGIN_PATH="/opt/ros/humble/lib:${IGN_GAZEBO_SYSTEM_PLUGIN_PATH:-}"
export IGN_GAZEBO_RESOURCE_PATH="/workspace/install/robotiq_hande_description/share:/workspace/install/ur_hande_description/share:/opt/ros/humble/share:${IGN_GAZEBO_RESOURCE_PATH:-}"

if [[ "${SIM_GPU_RENDERING:-0}" == "1" ]]; then
  GZ_ARGS=(-s -r --headless-rendering)
  echo "Starting independent Ogre2/EGL GPU server. Viewer is a separate process."
elif [[ "${SIM_HEADLESS}" == "1" ]]; then
  GZ_ARGS=(-s -r)
  echo "Starting dual-arm Ignition Gazebo in headless mode (SIM_HEADLESS=1)."
else
  GZ_ARGS=(-r)
  echo "Starting dual-arm Ignition Gazebo with GUI (SIM_HEADLESS=0)."
fi

nohup ign gazebo "${GZ_ARGS[@]}" "${WORLD_FILE}" >/tmp/gz_dual_arm_tabletop.log 2>&1 </dev/null &
sleep 3

nohup ros2 run ros_gz_bridge parameter_bridge \
  '/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock' \
  >/tmp/gz_dual_arm_clock_bridge.log 2>&1 </dev/null &
sleep 0.5

python3 /workspace/scripts/scene_runtime_config.py
CAMERA_HFOV="$(cat /tmp/scene_camera_hfov)"

make_robot_urdf() {
  local prefix="$1"
  local robot_name="$2"
  local controllers_file="$3"
  local initial_positions_file="$4"
  local output_file="$5"

  xacro /workspace/src/ur_hande_description/urdf/ur_hande.urdf.xacro \
    ur_type:=ur5e \
    name:="${robot_name}" \
    tf_prefix:="${prefix}" \
    use_fake_hardware:=false \
    sim_ignition:=true \
    sim_gazebo:=false \
    initial_positions_file:="${initial_positions_file}" \
    simulation_controllers:="${controllers_file}" \
    camera_hfov:="${CAMERA_HFOV}" \
    enable_gripper_camera:="$([[ "${prefix}" == right_ && "${ENABLE_GRIPPER_CAMERA:-0}" == 1 ]] && echo true || echo false)" \
    > "${output_file}"
}

make_robot_urdf left_ left_ur5e /workspace/simulation/config/ur5e_gz_controllers_left.yaml /tmp/scene_left_initial.yaml "${LEFT_URDF_FILE}"
make_robot_urdf right_ right_ur5e /workspace/simulation/config/ur5e_gz_controllers_right.yaml /tmp/scene_right_initial.yaml "${RIGHT_URDF_FILE}"

write_rsp_params() {
  local node_name="$1"
  local urdf_file="$2"
  local params_file="$3"

  {
    printf '%s:\n' "${node_name}"
    printf '  ros__parameters:\n'
    printf '    robot_description: |\n'
    sed 's/^/      /' "${urdf_file}"
  } > "${params_file}"
}

write_rsp_params left_robot_state_publisher "${LEFT_URDF_FILE}" "${LEFT_RSP_PARAMS}"
write_rsp_params right_robot_state_publisher "${RIGHT_URDF_FILE}" "${RIGHT_RSP_PARAMS}"

nohup ros2 run robot_state_publisher robot_state_publisher --ros-args -r __node:=left_robot_state_publisher --params-file "${LEFT_RSP_PARAMS}" >/tmp/rsp_left_dual_arm.log 2>&1 </dev/null &
nohup ros2 run robot_state_publisher robot_state_publisher --ros-args -r __node:=right_robot_state_publisher --params-file "${RIGHT_RSP_PARAMS}" >/tmp/rsp_right_dual_arm.log 2>&1 </dev/null &
sleep 0.5

# Spawn arms near the left/right sides of the table.
# Keep these values in sync with profiles/robots/ur5e_hande_dual/robot.yaml.
timeout 20 ros2 run ros_gz_sim create -world "${WORLD_NAME}" -name left_ur5e_hande -file "${LEFT_URDF_FILE}" -x 0.0 -y 0.60 -z 0.0 -Y 0.0
timeout 20 ros2 run ros_gz_sim create -world "${WORLD_NAME}" -name right_ur5e_hande -file "${RIGHT_URDF_FILE}" -x 0.0 -y -0.60 -z 0.0 -Y 0.0

spawn_controller() {
  local manager="$1"
  local controller_name="$2"
  local params_file="$3"
  timeout 20 ros2 run controller_manager spawner "${controller_name}" \
    --controller-manager "${manager}" \
    -p "${params_file}"
}

spawn_controller /left_controller_manager left_joint_state_broadcaster /workspace/simulation/config/ur5e_gz_controllers_left.yaml
spawn_controller /left_controller_manager left_joint_group_velocity_controller /workspace/simulation/config/ur5e_gz_controllers_left.yaml
spawn_controller /left_controller_manager left_hande_position_controller /workspace/simulation/config/ur5e_gz_controllers_left.yaml
spawn_controller /right_controller_manager right_joint_state_broadcaster /workspace/simulation/config/ur5e_gz_controllers_right.yaml
spawn_controller /right_controller_manager right_joint_group_velocity_controller /workspace/simulation/config/ur5e_gz_controllers_right.yaml
spawn_controller /right_controller_manager right_hande_position_controller /workspace/simulation/config/ur5e_gz_controllers_right.yaml

touch /tmp/dual_sim_ready
if [[ "${ENABLE_GRIPPER_CAMERA:-0}" == "1" ]]; then
  nohup ros2 run ros_gz_bridge parameter_bridge \
    '/right_gripper_camera/image_raw@sensor_msgs/msg/Image[ignition.msgs.Image' \
    '/right_gripper_camera/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo' \
    --ros-args -p override_frame_id:=right_gripper_camera_optical_frame \
    -p qos_overrides./right_gripper_camera/image_raw.publisher.reliability:=best_effort \
    >/tmp/gripper_camera_bridge.log 2>&1 </dev/null &
fi
echo "Spawned dual UR5e Hand-E scaffold in ${WORLD_NAME}."
echo "SIM_HEADLESS=${SIM_HEADLESS}"
echo "Logs: /tmp/gz_dual_arm_tabletop.log /tmp/rsp_left_dual_arm.log /tmp/rsp_right_dual_arm.log"
echo "Next: run ./scripts/backend11_lifecycle.sh start_dual_servo and start_dual_part23."
