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

WORLD_FILE="/workspace/simulation/worlds/ur_hande_tabletop.sdf"
WORLD_NAME="ur_hande_tabletop"
URDF_FILE="/tmp/ur5e_hande_tabletop.urdf"
ENTITY_NAME="${ENTITY_NAME:-ur5e_hande}"
SIM_HEADLESS="${SIM_HEADLESS:-1}"

SPAWN_X="${SPAWN_X:-0.0}"
SPAWN_Y="${SPAWN_Y:-0.0}"
SPAWN_Z="${SPAWN_Z:-0.0}"

# Keep test runs deterministic by clearing old sim and publisher processes.
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

if [[ "${SIM_HEADLESS}" == "1" ]]; then
  GZ_ARGS=(-s -r)
  echo "Starting Ignition Gazebo in headless mode (SIM_HEADLESS=1)."
else
  GZ_ARGS=(-r)
  echo "Starting Ignition Gazebo with GUI (SIM_HEADLESS=0)."
fi

nohup ign gazebo "${GZ_ARGS[@]}" "${WORLD_FILE}" >/tmp/gz_tabletop.log 2>&1 </dev/null &
sleep 3

nohup ros2 run ros_gz_bridge parameter_bridge \
  '/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock' \
  >/tmp/gz_clock_bridge.log 2>&1 </dev/null &
sleep 0.5

xacro /workspace/src/ur_hande_description/urdf/ur_hande.urdf.xacro \
  ur_type:=ur5e \
  name:=ur5e \
  use_fake_hardware:=false \
  sim_ignition:=true \
  sim_gazebo:=false \
  initial_positions_file:=/workspace/src/ur_hande_description/config/initial_positions.yaml \
  simulation_controllers:=/workspace/simulation/config/ur5e_gz_controllers.yaml \
  > "${URDF_FILE}"

nohup ros2 run robot_state_publisher robot_state_publisher "${URDF_FILE}" >/tmp/rsp_tabletop.log 2>&1 </dev/null &

timeout 20 ros2 run ros_gz_sim create \
  -world "${WORLD_NAME}" \
  -name "${ENTITY_NAME}" \
  -file "${URDF_FILE}" \
  -x "${SPAWN_X}" \
  -y "${SPAWN_Y}" \
  -z "${SPAWN_Z}"

spawn_controller() {
  local controller_name="$1"
  timeout 20 ros2 run controller_manager spawner "${controller_name}" \
    --controller-manager /controller_manager \
    -p /workspace/simulation/config/ur5e_gz_controllers.yaml
}

spawn_controller joint_state_broadcaster
spawn_controller joint_group_velocity_controller
spawn_controller hande_position_controller

echo "Spawned ${ENTITY_NAME} in ${WORLD_NAME} at (${SPAWN_X}, ${SPAWN_Y}, ${SPAWN_Z})."
echo "SIM_HEADLESS=${SIM_HEADLESS}"
echo "Logs: /tmp/gz_tabletop.log /tmp/rsp_tabletop.log"
