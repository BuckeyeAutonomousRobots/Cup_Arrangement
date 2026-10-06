#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Compose and lifecycle resolve the same container, including clean clones.
if [[ ! -f "${ROOT_DIR}/.env" ]]; then cp "${ROOT_DIR}/.env.example" "${ROOT_DIR}/.env"; fi
CONTAINER="${CONTAINER:-$(sed -n 's/^CONTAINER=//p' "${ROOT_DIR}/.env" | tail -n 1)}"
export CONTAINER="${CONTAINER:-cup_arrangement_backend}"
RECEIVER_PACKAGE="receiver"
BUILD_PACKAGES="robotiq_hande_description ur_hande_description ur_moveit_config servo_test_config teleop_bridge_msgs teleop_bridge ros_tcp_endpoint receiver"
TELEOP_TUNING_FILE="/workspace/src/teleop_bridge/config/teleop_tuning.yaml"
DUAL_LEFT_TUNING_FILE="/workspace/src/teleop_bridge/config/teleop_tuning_dual_left.yaml"
DUAL_RIGHT_TUNING_FILE="/workspace/src/teleop_bridge/config/teleop_tuning_dual_right.yaml"
ENV_FILE="${ROOT_DIR}/.env"
ACTIVE_SCENE_PROFILE_FILE="${ROOT_DIR}/profiles/active_scene_profile.txt"
DEFAULT_DUAL_SCENE_PROFILE="${ROOT_DIR}/profiles/scenes/dual_arm_tabletop/scene.yaml"
KEEP_RUNTIME_LOGS="${KEEP_RUNTIME_LOGS:-5}"
KEEP_RECORDING_SESSIONS="${KEEP_RECORDING_SESSIONS:-5}"
CLEAN_UNITY_BUILD_OUTPUTS="${CLEAN_UNITY_BUILD_OUTPUTS:-0}"

TELEOP_PATTERN="quest_controller_receiver|hand_pose_mapper|servo_command_bridge|simple_gripper_command_bridge|coupled_gripper_controller|reset_manager|keyboard_servo_override|debug_hand_generator|servo_response_sampler|task_pose_sync_publisher|contact_haptic_publisher|rubik_task_controller|runtime_task_manager|received_pose_to_target_twist|target_twist_to_servo_cmd|target_twist_to_gripper_cmd|coupled_hande_gripper_controller|target_twist_reset_manager|keyboard_servo_cmd|dual_debug_hand_generator|cube_pose_sync_publisher|gazebo_contact_haptic_publisher|rubik2x2_mechanism_controller|task_manager|ros_tcp_endpoint.*endpoint.py|default_server_endpoint|run_tabletop_sim.sh|run_dual_arm_tabletop_sim.sh|servo_gz.launch.py|servo_dual_gz.launch.py|servo_node_main|joint_states_filter|ros_gz_bridge.* /clock@|ros_gz_bridge.*dynamic_pose/info"

dc() {
  local extra=()
  if [[ "${ENABLE_GPU:-$(env_value ENABLE_GPU 0)}" == "1" ]]; then
    extra=(-f docker-compose.gpu.yaml)
  fi
  (cd "${ROOT_DIR}" && LOCAL_UID="$(id -u)" LOCAL_GID="$(id -g)" docker compose -f docker-compose.yaml "${extra[@]}" "$@")
}

container_running() {
  docker ps --format '{{.Names}}' | grep -qx "${CONTAINER}"
}

require_running() {
  if ! container_running; then
    echo "[error] Container ${CONTAINER} is not running."
    echo "        Start it with: ./scripts/backend11_lifecycle.sh up_container"
    exit 1
  fi
}

dexec() {
  docker exec "${CONTAINER}" bash -lc "$1"
}

float_arg() {
  python3 - "$1" <<'PY'
import sys
print(str(float(sys.argv[1])))
PY
}

prune_container_ros_logs() {
  if ! container_running; then
    return 0
  fi
  local keep="${1:-${KEEP_RUNTIME_LOGS}}"
  dexec "keep=${keep}; \
    for root in /home/robot/.ros/log /root/.ros/log; do \
      [ -d \"\$root\" ] || continue; \
      find \"\$root\" -mindepth 1 -maxdepth 1 -type d -print | sort | \
        awk -v keep=\"\$keep\" '{ paths[NR]=\$0 } END { limit=NR-keep; for (i=1; i<=limit; i++) print paths[i] }' | \
        xargs -r rm -rf; \
    done"
}

prune_recording_sessions() {
  local keep="${1:-${KEEP_RECORDING_SESSIONS}}"
  local root="${ROOT_DIR}/runtime/recordings/gripper_camera"
  [ -d "${root}" ] || return 0
  find "${root}" -mindepth 1 -maxdepth 1 -type d -print | sort | \
    awk -v keep="${keep}" '{ paths[NR]=$0 } END { limit=NR-keep; for (i=1; i<=limit; i++) print paths[i] }' | \
    xargs -r rm -rf
}

cleanup_runtime() {
  prune_container_ros_logs "${KEEP_RUNTIME_LOGS}"
  prune_recording_sessions "${KEEP_RECORDING_SESSIONS}"
  rm -rf "${ROOT_DIR}/../UnityApp/.utmp" \
         "${ROOT_DIR}/../UnityApp/Logs"
  if [ "${CLEAN_UNITY_BUILD_OUTPUTS}" = "1" ]; then
    rm -rf "${ROOT_DIR}/../UnityApp/BuildTest" \
           "${ROOT_DIR}/../UnityApp/App_Build"
  fi
  echo "[ok] Runtime cleanup complete."
  echo "     Kept latest ${KEEP_RUNTIME_LOGS} ROS log directories in the container."
  echo "     Kept latest ${KEEP_RECORDING_SESSIONS} gripper-camera recording sessions."
  echo "     Removed UnityApp/.utmp and UnityApp/Logs."
  if [ "${CLEAN_UNITY_BUILD_OUTPUTS}" = "1" ]; then
    echo "     Removed Unity build output folders: UnityApp/BuildTest and UnityApp/App_Build."
  else
    echo "     Preserved Unity build output folders. Set CLEAN_UNITY_BUILD_OUTPUTS=1 to delete them."
  fi
}

set_env_var() {
  local key="$1"
  local value="$2"
  local tmp_file
  tmp_file="$(mktemp)"
  awk -F= -v k="${key}" -v v="${value}" '
BEGIN { set = 0 }
$1 == k {
  print k "=" v
  set = 1
  next
}
{ print $0 }
END {
  if (set == 0) {
    print k "=" v
  }
}
' "${ENV_FILE}" > "${tmp_file}"
  mv "${tmp_file}" "${ENV_FILE}"
}

env_value() {
  local key="$1"
  local fallback="${2:-}"
  local value=""
  if [[ -f "${ENV_FILE}" ]]; then
    value="$(awk -F= -v k="${key}" '$1 == k {print $2}' "${ENV_FILE}" | tail -n 1)"
  fi
  echo "${value:-${fallback}}"
}

quest_tcp_host_port() {
  local port
  port="$(awk -F= '/^QUEST_TCP_HOST_PORT=/{print $2}' "${ENV_FILE}" | tail -n 1)"
  echo "${port:-5026}"
}

quest_tcp_unity_port() {
  local port
  port="$(awk -F= '/^QUEST_TCP_UNITY_PORT=/{print $2}' "${ENV_FILE}" | tail -n 1)"
  echo "${port:-5026}"
}

ros_tcp_host_port() {
  local port
  port="$(awk -F= '/^ROS_TCP_HOST_PORT=/{print $2}' "${ENV_FILE}" | tail -n 1)"
  echo "${port:-10001}"
}

ros_tcp_unity_port() {
  local port
  port="$(awk -F= '/^ROS_TCP_UNITY_PORT=/{print $2}' "${ENV_FILE}" | tail -n 1)"
  echo "${port:-10001}"
}

quest_tcp_host_bind() {
  local bind
  bind="$(awk -F= '/^QUEST_TCP_HOST_BIND=/{print $2}' "${ENV_FILE}" | tail -n 1)"
  echo "${bind:-127.0.0.1}"
}

ros_tcp_host_bind() {
  local bind
  bind="$(awk -F= '/^ROS_TCP_HOST_BIND=/{print $2}' "${ENV_FILE}" | tail -n 1)"
  echo "${bind:-127.0.0.1}"
}

adb_device_connected() {
  command -v adb >/dev/null 2>&1 || return 1
  adb devices | awk 'NR > 1 && $2 == "device" { found = 1 } END { exit(found ? 0 : 1) }'
}

mode_wired() {
  set_env_var "ROS_TCP_HOST_BIND" "127.0.0.1"
  set_env_var "QUEST_TCP_HOST_BIND" "127.0.0.1"
  set_env_var "ROS_TCP_UNITY_PORT" "$(ros_tcp_unity_port)"
  set_env_var "QUEST_TCP_UNITY_PORT" "$(quest_tcp_unity_port)"
  echo "[ok] Backend network mode set to WIRED in ${ENV_FILE}."
  echo "     ROS-TCP: Quest 127.0.0.1:$(ros_tcp_unity_port) -> host 127.0.0.1:$(ros_tcp_host_port) -> container 10000"
  echo "     Hand TCP: Quest 127.0.0.1:$(quest_tcp_unity_port) -> host 127.0.0.1:$(quest_tcp_host_port) -> container 5005"
  echo "     Restart the container to apply Docker port-binding changes."
}

mode_wireless() {
  set_env_var "ROS_TCP_HOST_BIND" "0.0.0.0"
  set_env_var "QUEST_TCP_HOST_BIND" "0.0.0.0"
  echo "[ok] Backend network mode set to WIRELESS in ${ENV_FILE}."
  echo "     ROS-TCP: 0.0.0.0:$(ros_tcp_host_port) -> container 10000"
  echo "     Hand TCP: 0.0.0.0:$(quest_tcp_host_port) -> container 5005"
  echo "     Restart the container to apply Docker port-binding changes."
  echo "     Unity/Quest should use your host LAN IP, not 127.0.0.1."
}

mode_status() {
  echo "--- network mode from ${ENV_FILE}"
  echo "ros tcp bind:   $(ros_tcp_host_bind):$(ros_tcp_host_port)"
  echo "quest tcp bind: $(quest_tcp_host_bind):$(quest_tcp_host_port)"

  if container_running; then
    echo "--- live docker port mapping"
    docker port "${CONTAINER}" || true
  else
    echo "--- live docker port mapping"
    echo "container not running"
  fi
}

workspace_ready() {
  if ! container_running; then
    return 1
  fi
  docker exec "${CONTAINER}" bash -lc "source /opt/ros/humble/setup.bash && \
    [ -f /workspace/install/setup.bash ] && \
    source /workspace/install/setup.bash && \
    ros2 pkg prefix teleop_bridge >/dev/null 2>&1 && \
    ros2 pkg prefix servo_test_config >/dev/null 2>&1 && \
    ros2 pkg executables teleop_bridge 2>/dev/null | awk '\$1 == \"teleop_bridge\" && \$2 == \"runtime_task_manager\" { found = 1 } END { exit(found ? 0 : 1) }'"
}

build_ws() {
  require_running
  dexec "source /opt/ros/humble/setup.bash && cd /workspace && \
    colcon build --packages-select ${BUILD_PACKAGES}"
  echo "[ok] Built workspace packages: ${BUILD_PACKAGES}"
}

receiver_ready() {
  if ! container_running; then
    return 1
  fi
  docker exec "${CONTAINER}" bash -lc "source /opt/ros/humble/setup.bash && \
    [ -f /workspace/install/setup.bash ] && \
    source /workspace/install/setup.bash && \
    ros2 pkg prefix ${RECEIVER_PACKAGE} >/dev/null 2>&1"
}

build_receiver() {
  require_running
  dexec "source /opt/ros/humble/setup.bash && cd /workspace && \
    colcon build --packages-select ${RECEIVER_PACKAGE}"
  echo "[ok] Built receiver package: ${RECEIVER_PACKAGE}"
}

ensure_receiver_built() {
  if receiver_ready; then
    echo "[ok] Receiver package already built (${RECEIVER_PACKAGE})."
    return 0
  fi
  echo "[info] Receiver package not ready. Building now..."
  build_receiver
}

ensure_ws_built() {
  if workspace_ready; then
    echo "[ok] Workspace already built (teleop_bridge + servo_test_config found)."
    return 0
  fi
  echo "[info] Workspace not ready in container. Building now..."
  build_ws
}

stop_nodes() {
  if ! container_running; then
    echo "[info] ${CONTAINER} not running; skip node stop."
    return 0
  fi

  dexec "self=\$\$; \
    for p in \$(pgrep -f '${TELEOP_PATTERN}' || true); do \
      [ \"\$p\" = \"\$self\" ] && continue; \
      kill -2 \"\$p\" 2>/dev/null || true; \
    done; \
    sleep 1; \
    for p in \$(pgrep -f '${TELEOP_PATTERN}' || true); do \
      [ \"\$p\" = \"\$self\" ] && continue; \
      kill -9 \"\$p\" 2>/dev/null || true; \
    done"

  echo "[ok] Stopped teleop/simulation processes inside ${CONTAINER}."
}

safe_down() {
  stop_nodes
  dc down --remove-orphans || true

  while IFS= read -r cid; do
    [ -z "${cid}" ] && continue
    docker rm -f "${cid}" >/dev/null 2>&1 || true
  done < <(docker ps -a --format '{{.ID}} {{.Names}}' | awk -v name="${CONTAINER}" '$2 == name {print $1}')

  echo "[ok] Compose stack is down and stale containers are cleaned."
}

up_container() {
  dc up -d
  docker ps --filter "name=^/${CONTAINER}$" --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
  docker port "${CONTAINER}" || true
}

up_container_build() {
  dc up -d --build
  docker ps --filter "name=^/${CONTAINER}$" --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
  docker port "${CONTAINER}" || true
}

wired_on() {
  local quest_unity_port quest_host_port ros_unity_port ros_host_port
  quest_unity_port="$(quest_tcp_unity_port)"
  quest_host_port="$(quest_tcp_host_port)"
  ros_unity_port="$(ros_tcp_unity_port)"
  ros_host_port="$(ros_tcp_host_port)"

  if ! command -v adb >/dev/null 2>&1; then
    echo "[error] adb is not installed or not in PATH."
    exit 1
  fi

  if ! adb_device_connected; then
    echo "[error] No ADB device is connected."
    echo "        Connect the Quest by USB, allow USB debugging in the headset, then retry."
    exit 1
  fi

  adb reverse "tcp:${quest_unity_port}" "tcp:${quest_host_port}"
  adb reverse "tcp:${ros_unity_port}" "tcp:${ros_host_port}"
  echo "[ok] Wired TCP tunnels enabled:"
  echo "     HandPoseSender: Quest 127.0.0.1:${quest_unity_port} -> Mac 127.0.0.1:${quest_host_port}"
  echo "     ROS-TCP:        Quest 127.0.0.1:${ros_unity_port} -> Mac 127.0.0.1:${ros_host_port}"
  echo "     Unity HandPoseSender should use targetIP=127.0.0.1 targetPort=${quest_unity_port}"
  echo "     Unity ROS Settings should use ROS IP Address=127.0.0.1 ROS Port=${ros_unity_port}"
}

wired_off() {
  local quest_unity_port ros_unity_port
  quest_unity_port="$(quest_tcp_unity_port)"
  ros_unity_port="$(ros_tcp_unity_port)"

  if ! command -v adb >/dev/null 2>&1; then
    echo "[error] adb is not installed or not in PATH."
    exit 1
  fi

  adb reverse --remove "tcp:${quest_unity_port}" || true
  adb reverse --remove "tcp:${ros_unity_port}" || true
  echo "[ok] Wired TCP tunnels removed for Quest tcp:${quest_unity_port} and tcp:${ros_unity_port}."
}

wired_status() {
  local quest_host_port quest_unity_port quest_bind ros_host_port ros_unity_port ros_bind
  quest_host_port="$(quest_tcp_host_port)"
  quest_unity_port="$(quest_tcp_unity_port)"
  quest_bind="$(quest_tcp_host_bind)"
  ros_host_port="$(ros_tcp_host_port)"
  ros_unity_port="$(ros_tcp_unity_port)"
  ros_bind="$(ros_tcp_host_bind)"

  echo "--- wired defaults"
  echo "quest hand host bind: ${quest_bind}"
  echo "quest hand host port: ${quest_host_port}"
  echo "quest hand app port:  ${quest_unity_port}"
  echo "quest hand tunnel:    Quest 127.0.0.1:${quest_unity_port} -> host 127.0.0.1:${quest_host_port}"
  echo "ros tcp host bind:    ${ros_bind}"
  echo "ros tcp host port:    ${ros_host_port}"
  echo "ros tcp app port:     ${ros_unity_port}"
  echo "ros tcp tunnel:       Quest 127.0.0.1:${ros_unity_port} -> host 127.0.0.1:${ros_host_port}"

  if command -v adb >/dev/null 2>&1; then
    echo "--- adb devices"
    adb devices || true
    echo "--- adb reverse"
    adb reverse --list 2>/dev/null || true
  else
    echo "--- adb"
    echo "adb not found"
  fi

  echo "--- docker port"
  docker port "${CONTAINER}" 2>/dev/null || true
}

maybe_wired_on() {
  if [ "$(quest_tcp_host_bind)" != "127.0.0.1" ] && [ "$(ros_tcp_host_bind)" != "127.0.0.1" ]; then
    return 0
  fi

  if ! command -v adb >/dev/null 2>&1; then
    echo "[warn] adb not found; skipping wired TCP tunnels."
    return 0
  fi

  if adb_device_connected; then
    wired_on
  else
    echo "[warn] No ADB device connected; skipping wired TCP tunnels."
  fi
}

start_receiver() {
  require_running
  local reduced="${REDUCED_PROCESS_MODE:-$(env_value REDUCED_PROCESS_MODE 1)}"
  ensure_receiver_built
  dexec "source /opt/ros/humble/setup.bash && \
    source /workspace/install/setup.bash && \
    self=\$\$; \
    for p in \$(pgrep -f 'quest_controller_receiver' || true); do \
      [ \"\$p\" = \"\$self\" ] && continue; kill -2 \"\$p\" 2>/dev/null || true; \
    done; \
    sleep 0.4; \
    for p in \$(pgrep -f 'quest_controller_receiver' || true); do \
      [ \"\$p\" = \"\$self\" ] && continue; kill -9 \"\$p\" 2>/dev/null || true; \
    done; \
    if [ '${reduced}' != 1 ]; then nohup /opt/ros/humble/bin/ros2 run receiver quest_controller_receiver >/tmp/qcr.log 2>&1 < /dev/null & fi"
  echo "[ok] Quest receiver configured (REDUCED_PROCESS_MODE=${reduced})."
}

start_part23() {
  require_running
  ensure_ws_built
  local part23_pattern="hand_pose_mapper|servo_command_bridge|simple_gripper_command_bridge|coupled_gripper_controller|reset_manager|received_pose_to_target_twist|target_twist_to_servo_cmd|target_twist_to_gripper_cmd|coupled_hande_gripper_controller|target_twist_reset_manager"
  dexec "source /opt/ros/humble/setup.bash && source /workspace/install/setup.bash && \
    self=\$\$; \
    for p in \$(pgrep -f '${part23_pattern}' || true); do \
      [ \"\$p\" = \"\$self\" ] && continue; kill -2 \"\$p\" 2>/dev/null || true; \
    done; \
    sleep 0.4; \
    for p in \$(pgrep -f '${part23_pattern}' || true); do \
      [ \"\$p\" = \"\$self\" ] && continue; kill -9 \"\$p\" 2>/dev/null || true; \
    done; \
    nohup /opt/ros/humble/bin/ros2 run teleop_bridge hand_pose_mapper --ros-args --params-file ${TELEOP_TUNING_FILE} >/tmp/part2_mapper.log 2>&1 < /dev/null & \
    nohup /opt/ros/humble/bin/ros2 run teleop_bridge servo_command_bridge --ros-args --params-file ${TELEOP_TUNING_FILE} >/tmp/part3_target_to_servo.log 2>&1 < /dev/null & \
    nohup /opt/ros/humble/bin/ros2 run teleop_bridge simple_gripper_command_bridge --ros-args --params-file ${TELEOP_TUNING_FILE} >/tmp/part3_target_to_gripper.log 2>&1 < /dev/null & \
    nohup /opt/ros/humble/bin/ros2 run teleop_bridge reset_manager --ros-args --params-file ${TELEOP_TUNING_FILE} >/tmp/part3_reset_manager.log 2>&1 < /dev/null &"
  echo "[ok] Started Part2/Part3 mapper/bridges."
}

start_part4() {
  require_running
  ensure_ws_built
  local part4_pattern="task_pose_sync_publisher|contact_haptic_publisher|rubik_task_controller|runtime_task_manager|cube_pose_sync_publisher|gazebo_contact_haptic_publisher|rubik2x2_mechanism_controller|task_manager|ros_tcp_endpoint.*endpoint.py|default_server_endpoint|ros_gz_bridge.*dynamic_pose/info"
  dexec "source /opt/ros/humble/setup.bash && source /workspace/install/setup.bash && \
    self=\$\$; \
    for p in \$(pgrep -f '${part4_pattern}' || true); do \
      [ \"\$p\" = \"\$self\" ] && continue; kill -2 \"\$p\" 2>/dev/null || true; \
    done; \
    sleep 0.6; \
    for p in \$(pgrep -f '${part4_pattern}' || true); do \
      [ \"\$p\" = \"\$self\" ] && continue; kill \"\$p\" 2>/dev/null || true; \
    done; \
    nohup /opt/ros/humble/bin/ros2 launch ros_tcp_endpoint endpoint.py >/tmp/part4_tcp_endpoint.log 2>&1 < /dev/null & \
    nohup /opt/ros/humble/bin/ros2 run ros_gz_bridge parameter_bridge '/world/ur_hande_tabletop/dynamic_pose/info@tf2_msgs/msg/TFMessage[ignition.msgs.Pose_V' >/tmp/part4_gz_tf_bridge.log 2>&1 < /dev/null & \
    nohup /opt/ros/humble/bin/ros2 run teleop_bridge task_pose_sync_publisher >/tmp/part4_cube_pose.log 2>&1 < /dev/null &"
  echo "[ok] Started Part4 sync services."
}

start_sim() {
  require_running
  ensure_ws_built
  local sim_headless="${SIM_HEADLESS:-$(env_value SIM_HEADLESS 1)}"
  dexec "source /opt/ros/humble/setup.bash && source /workspace/install/setup.bash && \
    self=\$\$; \
    for p in \$(pgrep -f 'run_tabletop_sim.sh' || true); do \
      [ \"\$p\" = \"\$self\" ] && continue; kill -2 \"\$p\" 2>/dev/null || true; \
    done; \
    sleep 0.4; \
    for p in \$(pgrep -f 'run_tabletop_sim.sh' || true); do \
      [ \"\$p\" = \"\$self\" ] && continue; kill -9 \"\$p\" 2>/dev/null || true; \
    done; \
    nohup env SIM_HEADLESS='${sim_headless}' /workspace/simulation/launch/run_tabletop_sim.sh >/tmp/run_tabletop_sim.log 2>&1 < /dev/null &"
  echo "[ok] Started Gazebo tabletop simulation (SIM_HEADLESS=${sim_headless})."
}

resolve_scene_profile_path() {
  local scene_profile="$1"
  if [[ "${scene_profile}" != /* ]]; then
    scene_profile="${ROOT_DIR}/${scene_profile}"
  fi
  echo "${scene_profile}"
}

current_dual_scene_profile() {
  local scene_profile="${DUAL_SCENE_PROFILE:-}"
  if [[ -z "${scene_profile}" && -f "${ACTIVE_SCENE_PROFILE_FILE}" ]]; then
    scene_profile="$(tr -d '\r\n' < "${ACTIVE_SCENE_PROFILE_FILE}")"
  fi
  if [[ -z "${scene_profile}" ]]; then
    scene_profile="${DEFAULT_DUAL_SCENE_PROFILE}"
  fi
  resolve_scene_profile_path "${scene_profile}"
}

scene_profile_for_task() {
  local task="${1:-}"
  case "${task}" in
    cup_arrangement|cup)
      echo "${ROOT_DIR}/profiles/scenes/dual_arm_cup_arrangement/scene.yaml"
      ;;
    pick_place_basic|pick_place|basic)
      echo "${ROOT_DIR}/profiles/scenes/dual_arm_tabletop/scene.yaml"
      ;;
    rubik_2x2|rubik|rubik2x2)
      echo "${ROOT_DIR}/profiles/scenes/dual_arm_rubik_2x2/scene.yaml"
      ;;
    cable_insertion|cable)
      echo "${ROOT_DIR}/profiles/scenes/dual_arm_cable_insertion/scene.yaml"
      ;;
    *)
      return 1
      ;;
  esac
}

write_active_scene_profile() {
  local scene_profile
  scene_profile="$(resolve_scene_profile_path "$1")"
  mkdir -p "$(dirname "${ACTIVE_SCENE_PROFILE_FILE}")"
  python3 - "$ROOT_DIR" "$scene_profile" "$ACTIVE_SCENE_PROFILE_FILE" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
scene = Path(sys.argv[2]).resolve()
out = Path(sys.argv[3])
try:
    value = scene.relative_to(root)
except ValueError:
    value = scene
out.write_text(str(value) + "\n", encoding="utf-8")
PY
}

generate_dual_world_from_profiles() {
  local generator="${ROOT_DIR}/simulation/tools/generate_dual_arm_world_from_profiles.py"
  local scene_profile="${1:-}"
  if [[ -z "${scene_profile}" ]]; then
    scene_profile="$(current_dual_scene_profile)"
  else
    scene_profile="$(resolve_scene_profile_path "${scene_profile}")"
  fi
  local world_output="${ROOT_DIR}/simulation/worlds/ur_hande_dual_arm_tabletop.sdf"
  if [[ ! -x "${generator}" ]]; then
    echo "[error] Missing world generator: ${generator}"
    exit 1
  fi
  echo "[info] Generating dual-arm world from ${scene_profile}"
  python3 "${generator}" \
    --repo-root "${ROOT_DIR}" \
    --scene-profile "${scene_profile}" \
    --output "${world_output}"
}

list_tasks() {
  cat <<'EOT'
Available tasks:
  cup_arrangement
  pick_place_basic
  rubik_2x2
  cable_insertion
EOT
}

select_task() {
  local task="${1:-}"
  if [[ -z "${task}" ]]; then
    echo "[error] Missing task name."
    list_tasks
    exit 1
  fi

  local scene_profile
  if ! scene_profile="$(scene_profile_for_task "${task}")"; then
    echo "[error] Unknown task: ${task}"
    list_tasks
    exit 1
  fi

  write_active_scene_profile "${scene_profile}"
  echo "[ok] Active task set to ${task} (${scene_profile})."

  if container_running; then
    echo "[info] Container is running; restarting Gazebo, Servo, and Part4 so the selected task is live."
    start_dual_sim
    sleep 2
    start_dual_servo
    sleep 2
    start_dual_part23
    sleep 1
    start_dual_part4
    status
  else
    generate_dual_world_from_profiles "${scene_profile}"
    echo "[info] Container is not running. Next bringup_dual will use ${task}."
  fi
}

start_dual_sim() {
  require_running
  ensure_ws_built
  local sim_headless="${SIM_HEADLESS:-$(env_value SIM_HEADLESS 1)}"
  generate_dual_world_from_profiles
  dexec "source /opt/ros/humble/setup.bash && source /workspace/install/setup.bash && \
    self=\$\$; \
    for p in \$(pgrep -f 'run_dual_arm_tabletop_sim.sh|run_tabletop_sim.sh' || true); do \
      [ \"\$p\" = \"\$self\" ] && continue; kill -2 \"\$p\" 2>/dev/null || true; \
    done; \
    sleep 0.4; \
    for p in \$(pgrep -f 'run_dual_arm_tabletop_sim.sh|run_tabletop_sim.sh' || true); do \
      [ \"\$p\" = \"\$self\" ] && continue; kill -9 \"\$p\" 2>/dev/null || true; \
    done; \
    rm -f /tmp/dual_sim_ready; \
    nohup env SIM_HEADLESS='${sim_headless}' ENABLE_GRIPPER_CAMERA='${ENABLE_GRIPPER_CAMERA:-$(env_value ENABLE_GRIPPER_CAMERA 0)}' /workspace/simulation/launch/run_dual_arm_tabletop_sim.sh >/tmp/run_dual_arm_tabletop_sim.log 2>&1 < /dev/null &"
  for attempt in $(seq 1 90); do
    if docker exec "${CONTAINER}" test -f /tmp/dual_sim_ready; then break; fi
    sleep 1
  done
  docker exec "${CONTAINER}" test -f /tmp/dual_sim_ready || {
    dexec "tail -n 60 /tmp/run_dual_arm_tabletop_sim.log"; return 1;
  }
  echo "[ok] Started dual-arm Gazebo tabletop scaffold (SIM_HEADLESS=${sim_headless})."
}

start_dual_servo() {
  require_running
  ensure_ws_built
  local servo_check_collisions="${SERVO_CHECK_COLLISIONS:-}"
  local servo_self_collision_threshold="${SERVO_SELF_COLLISION_THRESHOLD:-}"
  local servo_scene_collision_threshold="${SERVO_SCENE_COLLISION_THRESHOLD:-}"
  local servo_collision_check_rate="${SERVO_COLLISION_CHECK_RATE:-}"
  dexec "source /opt/ros/humble/setup.bash && source /workspace/install/setup.bash && \
    self=\$\$; \
    for p in \$(pgrep -f 'servo_gz.launch.py|servo_dual_gz.launch.py|servo_node_main|joint_states_filter' || true); do \
      [ \"\$p\" = \"\$self\" ] && continue; kill -2 \"\$p\" 2>/dev/null || true; \
    done; \
    sleep 0.4; \
    for p in \$(pgrep -f 'servo_gz.launch.py|servo_dual_gz.launch.py|servo_node_main|joint_states_filter' || true); do \
      [ \"\$p\" = \"\$self\" ] && continue; kill -9 \"\$p\" 2>/dev/null || true; \
    done; \
    nohup env SERVO_CHECK_COLLISIONS='${servo_check_collisions}' SERVO_SELF_COLLISION_THRESHOLD='${servo_self_collision_threshold}' SERVO_SCENE_COLLISION_THRESHOLD='${servo_scene_collision_threshold}' SERVO_COLLISION_CHECK_RATE='${servo_collision_check_rate}' /opt/ros/humble/bin/ros2 launch servo_test_config servo_dual_gz.launch.py >/tmp/servo_dual_gz.log 2>&1 < /dev/null &"
  echo "[ok] Started dual-arm servo launch."
}

start_servo() {
  require_running
  ensure_ws_built
  local servo_check_collisions="${SERVO_CHECK_COLLISIONS:-}"
  local servo_self_collision_threshold="${SERVO_SELF_COLLISION_THRESHOLD:-}"
  local servo_scene_collision_threshold="${SERVO_SCENE_COLLISION_THRESHOLD:-}"
  local servo_collision_check_rate="${SERVO_COLLISION_CHECK_RATE:-}"
  dexec "source /opt/ros/humble/setup.bash && source /workspace/install/setup.bash && \
    self=\$\$; \
    for p in \$(pgrep -f 'servo_gz.launch.py|servo_node_main|joint_states_filter' || true); do \
      [ \"\$p\" = \"\$self\" ] && continue; kill -2 \"\$p\" 2>/dev/null || true; \
    done; \
    sleep 0.4; \
    for p in \$(pgrep -f 'servo_gz.launch.py|servo_node_main|joint_states_filter' || true); do \
      [ \"\$p\" = \"\$self\" ] && continue; kill -9 \"\$p\" 2>/dev/null || true; \
    done; \
    nohup env SERVO_CHECK_COLLISIONS='${servo_check_collisions}' SERVO_SELF_COLLISION_THRESHOLD='${servo_self_collision_threshold}' SERVO_SCENE_COLLISION_THRESHOLD='${servo_scene_collision_threshold}' SERVO_COLLISION_CHECK_RATE='${servo_collision_check_rate}' /opt/ros/humble/bin/ros2 launch servo_test_config servo_gz.launch.py >/tmp/servo_gz.log 2>&1 < /dev/null &"
  echo "[ok] Started servo launch."
}

start_dual_part23() {
  require_running
  ensure_ws_built
  local reduced="${REDUCED_PROCESS_MODE:-$(env_value REDUCED_PROCESS_MODE 1)}"
  local dual_gripper_controller="${DUAL_GRIPPER_CONTROLLER:-coupled}"
  local dual_gripper_node="simple_gripper_command_bridge"
  if [[ "${dual_gripper_controller}" == "coupled" ]]; then
    dual_gripper_node="coupled_gripper_controller"
  elif [[ "${dual_gripper_controller}" != "position" ]]; then
    echo "[warn] Unknown DUAL_GRIPPER_CONTROLLER=${dual_gripper_controller}; using coupled."
    dual_gripper_controller="coupled"
    dual_gripper_node="coupled_gripper_controller"
  fi
  local part23_pattern="hand_pose_mapper|servo_command_bridge|simple_gripper_command_bridge|coupled_gripper_controller|reset_manager|received_pose_to_target_twist|target_twist_to_servo_cmd|target_twist_to_gripper_cmd|coupled_hande_gripper_controller|target_twist_reset_manager"
  dexec "source /opt/ros/humble/setup.bash && source /workspace/install/setup.bash && \
    self=\$\$; \
    for p in \$(pgrep -f '${part23_pattern}' || true); do \
      [ \"\$p\" = \"\$self\" ] && continue; kill -2 \"\$p\" 2>/dev/null || true; \
    done; \
    sleep 0.4; \
    for p in \$(pgrep -f '${part23_pattern}' || true); do \
      [ \"\$p\" = \"\$self\" ] && continue; kill -9 \"\$p\" 2>/dev/null || true; \
    done; \
    if [ '${reduced}' != 1 ]; then nohup /opt/ros/humble/bin/ros2 run teleop_bridge hand_pose_mapper --ros-args -r __ns:=/left_arm --params-file ${DUAL_LEFT_TUNING_FILE} >/tmp/dual_left_mapper.log 2>&1 < /dev/null & fi; \
    nohup /opt/ros/humble/bin/ros2 run teleop_bridge servo_command_bridge --ros-args -r __ns:=/left_arm --params-file ${DUAL_LEFT_TUNING_FILE} >/tmp/dual_left_target_to_servo.log 2>&1 < /dev/null & \
    nohup /opt/ros/humble/bin/ros2 run teleop_bridge ${dual_gripper_node} --ros-args -r __ns:=/left_arm --params-file ${DUAL_LEFT_TUNING_FILE} >/tmp/dual_left_target_to_gripper.log 2>&1 < /dev/null & \
    nohup /opt/ros/humble/bin/ros2 run teleop_bridge reset_manager --ros-args -r __ns:=/left_arm --params-file ${DUAL_LEFT_TUNING_FILE} >/tmp/dual_left_reset_manager.log 2>&1 < /dev/null & \
    if [ '${reduced}' != 1 ]; then nohup /opt/ros/humble/bin/ros2 run teleop_bridge hand_pose_mapper --ros-args -r __ns:=/right_arm --params-file ${DUAL_RIGHT_TUNING_FILE} >/tmp/dual_right_mapper.log 2>&1 < /dev/null & fi; \
    nohup /opt/ros/humble/bin/ros2 run teleop_bridge servo_command_bridge --ros-args -r __ns:=/right_arm --params-file ${DUAL_RIGHT_TUNING_FILE} >/tmp/dual_right_target_to_servo.log 2>&1 < /dev/null & \
    nohup /opt/ros/humble/bin/ros2 run teleop_bridge ${dual_gripper_node} --ros-args -r __ns:=/right_arm --params-file ${DUAL_RIGHT_TUNING_FILE} >/tmp/dual_right_target_to_gripper.log 2>&1 < /dev/null & \
    nohup /opt/ros/humble/bin/ros2 run teleop_bridge reset_manager --ros-args -r __ns:=/right_arm --params-file ${DUAL_RIGHT_TUNING_FILE} --params-file /tmp/scene_right_reset.yaml >/tmp/dual_right_reset_manager.log 2>&1 < /dev/null &"
  echo "[ok] Started dual-arm Part2/Part3 mapper/bridges (gripper=${dual_gripper_controller})."
}

start_dual_part4() {
  require_running
  ensure_ws_built
  local dual_sync_rate_hz="${DUAL_SYNC_RATE_HZ:-60.0}"
  local dual_haptic_rate_hz="${DUAL_HAPTIC_RATE_HZ:-60.0}"
  local enable_haptics="${ENABLE_HAPTICS:-1}"
  if [[ "${REDUCED_PROCESS_MODE:-$(env_value REDUCED_PROCESS_MODE 1)}" == 1 ]]; then enable_haptics=0; fi
  local dual_haptic_enable_servo_gap="${DUAL_HAPTIC_ENABLE_SERVO_GAP:-true}"
  local part4_pattern="task_pose_sync_publisher|contact_haptic_publisher|rubik_task_controller|runtime_task_manager|cube_pose_sync_publisher|gazebo_contact_haptic_publisher|rubik2x2_mechanism_controller|task_manager|ros_tcp_endpoint.*endpoint.py|default_server_endpoint|ros_gz_bridge.*dynamic_pose/info"
  dexec "source /opt/ros/humble/setup.bash && source /workspace/install/setup.bash && \
    self=\$\$; \
    for p in \$(pgrep -f '${part4_pattern}' || true); do \
      [ \"\$p\" = \"\$self\" ] && continue; kill -2 \"\$p\" 2>/dev/null || true; \
    done; \
    sleep 0.6; \
    for p in \$(pgrep -f '${part4_pattern}' || true); do \
      [ \"\$p\" = \"\$self\" ] && continue; kill \"\$p\" 2>/dev/null || true; \
    done; \
    nohup /opt/ros/humble/bin/ros2 launch ros_tcp_endpoint endpoint.py >/tmp/dual_part4_tcp_endpoint.log 2>&1 < /dev/null & \
    nohup /opt/ros/humble/bin/ros2 run ros_gz_bridge parameter_bridge '/world/ur_hande_dual_arm_tabletop/dynamic_pose/info@tf2_msgs/msg/TFMessage[ignition.msgs.Pose_V' >/tmp/dual_part4_gz_tf_bridge.log 2>&1 < /dev/null & \
    nohup /opt/ros/humble/bin/ros2 run teleop_bridge runtime_task_manager --ros-args -p repo_root:=/workspace -p active_scene_profile_file:=profiles/active_scene_profile.txt -p world_output:=simulation/worlds/ur_hande_dual_arm_tabletop.sdf >/tmp/dual_part4_task_manager.log 2>&1 < /dev/null & \
    nohup /opt/ros/humble/bin/ros2 run teleop_bridge task_pose_sync_publisher --ros-args -p target_frame:=world -p gz_dynamic_pose_topic:=/world/ur_hande_dual_arm_tabletop/dynamic_pose/info -p scene_layout_sdf_path:=/workspace/simulation/worlds/ur_hande_dual_arm_tabletop.sdf -p publish_rate_hz:=${dual_sync_rate_hz} >/tmp/dual_part4_cube_pose.log 2>&1 < /dev/null & \
    if grep -q 'name=\"Sync_Rubik2x2_cubie_' /workspace/simulation/worlds/ur_hande_dual_arm_tabletop.sdf; then \
      nohup /opt/ros/humble/bin/ros2 run teleop_bridge rubik_task_controller --ros-args -p world_name:=ur_hande_dual_arm_tabletop -p scene_layout_sdf_path:=/workspace/simulation/worlds/ur_hande_dual_arm_tabletop.sdf -p root_name:=Sync_Rubik2x2 >/tmp/dual_part4_rubik.log 2>&1 < /dev/null & \
    else \
      printf '[info] Rubik task is not active; skipping rubik_task_controller.\n' >/tmp/dual_part4_rubik.log; \
    fi; \
    if [ '${enable_haptics}' = '1' ]; then \
      nohup /opt/ros/humble/bin/ros2 run teleop_bridge contact_haptic_publisher --ros-args -p gz_dynamic_pose_topic:=/world/ur_hande_dual_arm_tabletop/dynamic_pose/info -p scene_layout_sdf_path:=/workspace/simulation/worlds/ur_hande_dual_arm_tabletop.sdf -p publish_rate_hz:=${dual_haptic_rate_hz} -p pulse_amplitude:=0.55 -p continuous_contact_amplitude:=0.0 -p proximity_amplitude:=0.0 -p enable_servo_collision_gap_haptics:=${dual_haptic_enable_servo_gap} -p collision_gap_amplitude:=0.35 -p collision_gap_activation_threshold_m:=0.08 -p collision_gap_release_threshold_m:=0.045 -p collision_gap_hold_sec:=0.25 -p require_both_finger_contact_for_pulse:=true -p enable_empty_close_contact_pulse:=true >/tmp/dual_part4_haptics.log 2>&1 < /dev/null & \
    else \
      printf '[info] Haptics disabled by ENABLE_HAPTICS=0.\n' >/tmp/dual_part4_haptics.log; \
    fi"
  echo "[ok] Started dual-arm Part4 sync services (sync=${dual_sync_rate_hz} Hz, haptics=${enable_haptics}, haptic_rate=${dual_haptic_rate_hz} Hz)."
}

keyboard() {
  require_running
  ensure_ws_built
  dexec "self=\$\$; \
    for p in \$(pgrep -f 'servo_command_bridge|keyboard_servo_override|target_twist_to_servo_cmd|keyboard_servo_cmd' || true); do \
      [ \"\$p\" = \"\$self\" ] && continue; kill -2 \"\$p\" 2>/dev/null || true; \
    done; \
    sleep 0.4; \
    for p in \$(pgrep -f 'servo_command_bridge|keyboard_servo_override|target_twist_to_servo_cmd|keyboard_servo_cmd' || true); do \
      [ \"\$p\" = \"\$self\" ] && continue; kill -9 \"\$p\" 2>/dev/null || true; \
    done"
  echo "[info] Starting interactive keyboard controller."
  echo "[info] This temporarily disables headset-to-Servo output. Run start_part23 after quitting to restore headset control."
  echo "[info] Press x or Ctrl-C inside the keyboard controller to quit."
  docker exec -it "${CONTAINER}" bash -lc "source /opt/ros/humble/setup.bash && \
    source /workspace/install/setup.bash && \
    ros2 run teleop_bridge keyboard_servo_override --ros-args --params-file ${TELEOP_TUNING_FILE}"
}

right_gamepad_on() {
  require_running
  ensure_ws_built
  dexec "source /opt/ros/humble/setup.bash && source /workspace/install/setup.bash && \
    ros2 param set /right_arm/hand_pose_mapper allow_unity_control_mode_switch false && \
    ros2 param set /right_arm/hand_pose_mapper control_mode gamepad"
  echo "[ok] Right arm mapper set to GAMEPAD/thumbstick mode. Left arm headset control is unchanged."
}

right_gamepad_off() {
  require_running
  ensure_ws_built
  dexec "source /opt/ros/humble/setup.bash && source /workspace/install/setup.bash && \
    ros2 param set /right_arm/hand_pose_mapper allow_unity_control_mode_switch false && \
    ros2 param set /right_arm/hand_pose_mapper control_mode hand_pose"
  echo "[ok] Right arm mapper restored to HAND_POSE mode."
}

stop_right_keyboard_override_processes() {
  dexec "self=\$\$; \
    for p in \$(pgrep -f 'servo_command_bridge.*__ns:=/right_arm|target_twist_to_servo_cmd.*__ns:=/right_arm|keyboard_servo_override.*right_arm/servo_node/delta_twist_cmds|keyboard_servo_cmd.*right_arm/servo_node/delta_twist_cmds' || true); do \
      [ \"\$p\" = \"\$self\" ] && continue; kill -2 \"\$p\" 2>/dev/null || true; \
    done; \
    sleep 0.3; \
    for p in \$(pgrep -f 'servo_command_bridge.*__ns:=/right_arm|target_twist_to_servo_cmd.*__ns:=/right_arm|keyboard_servo_override.*right_arm/servo_node/delta_twist_cmds|keyboard_servo_cmd.*right_arm/servo_node/delta_twist_cmds' || true); do \
      [ \"\$p\" = \"\$self\" ] && continue; kill -9 \"\$p\" 2>/dev/null || true; \
    done"
}

start_right_target_to_servo_bridge() {
  dexec "source /opt/ros/humble/setup.bash && source /workspace/install/setup.bash && \
    nohup /opt/ros/humble/bin/ros2 run teleop_bridge servo_command_bridge --ros-args -r __ns:=/right_arm --params-file ${DUAL_RIGHT_TUNING_FILE} >/tmp/dual_right_target_to_servo.log 2>&1 < /dev/null &"
}

right_keyboard() {
  require_running
  ensure_ws_built
  stop_right_keyboard_override_processes
  echo "[info] Starting right-arm keyboard override."
  echo "[info] Right Quest controller packets still arrive, but the right target-to-Servo bridge is stopped while keyboard is active."
  echo "[info] Left arm headset control remains active. Press x or Ctrl-C inside the keyboard controller to quit."
  docker exec -it "${CONTAINER}" bash -lc "source /opt/ros/humble/setup.bash && \
    source /workspace/install/setup.bash && \
    ros2 run teleop_bridge keyboard_servo_override --ros-args \
      -p output_topic:=/right_arm/servo_node/delta_twist_cmds \
      -p frame_id:=right_base_link \
      -p start_servo_service:=/right_arm/servo_node/start_servo"
}

right_keyboard_off() {
  require_running
  ensure_ws_built
  stop_right_keyboard_override_processes
  start_right_target_to_servo_bridge
  right_gamepad_off
  echo "[ok] Right-arm keyboard override stopped and right headset hand-pose bridge restarted."
}

debug_hand_start() {
  require_running
  ensure_ws_built
  local arm="${DEBUG_ARM:-both}"
  local pattern="${DEBUG_PATTERN:-line_x}"
  local duration period amp_x amp_y amp_z yaw
  duration="$(float_arg "${DEBUG_DURATION_SEC:-0.0}")"
  period="$(float_arg "${DEBUG_PERIOD_SEC:-4.0}")"
  amp_x="$(float_arg "${DEBUG_AMPLITUDE_X:-0.16}")"
  amp_y="$(float_arg "${DEBUG_AMPLITUDE_Y:-0.0}")"
  amp_z="$(float_arg "${DEBUG_AMPLITUDE_Z:-0.0}")"
  yaw="$(float_arg "${DEBUG_WORKSPACE_YAW_DEG:-0.0}")"

  dexec "source /opt/ros/humble/setup.bash && source /workspace/install/setup.bash && \
    self=\$\$; \
    for p in \$(pgrep -f 'quest_controller_receiver|debug_hand_generator|dual_debug_hand_generator' || true); do \
      [ \"\$p\" = \"\$self\" ] && continue; kill -2 \"\$p\" 2>/dev/null || true; \
    done; \
    sleep 0.4; \
    for p in \$(pgrep -f 'quest_controller_receiver|debug_hand_generator|dual_debug_hand_generator' || true); do \
      [ \"\$p\" = \"\$self\" ] && continue; kill -9 \"\$p\" 2>/dev/null || true; \
    done; \
    nohup /opt/ros/humble/bin/ros2 run teleop_bridge debug_hand_generator --ros-args \
      -p arm:=${arm} \
      -p pattern:=${pattern} \
      -p duration_sec:=${duration} \
      -p motion_period_sec:=${period} \
      -p amplitude_x:=${amp_x} \
      -p amplitude_y:=${amp_y} \
      -p amplitude_z:=${amp_z} \
      -p workspace_yaw_deg:=${yaw} \
      >/tmp/debug_hand_generator.log 2>&1 < /dev/null &"
  echo "[ok] Started dual debug hand generator (log: /tmp/debug_hand_generator.log)."
  echo "     arm=${arm}, pattern=${pattern}, duration=${duration}, period=${period}, amp=(${amp_x},${amp_y},${amp_z}), workspace_yaw=${yaw}"
  echo "     quest_controller_receiver was stopped to avoid competing headset/stale messages."
  echo "     Run ./scripts/backend11_lifecycle.sh start_receiver to restore headset input."
}

debug_hand_stop() {
  require_running
  dexec "self=\$\$; \
    for p in \$(pgrep -f 'debug_hand_generator|dual_debug_hand_generator' || true); do \
      [ \"\$p\" = \"\$self\" ] && continue; kill -2 \"\$p\" 2>/dev/null || true; \
    done; \
    sleep 0.2; \
    for p in \$(pgrep -f 'debug_hand_generator|dual_debug_hand_generator' || true); do \
      [ \"\$p\" = \"\$self\" ] && continue; kill -9 \"\$p\" 2>/dev/null || true; \
    done"
  echo "[ok] Stopped dual debug hand generator."
}

debug_servo_sample() {
  require_running
  ensure_ws_built
  local duration
  duration="$(float_arg "${DEBUG_SAMPLE_DURATION_SEC:-10.0}")"
  local arms="${DEBUG_SAMPLE_ARMS:-left,right}"
  dexec "source /opt/ros/humble/setup.bash && source /workspace/install/setup.bash && \
    ros2 run teleop_bridge servo_response_sampler --ros-args \
      -p duration_sec:=${duration} \
      -p arms:=${arms}"
}

debug_servo_motion() {
  require_running
  ensure_ws_built
  local sample_duration
  sample_duration="$(float_arg "${DEBUG_SAMPLE_DURATION_SEC:-10.0}")"
  local generator_duration="${DEBUG_DURATION_SEC:-$(
    python3 - <<PY
print(float("${sample_duration}") + 2.0)
PY
  )}"
  DEBUG_DURATION_SEC="${generator_duration}" debug_hand_start
  sleep 1
  DEBUG_SAMPLE_DURATION_SEC="${sample_duration}" debug_servo_sample
  debug_hand_stop
}

debug_self_collision() {
  require_running
  ensure_ws_built
  dexec "source /opt/ros/humble/setup.bash && source /workspace/install/setup.bash && \
    ros2 launch servo_test_config self_collision_inspector_dual.launch.py"
}

bringup_all() {
  up_container
  prune_container_ros_logs "${KEEP_RUNTIME_LOGS}"
  ensure_ws_built
  start_sim
  sleep 2
  start_servo
  sleep 2
  start_receiver
  start_part23
  start_part4
  status
}

bringup_wired() {
  up_container
  wired_on
  prune_container_ros_logs "${KEEP_RUNTIME_LOGS}"
  ensure_ws_built
  start_sim
  sleep 2
  start_servo
  sleep 2
  start_receiver
  start_part23
  start_part4
  status
}

bringup_dual() {
  # Explicit opt-in for this host; never pause the active backend itself.
  if [[ "${PAUSE_LEGACY_BACKEND:-$(env_value PAUSE_LEGACY_BACKEND 0)}" == 1 && "${CONTAINER}" != motion_planner_11 ]]; then
    if [[ "$(docker inspect -f '{{.State.Running}} {{.State.Paused}}' motion_planner_11 2>/dev/null || true)" == 'true false' ]]; then
      docker pause motion_planner_11
    fi
  fi
  up_container
  maybe_wired_on
  prune_container_ros_logs "${KEEP_RUNTIME_LOGS}"
  ensure_ws_built
  start_dual_sim
  sleep 2
  start_dual_servo
  sleep 2
  start_receiver
  start_dual_part23
  start_dual_part4
  if [[ "${ENABLE_GPU:-$(env_value ENABLE_GPU 0)}" == "1" && "${ENABLE_VIEWER:-1}" == "1" ]]; then
    bash "${ROOT_DIR}/scripts/gpu_viewer.sh" start
  fi
  status
}

status() {
  docker ps --filter "name=^/${CONTAINER}$" --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
  if container_running; then
    dexec "echo '--- processes'; pgrep -fa '${TELEOP_PATTERN}' || true"
    dexec "echo '--- tcp listen'; netstat -ant 2>/dev/null | grep LISTEN | grep ':5005 ' || true"
    dexec "echo '--- recent logs'; \
      for f in /tmp/qcr.log /tmp/part2_mapper.log /tmp/part3_target_to_servo.log /tmp/part3_target_to_gripper.log /tmp/part3_reset_manager.log /tmp/part4_tcp_endpoint.log /tmp/part4_gz_tf_bridge.log /tmp/part4_cube_pose.log /tmp/run_tabletop_sim.log /tmp/run_dual_arm_tabletop_sim.log /tmp/gz_clock_bridge.log /tmp/gz_dual_arm_clock_bridge.log /tmp/servo_gz.log /tmp/servo_dual_gz.log /tmp/dual_left_mapper.log /tmp/dual_left_target_to_servo.log /tmp/dual_left_target_to_gripper.log /tmp/dual_left_reset_manager.log /tmp/dual_right_mapper.log /tmp/dual_right_target_to_servo.log /tmp/dual_right_target_to_gripper.log /tmp/dual_right_reset_manager.log /tmp/dual_part4_tcp_endpoint.log /tmp/dual_part4_gz_tf_bridge.log /tmp/dual_part4_task_manager.log /tmp/dual_part4_cube_pose.log /tmp/dual_part4_rubik.log /tmp/dual_part4_haptics.log; do \
        echo '===== ' \$f; [ -f \$f ] && tail -n 8 \$f || echo missing; \
      done"
  fi
}

usage() {
  cat <<'EOH'
Usage:
  ./scripts/backend11_lifecycle.sh mode_wired
  ./scripts/backend11_lifecycle.sh mode_wireless
  ./scripts/backend11_lifecycle.sh mode_status
  ./scripts/backend11_lifecycle.sh safe_down
  ./scripts/backend11_lifecycle.sh up_container
  ./scripts/backend11_lifecycle.sh up_container_build
  ./scripts/backend11_lifecycle.sh restart_container
  ./scripts/backend11_lifecycle.sh wired_on
  ./scripts/backend11_lifecycle.sh wired_off
  ./scripts/backend11_lifecycle.sh wired_status
  ./scripts/backend11_lifecycle.sh build_ws
  ./scripts/backend11_lifecycle.sh clean_build_ws
  ./scripts/backend11_lifecycle.sh select_task cup_arrangement
  ./scripts/backend11_lifecycle.sh randomize_cups 1
  ./scripts/backend11_lifecycle.sh cup_debug
  ./scripts/backend11_lifecycle.sh build_receiver
  ./scripts/backend11_lifecycle.sh cleanup_runtime
  ./scripts/backend11_lifecycle.sh bringup_all
  ./scripts/backend11_lifecycle.sh bringup_wired
  ./scripts/backend11_lifecycle.sh bringup_dual
  ./scripts/backend11_lifecycle.sh stop_nodes
  ./scripts/backend11_lifecycle.sh start_receiver
  ./scripts/backend11_lifecycle.sh start_part23
  ./scripts/backend11_lifecycle.sh start_dual_part23
  ./scripts/backend11_lifecycle.sh start_part4
  ./scripts/backend11_lifecycle.sh start_dual_part4
  ./scripts/backend11_lifecycle.sh start_sim
  ./scripts/backend11_lifecycle.sh generate_dual_world
  ./scripts/backend11_lifecycle.sh list_tasks
  ./scripts/backend11_lifecycle.sh select_task pick_place_basic
  ./scripts/backend11_lifecycle.sh select_task rubik_2x2
  ./scripts/backend11_lifecycle.sh select_task cable_insertion
  DUAL_SCENE_PROFILE=profiles/scenes/dual_arm_cable_insertion/scene.yaml ./scripts/backend11_lifecycle.sh generate_dual_world
  ./scripts/backend11_lifecycle.sh start_dual_sim
  ./scripts/backend11_lifecycle.sh start_servo
  ./scripts/backend11_lifecycle.sh start_dual_servo
  ./scripts/backend11_lifecycle.sh keyboard
  ./scripts/backend11_lifecycle.sh right_gamepad_on
  ./scripts/backend11_lifecycle.sh right_gamepad_off
  ./scripts/backend11_lifecycle.sh right_keyboard
  ./scripts/backend11_lifecycle.sh right_keyboard_off
  ./scripts/backend11_lifecycle.sh debug_hand_start
  ./scripts/backend11_lifecycle.sh debug_hand_stop
  ./scripts/backend11_lifecycle.sh debug_servo_sample
  ./scripts/backend11_lifecycle.sh debug_servo_motion
  ./scripts/backend11_lifecycle.sh debug_self_collision
  ./scripts/backend11_lifecycle.sh status
  ./scripts/log_tools/tail_backend_logs.sh
  ./scripts/log_tools/watch_topic_rates.sh
  python3 scripts/test_tools/eval_scripts/03_ros_topic_rate_audit.py

Debug options:
  DEBUG_ARM=left|right|both
  DEBUG_PATTERN=line_x|line_y|line_z|circle_xy|circle_xz
  DEBUG_DURATION_SEC=12
  DEBUG_SAMPLE_DURATION_SEC=10
  DEBUG_AMPLITUDE_X=0.16 DEBUG_AMPLITUDE_Y=0.0 DEBUG_AMPLITUDE_Z=0.0
EOH
}

cmd="${1:-}"
case "${cmd}" in
  mode_wired) mode_wired ;;
  mode_wireless) mode_wireless ;;
  mode_status) mode_status ;;
  safe_down) safe_down ;;
  up_container) up_container ;;
  up_container_build) up_container_build ;;
  restart_container) safe_down; up_container ;;
  wired_on) wired_on ;;
  wired_off) wired_off ;;
  wired_status) wired_status ;;
  build_ws) build_ws ;;
  clean_build_ws)
    require_running
    dexec "cd /workspace && rm -rf build install log"
    build_ws ;;
  randomize_cups|cup_debug)
    require_running
    action=randomize; [[ "$cmd" == cup_debug ]] && action=debug
    seed="${2:-0}"
    [[ "$seed" =~ ^[0-9]+$ ]] || { echo "seed must be a nonnegative integer"; exit 2; }
    run_id="${BASHPID}_${RANDOM}"
    # Docker exec does not proxy host signals without a TTY. Forward only to
    # this invocation, then wait for its zero-command / publisher cleanup.
    docker exec "$CONTAINER" bash -lc "source /opt/ros/humble/setup.bash && source /workspace/install/setup.bash && exec python3 /workspace/scripts/cup_task.py $action --seed $seed --run-id $run_id" &
    cup_job=$!
    trap 'docker exec "$CONTAINER" pkill -TERM -f "^python3 /workspace/scripts/cup_task.py .* --run-id ${run_id}$" >/dev/null 2>&1 || true; wait "$cup_job" || true; exit 130' INT TERM
    wait "$cup_job"
    trap - INT TERM ;;


  build_receiver) build_receiver ;;
  cleanup_runtime) cleanup_runtime ;;
  bringup_all) bringup_all ;;
  bringup_wired) bringup_wired ;;
  bringup_dual) bringup_dual ;;
  stop_nodes) stop_nodes ;;
  start_receiver) start_receiver ;;
  start_part23) start_part23 ;;
  start_dual_part23) start_dual_part23 ;;
  start_part4) start_part4 ;;
  start_dual_part4) start_dual_part4 ;;
  start_sim) start_sim ;;
  generate_dual_world) generate_dual_world_from_profiles ;;
  list_tasks) list_tasks ;;
  select_task) select_task "${2:-}" ;;
  start_dual_sim) start_dual_sim ;;
  start_servo) start_servo ;;
  start_dual_servo) start_dual_servo ;;
  keyboard) keyboard ;;
  right_gamepad_on) right_gamepad_on ;;
  right_gamepad_off) right_gamepad_off ;;
  right_keyboard) right_keyboard ;;
  right_keyboard_off) right_keyboard_off ;;
  debug_hand_start) debug_hand_start ;;
  debug_hand_stop) debug_hand_stop ;;
  debug_servo_sample) debug_servo_sample ;;
  debug_servo_motion) debug_servo_motion ;;
  debug_self_collision) debug_self_collision ;;
  status) status ;;
  *) usage; exit 1 ;;
esac
