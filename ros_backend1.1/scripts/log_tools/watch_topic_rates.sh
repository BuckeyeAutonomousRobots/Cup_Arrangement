#!/usr/bin/env bash
set -euo pipefail

CONTAINER="${CONTAINER:-motion_planner_11}"
ROS_ENV="${ROS_ENV:-source /opt/ros/humble/setup.bash && source /workspace/install/setup.bash}"
WINDOW_SEC="${WINDOW_SEC:-5}"

topics=(
  /left_arm/received_pose_states
  /right_arm/received_pose_states
  /left_arm/target_twist_states
  /right_arm/target_twist_states
  /joint_states
  /unity_sync/Sync_RedCube_pose
)

if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER}"; then
  echo "[error] Container ${CONTAINER} is not running." >&2
  exit 1
fi

for topic in "${topics[@]}"; do
  echo "===== ${topic}"
  docker exec "${CONTAINER}" bash -lc "${ROS_ENV} && timeout '${WINDOW_SEC}' ros2 topic hz '${topic}' || true"
done
