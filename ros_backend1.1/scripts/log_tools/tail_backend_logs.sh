#!/usr/bin/env bash
set -euo pipefail

CONTAINER="${CONTAINER:-motion_planner_11}"
LINES="${LINES:-80}"

if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER}"; then
  echo "[error] Container ${CONTAINER} is not running." >&2
  exit 1
fi

logs=(
  /tmp/qcr.log
  /tmp/run_dual_arm_tabletop_sim.log
  /tmp/servo_dual_gz.log
  /tmp/dual_left_mapper.log
  /tmp/dual_left_target_to_servo.log
  /tmp/dual_left_target_to_gripper.log
  /tmp/dual_left_reset_manager.log
  /tmp/dual_right_mapper.log
  /tmp/dual_right_target_to_servo.log
  /tmp/dual_right_target_to_gripper.log
  /tmp/dual_right_reset_manager.log
  /tmp/dual_part4_tcp_endpoint.log
  /tmp/dual_part4_task_manager.log
  /tmp/dual_part4_cube_pose.log
  /tmp/dual_part4_haptics.log
)

for log in "${logs[@]}"; do
  echo "===== ${log}"
  docker exec "${CONTAINER}" bash -lc "[ -f '${log}' ] && tail -n '${LINES}' '${log}' || echo missing"
done
