#!/bin/bash
set -e

source /opt/ros/humble/setup.bash

WS_DIR="/workspace"
OVERLAY_SETUP="${WS_DIR}/install/setup.bash"

# Auto-build the colcon overlay once per container (when install/ doesn't exist).
# Disable by setting AUTO_COLCON_BUILD=0.
if [ "${AUTO_COLCON_BUILD:-1}" != "0" ] && [ ! -f "${OVERLAY_SETUP}" ]; then
  if [ -d "${WS_DIR}/src" ] && [ "$(ls -A "${WS_DIR}/src" 2>/dev/null || true)" != "" ]; then
    echo "[entrypoint] ${OVERLAY_SETUP} missing; running: colcon build --symlink-install"
    cd "${WS_DIR}"
    colcon build --symlink-install
    echo "[entrypoint] colcon build finished"
  else
    echo "[entrypoint] ${WS_DIR}/src is empty; skipping colcon build"
  fi
fi

if [ -f "${OVERLAY_SETUP}" ]; then
  source "${OVERLAY_SETUP}"
fi

exec "$@"
