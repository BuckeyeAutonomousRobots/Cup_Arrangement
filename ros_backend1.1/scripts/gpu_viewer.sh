#!/usr/bin/env bash
# Viewer only: never starts or stops the Gazebo server.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER="${CONTAINER:-$(sed -n 's/^CONTAINER=//p' "${ROOT_DIR}/.env" | tail -1)}"
case "${1:-start}" in
  start)
    docker exec "${CONTAINER}" bash -lc '
      if [[ -f /tmp/gpu_viewer.pid ]] && kill -0 "$(cat /tmp/gpu_viewer.pid)" 2>/dev/null; then exit 0; fi
      source /workspace/install/setup.bash
      export IGN_GAZEBO_RESOURCE_PATH="/workspace/install/robotiq_hande_description/share:/workspace/install/ur_hande_description/share:/opt/ros/humble/share"
      # Xvfb is a software-rendered inspection desktop. Server camera stays EGL/NVIDIA.
      nohup env LIBGL_ALWAYS_SOFTWARE=1 __GLX_VENDOR_LIBRARY_NAME=mesa ign gazebo -g --render-engine-gui ogre2 --gui-config /workspace/simulation/config/gpu_viewer.config >/tmp/gpu_viewer.log 2>&1 </dev/null &
      echo $! >/tmp/gpu_viewer.pid
    '
    ;;
  stop)
    docker exec "${CONTAINER}" bash -lc 'if [[ -f /tmp/gpu_viewer.pid ]]; then kill -TERM "$(cat /tmp/gpu_viewer.pid)" 2>/dev/null || true; fi'
    ;;
  *) echo "Usage: $0 [start|stop]" >&2; exit 2 ;;
esac
