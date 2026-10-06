#!/usr/bin/env bash
set -euo pipefail

set +u
source /opt/ros/humble/setup.bash
set -u

TOPIC="/world/ur_hande_tabletop/stats"

read_rtf() {
  timeout 8 ign topic -e -n 1 -t "${TOPIC}" | awk '/real_time_factor:/ {print $2; exit}'
}

rtf_initial="$(read_rtf || true)"
sleep 15
rtf_15s="$(read_rtf || true)"

echo "RTF initial: ${rtf_initial:-NA}"
echo "RTF +15s:   ${rtf_15s:-NA}"
