#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/.env"
LIFECYCLE="${ROOT_DIR}/scripts/backend11_lifecycle.sh"

usage() {
  cat <<'EOH'
Usage:
  ./scripts/bump_udp_port.sh
  ./scripts/bump_udp_port.sh <new_port>

Behavior:
  - No argument: increments QUEST_TCP_HOST_PORT by +1
  - With <new_port>: sets QUEST_TCP_HOST_PORT to that value
  - Then runs:
      1) restart_container
      2) build_ws
      3) start_receiver
      4) status
EOH
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "[error] Missing ${ENV_FILE}"
  exit 1
fi

current_port="$({ awk -F= '/^QUEST_TCP_HOST_PORT=/{print $2}' "${ENV_FILE}" | tail -n 1; } )"
if [[ -z "${current_port}" ]]; then
  current_port="$({ awk -F= '/^QUEST_UDP_HOST_PORT=/{print $2}' "${ENV_FILE}" | tail -n 1; } )"
fi
if [[ -z "${current_port}" ]]; then
  current_port=5031
fi

if [[ $# -gt 1 ]]; then
  usage
  exit 1
fi

if [[ $# -eq 1 ]]; then
  next_port="${1}"
else
  next_port="$((current_port + 1))"
fi

if ! [[ "${next_port}" =~ ^[0-9]+$ ]]; then
  echo "[error] Port must be an integer: ${next_port}"
  exit 1
fi

if (( next_port < 1024 || next_port > 65535 )); then
  echo "[error] Port out of range (1024-65535): ${next_port}"
  exit 1
fi

tmp_file="$(mktemp)"
awk -F= -v new_port="${next_port}" '
BEGIN { set = 0 }
$1 == "QUEST_TCP_HOST_PORT" {
  print "QUEST_TCP_HOST_PORT=" new_port
  set = 1
  next
}
$1 == "QUEST_UDP_HOST_PORT" { next }
{ print $0 }
END {
  if (set == 0) {
    print "QUEST_TCP_HOST_PORT=" new_port
  }
}
' "${ENV_FILE}" > "${tmp_file}"
mv "${tmp_file}" "${ENV_FILE}"

echo "[ok] Updated QUEST_TCP_HOST_PORT: ${current_port} -> ${next_port}"

cd "${ROOT_DIR}"
"${LIFECYCLE}" restart_container
"${LIFECYCLE}" build_ws
"${LIFECYCLE}" start_receiver
"${LIFECYCLE}" status

echo
echo "Next step (Unity wired mode): set HandPoseSender.targetIP = 127.0.0.1 and targetPort = ${next_port}, then run ./scripts/backend11_lifecycle.sh wired_on and rebuild APK."
