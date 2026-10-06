#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

DURATION="${DURATION:-60}"
INTERVAL="${INTERVAL:-1}"
WARMUP="${WARMUP:-8}"
FAKE_ARM="${FAKE_ARM:-both}"
FAKE_PATTERN="${FAKE_PATTERN:-line_y}"
FAKE_PERIOD="${FAKE_PERIOD:-8}"
FAKE_AMPLITUDE_X="${FAKE_AMPLITUDE_X:-0.0}"
FAKE_AMPLITUDE_Y="${FAKE_AMPLITUDE_Y:-0.55}"
FAKE_AMPLITUDE_Z="${FAKE_AMPLITUDE_Z:-0.0}"

cd "${BACKEND_ROOT}"

python3 scripts/test_tools/eval_scripts/13_dynamic_novnc_headed_performance_test.py \
  --duration "${DURATION}" \
  --interval "${INTERVAL}" \
  --warmup "${WARMUP}" \
  --fake-arm "${FAKE_ARM}" \
  --fake-pattern "${FAKE_PATTERN}" \
  --fake-period "${FAKE_PERIOD}" \
  --fake-amplitude-x "${FAKE_AMPLITUDE_X}" \
  --fake-amplitude-y "${FAKE_AMPLITUDE_Y}" \
  --fake-amplitude-z "${FAKE_AMPLITUDE_Z}" \
  "$@"
