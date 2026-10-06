#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "$0")/backend11_lifecycle.sh" randomize_cups "${1:-$(date +%s)}"
