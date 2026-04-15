#!/usr/bin/env bash
# Kill any running candytron_mcp.py / mcpclient_speech_face.py processes.

set -uo pipefail

kill_matching() {
    local pattern="$1"
    # -f to match the full command line (we run via `uv run ...`)
    local pids
    pids=$(pgrep -f "$pattern" || true)
    if [[ -z "$pids" ]]; then
        echo "[stop-demo] no processes matching $pattern"
        return
    fi
    echo "[stop-demo] killing $pattern: $pids"
    # shellcheck disable=SC2086
    kill $pids 2>/dev/null || true
    sleep 1
    # shellcheck disable=SC2086
    kill -9 $pids 2>/dev/null || true
}

kill_matching 'candytron_mcp.py'
kill_matching 'mcpclient_speech_face.py'
echo "[stop-demo] done"
