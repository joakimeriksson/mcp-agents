#!/usr/bin/env bash
# Kill any running candytron_mcp.py / mcpclient_speech_face.py / face agent processes.
# Add --all to also stop a running kokoro-voice-server, --purge to delete the local voice store.

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
kill_matching 'face/agent.py\|agent.py --llm-model'
if [[ "${1:-}" == "--all" ]]; then
    kill_matching 'voice_server.py'
fi
if [[ "${1:-}" == "--purge" || "${2:-}" == "--purge" ]]; then
    # Forget the opt-in voice store (persist_named). Face enrolments live in
    # face/known_faces/ and are left alone; delete that directory by hand.
    rm -rf "$(dirname "$0")/mcpclient_speech/known_voices" && echo "[stop-demo] purged mcpclient_speech/known_voices"
fi
echo "[stop-demo] done"
