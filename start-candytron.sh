#!/usr/bin/env bash
# Start the full CandyTron 4000 stack and connect to the robot:
#   1. kokoro-voice-server on :8880 (skipped if one is already running)
#   2. candytron_mcp on :7999 — real Niryo Ned 2 + table camera, or simulated
#   3. mcpclient_speech_face (the CandyTron eye + face/speech client)
#
# Usage:
#   ./start-candytron.sh --sim                         # simulated robot + camera
#   ./start-candytron.sh --robot-ip 10.10.10.10        # real robot
#   ./start-candytron.sh --robot-ip 10.10.10.10 --robot-camera 1 --camera 0
#
# Options (everything else is passed through to the client, e.g. --camera N,
# --mic N, --llm-model, --debug-audio):
#   --sim                   simulate robot AND camera (no hardware)
#   --simulate-robot        simulate only the arm (real table camera)
#   --simulate-camera       simulate only the camera (real arm)
#   --robot-ip IP           Niryo Ned 2 address (default $NIRYO_IP or 10.10.10.10)
#   --robot-camera N        camera index for the TABLE camera (candytron_mcp)
#   --no-voice-server       don't start kokoro-voice-server (Piper only)
#   --face-agent            use face/agent.py (person memory, direct-audio capable)
#                           instead of mcpclient_speech_face as the client
#
# Run ./setup.sh once first. Ctrl-C stops everything this script started
# (a voice server that was already running is left alone).

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
VOICE_DIR="${KOKORO_VOICE_SERVER_DIR:-$ROOT/../kokoro-voice-server}"
VOICE_PORT="${VOICE_PORT:-8880}"
VOICE_LANGS="${VOICE_LANGS:-sv,en,fr,es,it}"
PORT=7999
LOG_DIR="${CANDYTRON_LOG_DIR:-/tmp}"
# Session transcript (JSONL) of every turn, tool call and VAD capture. On by
# default: after the first fair we had no record of what actually happened.
SESSION_LOG_DIR="${CANDYTRON_SESSION_LOG_DIR:-$ROOT/mcpclient_speech/logs}"

SERVER_ARGS=()
CLIENT_ARGS=()
START_VOICE=1
CLIENT=speech
while [[ $# -gt 0 ]]; do
    case "$1" in
        --sim)              SERVER_ARGS+=(--simulate-robot --simulate-camera); shift ;;
        --simulate-robot)   SERVER_ARGS+=(--simulate-robot); shift ;;
        --simulate-camera)  SERVER_ARGS+=(--simulate-camera); shift ;;
        --robot-ip)         SERVER_ARGS+=(--robot-ip "$2"); shift 2 ;;
        --robot-camera)     SERVER_ARGS+=(--camera "$2"); shift 2 ;;
        --no-voice-server)  START_VOICE=0; shift ;;
        --face-agent)       CLIENT=face; shift ;;
        -h|--help)          sed -n '2,25p' "$0"; exit 0 ;;
        *)                  CLIENT_ARGS+=("$1"); shift ;;
    esac
done

STARTED=()
cleanup() {
    echo
    for pid in "${STARTED[@]:-}"; do
        [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null && { echo "[candytron] stopping pid $pid"; kill "$pid" 2>/dev/null || true; }
    done
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

up() { curl -sf -m 2 -o /dev/null "$1"; }
wait_for() {  # url name seconds
    local i
    for ((i = 0; i < $3; i++)); do
        up "$1" && { echo "[candytron] $2 up"; return 0; }
        sleep 1
    done
    echo "[candytron] $2 did not come up in $3s" >&2
    return 1
}

# ---------- 0. Ollama ----------
if ! up http://localhost:11434/api/tags; then
    echo "[candytron] Ollama not running — starting"
    nohup ollama serve >"$LOG_DIR/candytron-ollama.log" 2>&1 &
    wait_for http://localhost:11434/api/tags "ollama" 30
fi

# ---------- 1. Voice server ----------
if [[ $START_VOICE -eq 1 ]]; then
    if up "http://127.0.0.1:$VOICE_PORT/health"; then
        echo "[candytron] voice server already running on :$VOICE_PORT (leaving it)"
        if ! curl -sf -m 2 "http://127.0.0.1:$VOICE_PORT/health" | grep -q '"speaker":true'; then
            echo "[candytron] note: that server was started without --speaker ecapa; voice id will run fail-open (off)" >&2
        fi
    elif [[ -f "$VOICE_DIR/voice_server.py" && -d "$VOICE_DIR/.venv" ]]; then
        echo "[candytron] starting kokoro-voice-server on :$VOICE_PORT (log: $LOG_DIR/candytron-voice-server.log)"
        (cd "$VOICE_DIR" && PYTORCH_ENABLE_MPS_FALLBACK=1 exec uv run python voice_server.py \
            --engine kokoro-svml --voice Greta --port "$VOICE_PORT" --langs "$VOICE_LANGS" --whisper base --speaker ecapa) \
            >"$LOG_DIR/candytron-voice-server.log" 2>&1 &
        STARTED+=($!)
        # cold start loads Kokoro + Whisper: allow a few minutes
        wait_for "http://127.0.0.1:$VOICE_PORT/health" "voice server" 240 || {
            echo "[candytron] continuing with Piper voices only" >&2
        }
    else
        echo "[candytron] kokoro-voice-server not set up at $VOICE_DIR (run ./setup.sh) — Piper voices only" >&2
    fi
fi

# ---------- 2. candytron_mcp ----------
if nc -z 127.0.0.1 "$PORT" 2>/dev/null; then
    echo "[candytron] something already listens on :$PORT — stop it first (./stop-demo.sh)" >&2
    exit 1
fi
echo "[candytron] starting candytron_mcp on :$PORT ${SERVER_ARGS[*]:-(real robot + camera)}"
(cd "$ROOT/candytron_mcp" && exec uv run candytron_mcp.py --port "$PORT" "${SERVER_ARGS[@]}") &
SERVER_PID=$!
STARTED+=("$SERVER_PID")
for _ in $(seq 1 240); do   # real camera calibration can take a while
    nc -z 127.0.0.1 "$PORT" 2>/dev/null && break
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[candytron] candytron_mcp exited before becoming ready (robot/camera problem?)" >&2
        exit 1
    fi
    sleep 0.5
done
nc -z 127.0.0.1 "$PORT" 2>/dev/null || { echo "[candytron] candytron_mcp never opened :$PORT" >&2; exit 1; }
echo "[candytron] candytron_mcp up"

# ---------- 3. Client ----------
if [[ $CLIENT == face ]]; then
    echo "[candytron] starting face agent (--service-server)"
    (cd "$ROOT/face" && exec uv run agent.py --llm-model gemma4:latest \
        --service-server "http://127.0.0.1:$PORT/sse" "${CLIENT_ARGS[@]}") &
else
    echo "[candytron] starting mcpclient_speech_face"
    (cd "$ROOT/mcpclient_speech" && exec uv run mcpclient_speech_face.py \
        --server "http://127.0.0.1:$PORT/sse" --log-dir "$SESSION_LOG_DIR" \
        "${CLIENT_ARGS[@]}") &
fi
CLIENT_PID=$!
STARTED+=("$CLIENT_PID")

# Foreground-wait on the client; the trap stops the rest.
wait "$CLIENT_PID"
