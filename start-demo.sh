#!/usr/bin/env bash
# Start candytron_mcp (simulated) + mcpclient_speech_face.
# The client has the SSE URL hardcoded to 127.0.0.1:7999,
# so we must start the server on --port 7999.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

SERVER_DIR="$ROOT/candytron_mcp"
CLIENT_DIR="$ROOT/mcpclient_speech"
PORT=7999

cleanup() {
    if [[ -n "${CLIENT_PID:-}" ]] && kill -0 "$CLIENT_PID" 2>/dev/null; then
        echo "[start-demo] stopping client (pid $CLIENT_PID)"
        kill "$CLIENT_PID" 2>/dev/null || true
        wait "$CLIENT_PID" 2>/dev/null || true
    fi
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[start-demo] stopping candytron_mcp (pid $SERVER_PID)"
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo "[start-demo] starting candytron_mcp on :$PORT (simulated)"
(cd "$SERVER_DIR" && uv run candytron_mcp.py \
    --simulate-robot --simulate-camera --port "$PORT") &
SERVER_PID=$!

# Wait for the SSE port to accept connections
echo "[start-demo] waiting for server on 127.0.0.1:$PORT ..."
for i in {1..40}; do
    if nc -z 127.0.0.1 "$PORT" 2>/dev/null; then
        echo "[start-demo] server up"
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[start-demo] server exited before becoming ready" >&2
        exit 1
    fi
    sleep 0.25
done

echo "[start-demo] starting mcpclient_speech_face"
(cd "$CLIENT_DIR" && uv run mcpclient_speech_face.py "$@") &
CLIENT_PID=$!

# Foreground-wait so the EXIT/INT/TERM trap runs and kills the server.
# If the client exits first, stop the server; if Ctrl-C arrives, the
# trap kills both.
wait "$CLIENT_PID"
