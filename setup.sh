#!/usr/bin/env bash
# One-time (idempotent) setup for the full CandyTron 4000 stack on a Mac:
#   - uv, portaudio (brew), Ollama + the gemma4 model
#   - the three uv projects: candytron_mcp, mcpclient_speech, face
#   - Piper TTS voices (offline fallback)
#   - kokoro-voice-server (Kokoro voices incl. Swedish) as a sibling checkout
#   - a check for the YOLO candy-detection model the real camera needs
#
# Re-run any time; every step is a no-op when already done.
# Then: ./start-candytron.sh --sim     (simulated)   or
#       ./start-candytron.sh --robot-ip 10.10.10.10  (real robot)

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
VOICE_DIR="${KOKORO_VOICE_SERVER_DIR:-$ROOT/../kokoro-voice-server}"
OLLAMA_MODEL="${OLLAMA_MODEL:-gemma4:latest}"

step() { echo; echo "==> $*"; }
have() { command -v "$1" >/dev/null 2>&1; }

step "Tools"
if ! have uv; then
    echo "uv not found — installing via Homebrew"
    brew install uv
fi
if ! have ollama; then
    echo "Ollama not found. Install it from https://ollama.com (or: brew install ollama), then re-run." >&2
    exit 1
fi
# pyaudio (a mcpclient_speech dependency) needs the portaudio headers to build
if have brew && ! brew list --versions portaudio >/dev/null 2>&1; then
    echo "installing portaudio (needed to build pyaudio)"
    brew install portaudio
fi
echo "uv: $(uv --version) | ollama: $(ollama --version 2>/dev/null | head -1)"

step "Ollama model $OLLAMA_MODEL (audio-capable; used for chat AND speech-to-text)"
if ! curl -sf -m 3 http://localhost:11434/api/tags >/dev/null; then
    echo "Ollama is not running — starting it in the background (ollama serve)"
    nohup ollama serve >/tmp/candytron-ollama.log 2>&1 &
    for _ in $(seq 1 30); do curl -sf -m 2 http://localhost:11434/api/tags >/dev/null && break; sleep 1; done
fi
if ollama list | awk '{print $1}' | grep -qx "$OLLAMA_MODEL"; then
    echo "already pulled"
else
    ollama pull "$OLLAMA_MODEL"
fi

step "Python environments (uv sync) — first run downloads torch etc., takes a while"
for d in candytron_mcp mcpclient_speech face; do
    echo "-- $d"
    (cd "$ROOT/$d" && uv sync)
done

step "Piper TTS voices (offline fallback for every language) -> face/piper_models/"
(cd "$ROOT/face" && uv run download_models.py)

step "kokoro-voice-server (Kokoro voices: Swedish Stina & friends, en/fr/es/it) at $VOICE_DIR"
if [[ ! -f "$VOICE_DIR/voice_server.py" ]]; then
    git clone https://github.com/joakimeriksson/kokoro-voice-server "$VOICE_DIR"
fi
(cd "$VOICE_DIR" && uv sync)
echo "Model weights (~300 MB) download from HuggingFace on the server's first start."

step "Face recognition model (InsightFace buffalo_l, ~300 MB, one-time download)"
(cd "$ROOT/face" && uv run python -c "
from insightface.app import FaceAnalysis
FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']).prepare(ctx_id=-1, det_size=(640, 640))
print('ready')" 2>/dev/null | tail -1)

step "Candy-detection model for the real table camera"
YOLO_SRC="${CANDY_YOLO_WEIGHTS:-$ROOT/../candytron-demo/systems/detector/models/best-m.pt}"
if [[ -f "$ROOT/candytron_mcp/models/best-m.pt" ]]; then
    echo "found candytron_mcp/models/best-m.pt"
elif [[ -f "$YOLO_SRC" ]]; then
    mkdir -p "$ROOT/candytron_mcp/models" && cp "$YOLO_SRC" "$ROOT/candytron_mcp/models/best-m.pt"
    echo "copied from $YOLO_SRC"
else
    cat <<MSG
MISSING: candytron_mcp/models/best-m.pt (YOLO weights trained on the candy set;
not in git). Copy it from the machine that trained it (candytron-demo/systems/detector/models/,
or set CANDY_YOLO_WEIGHTS=/path/to/best-m.pt)
before running with the real camera. Simulated mode (--sim) does not need it.
MSG
fi

echo
echo "Setup complete."
echo "  Simulated demo:  ./start-candytron.sh --sim"
echo "  Real robot:      ./start-candytron.sh --robot-ip 10.10.10.10 [--robot-camera N] [--camera M]"
