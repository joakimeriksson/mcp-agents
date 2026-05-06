# Generic speech-to-speech LLM-based MCP client

An MCP client that connects to any FastMCP server (over SSE) and exposes
its tools through voice interaction with a local LLM. Two variants:

- **`mcpclient_speech.py`** — push-to-talk. Hold Ctrl (or click the eye)
  to record, release to send.
- **`mcpclient_speech_face.py`** — face-triggered, continuous. A camera
  detects when someone is in focus and the system listens until VAD
  closes. Maintains per-person memory (name, language, preferences).

## Prerequisites

- **Ollama** with a tool-capable model pulled (default
  `PetrosStav/gemma3-tools:12b`).
- **whisper.cpp** built locally — used by `record.py` for STT in the
  push-to-talk client. Path is hardcoded at the top of `record.py`
  (`whisperdir`); model is `ggml-medium.bin`.
- **piper** with voices for `en`, `sv`, `de`, `fr`, `es`. Paths are in
  `piperscript`.
- **`../face/`** sibling directory — the face client imports
  `VoiceInput`, `VoiceOutput`, `FaceTracker` from there.
- A running FastMCP server on SSE (default `http://127.0.0.1:8000/sse`).

## Configuration

Settings are read from `config.toml` (next to the scripts) on startup;
CLI arguments override anything in the config, and the config overrides
the built-in defaults in `config.py`. Sections:

```toml
[llm]
model    = "PetrosStav/gemma3-tools:12b"
base_url = "http://localhost:11434/v1/"
api_key  = "ollama"

[face]
omit_names_and_prefs = false   # set true to skip name/preference distillation

[devices]
# microphone = 5   # PyAudio / sounddevice device index
# camera     = 0   # face client only
```

Use `--config <path>` to load a different file.

Unknown sections or keys produce a warning on stderr and are ignored.

## Running

```bash
# Push-to-talk
uv run python mcpclient_speech.py [--config path]

# Face-tracking
uv run python mcpclient_speech_face.py \
    [--config path] [--server URL] \
    [--camera N] [--mic N] \
    [--llm-model MODEL] [--llm-url URL] [-v|-vv]

# List available microphones / cameras
python hardware_devices.py -m            # microphones
python hardware_devices.py -c            # cameras
uv run python mcpclient_speech_face.py -m   # built into face client
uv run python mcpclient_speech_face.py -l   # list cameras

# Start the candytron MCP server + face client together (tmux)
./start-candytron.sh
```

The MCP server must already be running on SSE before the client starts.

## Keyboard shortcuts (window must be focused)

| Key      | Push-to-talk           | Face client                         |
| -------- | ---------------------- | ----------------------------------- |
| Ctrl     | Hold to record         | —                                   |
| Space    | Stop recording, send   | Stop listening, transcribe captured audio |
| `m`      | —                      | Toggle mute (mic + TTS)             |
| `c`      | Clear conversation     | —                                   |
| `r`      | Repeat last response   | —                                   |
| `q`      | Quit                   | Quit                                |

Pressing space in the face client while VAD is stuck on background
noise will flush the captured speech to Whisper instead of discarding
it; the LLM responds as usual. Mute (`m`) immediately silences the
microphone *and* aborts any TTS playback in progress.

## Tips

- **Background noise**: the face client uses VAD to decide when speech
  ends. Loud or constant background noise (fans, music, a busy room)
  can keep VAD open indefinitely, so the system never moves on to
  process what was said. Lowering the microphone input level in your
  operating system's audio settings usually fixes this. Press space to
  force the captured audio through immediately.
