# Face Agent

A camera-based agent that sees, recognizes, and talks to people. It greets
known faces, asks strangers their name, responds to speech, and says goodbye
when someone leaves. An LLM (Ollama) drives the conversation, and MCP servers
can give it extra capabilities like controlling smart home devices.

## Architecture

```
 Camera frames              Microphone audio          Speakers
      |                          |                       ^
      v                          v                       |
 +--------------+        +-------------+          +-------------+
 | face_tracker |        | voice_input |          | voice_output|
 |  detection   |        |  VAD        |          |  piper TTS  |
 |  recognition |        |  Whisper    |          +------+------+
 |  emotion     |        |  transcribe |                 |
 |  tracking    |        +------+------+                 |
 +------+-------+               |                       |
        |                       |         +------+-------+------+
        |  FaceEvents           |         | WebRTC AEC (livekit)|
        v                       v         | echo cancellation   |
 +------+-----------------------+---------+ barge-in detection  |
 |                    agent                                     |
 |                                                              |
 |  _on_face_event    _on_heard_speech     speak() w/ AEC      |
 |       |                  |                    |              |
 |       v                  v                    v              |
 |   [decide]           [respond]          [play + monitor]    |
 |       |                  |                                   |
 |       v                  v                                   |
 |     llm  (Ollama)  +  write_fact / set_name tools           |
 +---------------------------+----------------------------------+
                             |
              +--------------+--------------+
              |                             |
       +--------------+             +----------------+
       | people_memory|             | mcp_client     |
       | per-person   |             | MCP server     |
       | facts,       |             | config/loader  |
       | dialogues    |             +----------------+
       +--------------+

All modules use events.py for publish/subscribe communication.
```

## Key design decisions

### Stable person IDs

People are identified by a stable `person_id` (e.g. `p001`) that never changes,
even on rename. The face database (`known_faces/faces_arcface.pkl` for the default
InsightFace backend, `faces.pkl` for the legacy dlib backend) stores person IDs,
not names. Display names live in `people/{person_id}.json` and are the single
source of truth.

### Face recognition backend

Detection and embedding run through a pluggable backend, selected by
`[tracker] backend` in `face_config.toml`:

- **`insightface`** (default) -- SCRFD detector + ArcFace 512-d embedder on
  onnxruntime, compared with cosine distance. Far more robust at distance and
  off-angle. The `buffalo_l` model pack (~300 MB) downloads automatically to
  `~/.insightface/models` on first run. Providers auto-select CUDA → CoreML →
  CPU; for CUDA on Nvidia, install `onnxruntime-gpu` instead of `onnxruntime`.
- **`dlib`** -- the legacy HOG detector + 128-d encoder, Euclidean distance.
  Kept as a fallback / A-B comparison.

The two embedding spaces are incompatible, so each backend keeps its own db file
(`faces_arcface.pkl` vs `faces.pkl`) and they never clobber each other. To carry
existing people over to the InsightFace backend without re-meeting anyone,
re-embed the saved face crops:

```
uv run python migrate_db.py --db-dir known_faces
```

### Single facts storage

All personal information lives in `person.facts` (a flat list of strings).
No separate preferences or slots. A background tool-calling agent runs
`write_fact`, `replace_fact`, and `set_name` after each conversation turn.
Facts support replacement (e.g. "Favourite food is sushi" replaces
"Favourite food is chokladbullar").

### Canned greetings by default

Greetings use random templates ("Hi Joakim!") with no LLM call -- instant.
Pass `--smart-greeting` to enable LLM-generated greetings that reference facts.

### LLM-based name extraction

When an unknown person speaks, the LLM extracts their name (not a regex).
Handles multilingual input, messy transcription, and proper capitalization.

### WebRTC echo cancellation

`agent.speak()` uses a full-duplex `sd.Stream` with the livekit
`AudioProcessingModule` (WebRTC AEC3). The TTS render reference is fed to
the AEC, which subtracts the echo from the mic signal. The residual is
monitored for barge-in -- if a human speaks over the agent, TTS is interrupted.

## Agent decision logic

The agent is event-driven. Two inputs trigger decisions:

### Face events (from face_tracker)

- **FACE_DISAPPEARED** (known person) -- say goodbye (skipped if TTS is busy)
- **IDENTITY_CONFIRMED** (recognized face) -- greet if cooldown expired
- **FACE_APPEARED** (new track) -- if known: greet; if unknown: ask name later
- **FACE_ENGAGED** / **FACE_DISENGAGED** -- the focused face started / stopped
  looking at the camera ("makes contact"). Driven by head pose
  (`engage_max_yaw` / `engage_max_pitch`, held for `engage_dwell_seconds`);
  requires the InsightFace backend. Use FACE_ENGAGED to start an interaction
  only when someone actually faces the camera, not merely walks past.

### Speech events (from ContinuousListener)

1. Find who is in focus (primary face)
2. If unidentified, try to learn their name via LLM
3. Generate response (fast path, `reasoning_effort: none`)
4. Speak response (with AEC + barge-in monitoring)
5. Background: extract facts via tool calling

### Busy gate

The agent does one thing at a time. A `_busy` flag prevents overlapping
interactions. Goodbye is skipped if TTS is already active. A 90-second
watchdog auto-clears the flag if something hangs.

### State indicator

The camera window shows a large color-coded badge (top-right):

| State | Color | Meaning |
|-------|-------|---------|
| IDLE | gray | No one around |
| LISTENING | green | Waiting for speech |
| LISTENING... | bright green | VAD detected speech, recording |
| TRANSCRIBING | orange | Whisper processing audio |
| THINKING | deep orange | LLM generating response |
| TALKING | blue | TTS playing (with AEC) |

### Two-path LLM strategy

Response generation uses `reasoning_effort: none` for speed (~0.3-1s).
Background fact extraction uses tool calling without `reasoning_effort`
(~5-13s, thinking enabled) because Ollama requires thinking for tool calls.
The user never waits for fact extraction -- it runs in a daemon thread.

## Modules

| File | Purpose | Standalone |
|------|---------|------------|
| `agent.py` | Decision logic, AEC speak, main entry point | `pixi run agent` |
| `llm.py` | LLM calls, canned greetings, tool agents, fact extraction | -- |
| `face_tracker.py` | Detection, recognition, emotion, tracking, events | `pixi run vision` |
| `voice_input.py` | Mic monitoring, VAD, Whisper, EchoDetector (AEC) | `pixi run listen` |
| `voice_output.py` | Piper TTS with interruptible playback | `pixi run speak` |
| `people_memory.py` | Per-person storage (facts, dialogues, asked_topics) | `pixi run people` |
| `mcp_client.py` | Load MCP server configs for the LLM | -- |
| `events.py` | Shared EventDispatcher (pub/sub used by all modules) | -- |
| `main.py` | UI: camera display, overlays, keyboard controls | `pixi run run` |
| `debug_shell.py` | Interactive REPL for inspecting/editing memory and agent state | -- |
| `test_echo.py` | Standalone AEC test tool (play TTS, show echo cancellation) | `pixi run python test_echo.py` |

## Quick start

```bash
# Install dependencies
pixi install

# Pull an Ollama model (gemma4:e2b recommended for speed + tool calling)
ollama pull gemma4:e2b

# Run the agent (camera + mic + speaker)
pixi run python agent.py --llm-model gemma4:e2b --shell
```

The `--shell` flag opens an interactive debug shell alongside the agent.
Type `help` for commands, `status` for full agent/listener/TTS/AEC state.

## CLI options

```
--llm-model NAME       Ollama model (default: qwen3:8b, recommended: gemma4:e2b)
--ollama-url URL       Ollama API URL (default: http://localhost:11434/v1)
--camera N             Camera index (default: 0)
--fps N                Target frame rate (default: 15)
--agent-name NAME      Name the agent uses for itself (default: Face Agent)
--smart-greeting       Use LLM for greetings (slower, references facts)
--no-auto-greet        Don't auto-greet known faces
--no-auto-ask          Don't auto-ask unknown faces for their name
--en-voice NAME        Piper TTS voice (default: en_US-lessac-medium)
--db-dir DIR           Face database directory (default: known_faces)
--people-dir DIR       People memory directory (default: people)
--mcp-config PATH      MCP servers JSON config file
--mcp-server URL       MCP server SSE URL (repeatable)
--shell                Start interactive debug shell alongside agent
```

## Debug shell commands

```
# Memory
list                       List all known people
show <name>                Full JSON dump
facts <name>               Facts + asked topics
context <name>             LLM context block
missing <name>             Interview topics not yet asked
add-fact <name> <fact>     Add a fact
rename <old> <new>         Change display name (ID stays same)
delete <name>              Remove from memory + face DB
reset-topics <name>        Re-ask interview questions

# Agent (when running with --shell)
status                     Full state dump (agent/listener/TTS/AEC/faces)
tracks                     Visible face tracks
focus                      Currently focused face
greet <track_id>           Force a greeting
ask <track_id>             Force asking for name
speak <text>               Speak text via TTS
pause / resume             Pause/resume speech listening
```

## Data layout

```
known_faces/
  faces_arcface.pkl      InsightFace/ArcFace 512-d encodings, cosine (schema v3, default)
  faces.pkl              Legacy dlib 128-d encodings, Euclidean (schema v2, dlib backend)
  p001/                  Face images for person p001
    20260410_143022.jpg
  p002/
    ...

people/
  p001.json              Person record (name, facts, dialogues, asked_topics)
  p002.json
  ...
```

Person IDs are stable (`p001`, `p002`, ...) and never change on rename.
The `name` field inside the JSON is the display name.

## Adding MCP servers

MCP servers give the LLM access to external tools (lights, search, etc.).

### Config file

Create `mcp_servers.json`:

```json
{
    "servers": [
        {
            "name": "lights",
            "description": "Control smart home lights",
            "type": "sse",
            "url": "http://localhost:8000/sse"
        },
        {
            "name": "dirigera",
            "description": "Control IKEA smart home devices",
            "type": "stdio",
            "command": "uv",
            "args": ["--directory", "../dirigera/fastmcp", "run", "dirigeramcp.py"]
        }
    ]
}
```

### CLI flags

```bash
pixi run python agent.py --llm-model gemma4:e2b --mcp-server http://localhost:8000/sse
pixi run python agent.py --llm-model gemma4:e2b --mcp-config path/to/config.json
```

## Testing echo cancellation

```bash
# Default test (background noise only -- should NOT trigger barge-in)
pixi run python test_echo.py "Hello, this is a test."

# Custom threshold
pixi run python test_echo.py --threshold 0.05 "Hello Joakim!"

# Interactive mode
pixi run python test_echo.py --interactive
```

The test shows live `raw` (mic with echo) and `clean` (after AEC) levels.
The AEC typically reduces echo from ~0.80 RMS down to ~0.002 RMS.
