# MCP-Agents
Example code using MCPs and Agents.

## CandyTron 4000

A candy-shuffling robot demo: an MCP server (`candytron_mcp/`) controls the
robot arm + camera, and a speech client (`mcpclient_speech/`) lets you talk to
it — face recognition, multi-language speech in/out, and an LLM (via Ollama)
that calls the robot's MCP tools.

### Setup (one command)

```bash
./setup.sh
```

Idempotent — installs/verifies `uv`, `portaudio`, Ollama + `gemma4:latest`
(audio-capable: it is the chat model *and* the speech-to-text model), syncs the
three uv projects (`candytron_mcp`, `mcpclient_speech`, `face`), downloads the
Piper voices (offline fallback), clones and syncs
[kokoro-voice-server](https://github.com/joakimeriksson/kokoro-voice-server)
as a sibling checkout (`../kokoro-voice-server`, override with
`KOKORO_VOICE_SERVER_DIR`), and warms the InsightFace face model.

One thing it can only *check*: the YOLO candy-detection weights
`candytron_mcp/models/best-m.pt` are not in git — copy them from the machine
that trained them before using the real table camera.

### Run

```bash
./start-candytron.sh --sim                       # simulated robot + camera
./start-candytron.sh --robot-ip 10.10.10.10      # real Niryo Ned 2 + table camera
```

`start-candytron.sh` brings up the whole stack in order — Ollama, the
kokoro-voice-server on `:8880` (left alone if one is already running; TTS
falls back to Piper if it isn't available), `candytron_mcp` on `:7999`, then
the CandyTron eye/speech client — and Ctrl-C stops what it started.
`./start-demo.sh` is the same as `--sim`; `./stop-demo.sh` kills stragglers
(`--all` also stops the voice server).

Options: `--robot-ip IP` (or `$NIRYO_IP`), `--robot-camera N` for the table
camera, `--simulate-robot` / `--simulate-camera` individually,
`--no-voice-server`, `--face-agent` to use the face agent as the client.
Anything else goes to the client, e.g. `--camera` (people camera), `--mic N`,
`--debug-audio`.

**Cameras are picked by name, not index.** An iPhone waking up as a Continuity
Camera renumbers every index — and when it is idle it streams black frames, so
the demo looks broken. Both `--camera` and `--robot-camera` therefore accept
part of a device name, which does not move:

```bash
./start-candytron.sh --robot-camera brio --camera macbook
```

`mcpclient_speech/config.toml` defaults the people camera to `"macbook"` (the
built-in laptop camera). `-l` lists cameras with their names, `-m` the
microphones. A named device that turns out to be black is skipped and the next
match is tried.

**Direct audio** — `--direct-audio` (or `direct_audio = true` under `[llm]` in
`mcpclient_speech/config.toml`) sends the captured speech straight into gemma4:
one call does hearing + persona + live candy scene + tool calls (≈0.4 s per turn
vs ≈1 s for the STT→text path). The transcript for the history/log is produced
in the background. The classic two-step path remains the default.

The eye window can show an audio debug panel — VU meters (mic level, VAD
probability, end-of-utterance countdown) and separate in/out oscilloscopes.
Toggle it with `[debug] audio_panel` in `mcpclient_speech/config.toml` or
force it on with `--debug-audio`.

### Voice identity (who is actually talking)

With the voice server started with `--speaker ecapa` (the launcher does this),
every utterance also gets a **speaker embedding**, matched on this machine
against the visitor in focus. Once a visitor has spoken once, a sentence that
is clearly someone else's voice is dropped instead of answered — the crowd
problem the noise gate only halves. Decisions are only made on clips of about
two seconds or more; shorter ones, an unknown voice, or a server without the
endpoint all fail open (the robot keeps answering).

Privacy, by construction:

- The voice server keeps no audio and no identities; it returns a vector and
  forgets the clip. All matching happens in the client.
- Nothing is written to disk by default. Voiceprints are biometric data, so
  the store lives in memory and dies with the process.
- `persist_named = true` in `[voice_id]` keeps voices in
  `mcpclient_speech/known_voices/` (gitignored) — and only for people who
  gave a name, never provisional visitors. `./stop-demo.sh --purge` deletes it.
- Everything runs on `127.0.0.1`; no audio, embedding or transcript leaves
  the machine. The speaker model is fetched once from HuggingFace into the
  local cache; set `HF_HUB_OFFLINE=1` for a guaranteed no-network run.

Thresholds in `[voice_id]` (`confirm`, `reject`, `min_seconds`) were
calibrated on the synthetic Kokoro voices; expect to loosen `confirm` a little
after a session in a real room. Face enrolments are a separate, older store
(`face/known_faces/`) and are still written to disk by the tracker.

### Run with the Face Agent

Alternatively, drive CandyTron from the face agent — it recognizes and
remembers people, and adopts the server's CandyTron persona, name, and
init/exit lifecycle via `--service-server`:

```bash
cd candytron_mcp && uv run candytron_mcp.py --simulate-robot --simulate-camera --port 7999
cd face && uv run agent.py --llm-model gemma4 --service-server http://127.0.0.1:7999/sse
```

The standalone face agent (camera + conversation, no robot) is documented in
[face/README.md](face/README.md).

### Real robot, by hand

The same thing `start-candytron.sh` does, step by step:

```bash
cd ../kokoro-voice-server && uv run python voice_server.py --engine kokoro-svml --voice Greta --port 8880 --whisper base
cd candytron_mcp && uv run candytron_mcp.py --port 7999 --robot-ip 10.10.10.10   # add --camera N for the table camera
cd mcpclient_speech && uv run mcpclient_speech_face.py --server http://127.0.0.1:7999/sse
```

## Dirigera
Dirigera is MCP and Agents for home automation via the IKEA dirigera hub.

### Installation
You will need a Dirigera Hub and then use the dirigera library to get a token.



```bash
pip install dirigera
```


