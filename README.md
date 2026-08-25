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
Anything else goes to the client, e.g. `--camera N` (face camera, `-l` lists
them), `--mic N`, `--debug-audio`.

The eye window can show an audio debug panel — VU meters (mic level, VAD
probability, end-of-utterance countdown) and separate in/out oscilloscopes.
Toggle it with `[debug] audio_panel` in `mcpclient_speech/config.toml` or
force it on with `--debug-audio`.

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
cd ../kokoro-voice-server && uv run python voice_server.py --engine kokoro-svml --voice Stina --port 8880 --whisper base
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


