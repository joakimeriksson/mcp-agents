# MCP-Agents
Example code using MCPs and Agents.

## CandyTron 4000

A candy-shuffling robot demo: an MCP server (`candytron_mcp/`) controls the
robot arm + camera, and a speech client (`mcpclient_speech/`) lets you talk to
it — face recognition, multi-language speech in/out, and an LLM (via Ollama)
that calls the robot's MCP tools.

### Prerequisites

```bash
# 1. Ollama with a tool-calling model — we use gemma4 (set it in
#    ollama_config in mcpclient_speech/mcpclient_speech_face.py)
ollama pull gemma4

# 2. Piper TTS voices (downloads into face/piper_models/)
cd face && uv run download_models.py && cd ..
```

Optional, for the Kokoro voices: start a
[kokoro-voice-server](https://github.com/joakimeriksson/kokoro-voice-server)
on `:8880` — TTS for every Kokoro-supported language (Swedish with Stina &
friends, plus en/fr/es/it) then goes through it, with automatic fallback to
Piper if it isn't running. German has no Kokoro voice and always uses Piper.
Configured in `face/languages.toml`.

```bash
git clone https://github.com/joakimeriksson/kokoro-voice-server
cd kokoro-voice-server && uv sync
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python voice_server.py \
    --engine kokoro-svml --voice Stina --port 8880
```

### Run the demo (simulated robot)

```bash
./start-demo.sh
```

This starts `candytron_mcp` on port 7999 with `--simulate-robot
--simulate-camera` and then the speech-face client. Stop everything with
Ctrl-C (or `./stop-demo.sh`).

If the wrong camera opens, pick another with `FACE_CAMERA_INDEX=<n>`
(the client defaults to index 4):

```bash
FACE_CAMERA_INDEX=0 ./start-demo.sh
```

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

### Real robot

Start the server yourself without the simulation flags, then either client:

```bash
cd candytron_mcp && uv run candytron_mcp.py --port 7999
cd mcpclient_speech && uv run mcpclient_speech_face.py
```

## Dirigera
Dirigera is MCP and Agents for home automation via the IKEA dirigera hub.

### Installation
You will need a Dirigera Hub and then use the dirigera library to get a token.



```bash
pip install dirigera
```


