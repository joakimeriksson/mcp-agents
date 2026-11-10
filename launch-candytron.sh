#!/usr/bin/env bash
# CandyTron 4000 launcher: a config panel (table/people camera by live picture,
# microphone with level meter, robot IP + Find robot, voice options) and a
# startup status checklist for ./start-candytron.sh.
#   ./launch-candytron.sh          # remembered settings (~/.config/candytron/launcher.json)
#   ./launch-candytron.sh --sim    # preselect simulated robot + table camera
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT/mcpclient_speech"
exec uv run --no-sync python launcher.py "$@"
