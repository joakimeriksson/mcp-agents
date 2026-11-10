#!/usr/bin/env bash
# CandyTron 4000 launcher: a config panel (table/people camera by live picture,
# microphone with level meter, robot IP + Find robot, voice options) and a
# startup status checklist for ./start-candytron.sh.
#   ./launch-candytron.sh          # remembered settings (~/.config/candytron/launcher.json)
#   ./launch-candytron.sh --sim    # preselect simulated robot + table camera
#   ./launch-candytron.sh --tk     # the old Tk launcher window instead of the web panel
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT/mcpclient_speech"
if [[ "${1:-}" == "--tk" ]]; then
    shift
    exec uv run --no-sync python launcher.py "$@"     # the old Tk window
fi
# Web control panel on http://127.0.0.1:8090 (opens in the browser).
# --host 0.0.0.0 to reach it from a phone; --autostart; --sim.
exec uv run --no-sync python webpanel.py "$@"
