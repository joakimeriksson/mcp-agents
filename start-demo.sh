#!/usr/bin/env bash
# Simulated CandyTron 4000 demo (no robot, no table camera).
# Thin wrapper: ./start-candytron.sh --sim  — see that script for all options.
exec "$(cd "$(dirname "$0")" && pwd)/start-candytron.sh" --sim "$@"
