#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec uv run candytron_mcp.py --simulate-robot --simulate-camera --port 7999 "$@"
