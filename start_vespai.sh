#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR"
source "$SCRIPT_DIR/.venv/bin/activate"

export PYTHONPATH="/usr/lib/python3/dist-packages${PYTHONPATH:+:$PYTHONPATH}"

exec python vespai.py
