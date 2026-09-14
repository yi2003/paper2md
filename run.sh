#!/usr/bin/env bash
# Start Paper2MD.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  echo "No .venv found — run ./install.sh first." >&2
  exit 1
fi

exec .venv/bin/python -m app
