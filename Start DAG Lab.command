#!/bin/zsh
set -eu
cd "$(dirname "$0")"
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

if [ ! -x .venv/bin/python ]; then
  echo "Preparing DAG Lab's Python environment (first launch only)…"
  python3 -m venv .venv
fi
if [ ! -f .venv/.daglab-ready ]; then
  .venv/bin/python -m pip install -r requirements-lock.txt
  .venv/bin/python -m pip install --no-deps -e .
  touch .venv/.daglab-ready
fi
if [ ! -f frontend/dist/index.html ]; then
  if ! command -v npm >/dev/null 2>&1; then
    echo "Install Node.js 20 or newer, then open this launcher again."
    read -r "?Press Return to close."
    exit 1
  fi
  echo "Building the interface (first launch only)…"
  (cd frontend && npm ci && npm run build)
fi
echo "Opening DAG Lab. Keep this window open while you work."
echo "Press Control-C here to stop the application and any active search."
exec .venv/bin/python -m daglab.cli
