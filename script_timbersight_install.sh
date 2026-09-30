#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PYTHON_BIN="${TIMBERSIGHT_PYTHON:-python3}"
VENV_DIR="${TIMBERSIGHT_VENV:-$PROJECT_DIR/.venv}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  echo "Install Python 3.11 or newer, or set TIMBERSIGHT_PYTHON." >&2
  exit 1
fi

if ! "$PYTHON_BIN" -c 'import sys; raise SystemExit(sys.version_info < (3, 11))'; then
  echo "TimberSight requires Python 3.11 or newer." >&2
  exit 1
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "Creating virtual environment: $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

echo "Installing TimberSight development, ML and web dependencies…"
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install \
  -e "${PROJECT_DIR}[dev,ml]" \
  -r "$PROJECT_DIR/requirements-web.txt"

echo
echo "Installation complete."
echo "Start TimberSight with: $PROJECT_DIR/script_timbersight_start.sh"
